from fastapi import (
    APIRouter, HTTPException, Query, UploadFile, File as FastAPIFile, Form, Depends
)
from typing import List, Optional
from uuid import uuid4
from datetime import datetime
import sys
import os
from pathlib import Path
import traceback
from loguru import logger

# Add parent directory to path for celery_config import
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from .database import get_database_service, DatabaseService
from .models import File, ApiResponse

# Import celery tasks
from celery_tasks import (
    batch_process_files_task,
    get_task_status,
    cancel_task,
    start_file_processing_pipeline
)

router = APIRouter()

@router.get("/folder/{folder_id}", response_model=List[File])
async def get_files_by_folder(
    folder_id: str, 
    user_id: str = Query(..., description="User ID"),
    db: DatabaseService = Depends(get_database_service)
):
    """Get all files in a specific folder with processing status"""
    try:
        # Get files with markdown content and processing status
        response = await (
            db.client.from_("files")
            .select("*, markdown_content(word_count, created_at)")
            .eq("folder_id", folder_id)
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        if not response.data:
            return []

        # Enrich file data with processing information
        files = []
        for file_data in response.data:
            # Add markdown info if available
            markdown_info = file_data.get("markdown_content")
            if markdown_info:
                file_data["word_count"] = markdown_info[0].get("word_count") if markdown_info else None
                file_data["markdown_created_at"] = markdown_info[0].get("created_at") if markdown_info else None
            
            files.append(file_data)

        return files

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching files: {str(e)}"
        )


@router.post("/upload", response_model=ApiResponse)
async def upload_file(
    file: UploadFile = FastAPIFile(...),
    folder_id: str = Form(...),
    user_id: str = Form(...),
    auto_process: bool = Form(True),
    db: DatabaseService = Depends(get_database_service)
):
    """Upload a file with enhanced error handling and logging"""
    logger.info(f"Upload request: user_id={user_id}, folder_id={folder_id}, filename={file.filename}")
    
    try:
        # Step 1: Check database service initialization
        logger.info("Checking database service initialization...")
        if not db._initialized:
            logger.error("Database service not initialized!")
            raise HTTPException(status_code=500, detail="Database service not initialized")
        
        # Step 2: Verify folder exists and belongs to user
        logger.info(f"Verifying folder {folder_id} for user {user_id}")
        folder_response = await db.client.from_("folders").select("*").eq("id", folder_id).eq("user_id", user_id).execute()
        
        if not folder_response.data:
            logger.error(f"Folder {folder_id} not found for user {user_id}")
            raise HTTPException(status_code=404, detail="Folder not found")
        
        logger.info("✅ Folder verification successful")
        
        # Step 3: Generate file metadata
        file_id = str(uuid4())
        file_extension = ""
        if file.filename:
            file_extension = os.path.splitext(file.filename)[1]
        storage_path = f"{user_id}/{folder_id}/{file_id}{file_extension}"
        
        logger.info(f"Generated file_id: {file_id}, storage_path: {storage_path}")
        
        # Step 4: Read file content
        logger.info("Reading file content...")
        file_content = await file.read()
        logger.info(f"File content size: {len(file_content)} bytes")
        
        # Step 5: Upload to Supabase storage
        logger.info("Uploading to Supabase storage...")
        try:
            # AWAIT the storage upload since it's async
            upload_result = await db.service.storage.from_("documents").upload(
                storage_path,
                file_content,
                {
                    "content-type": file.content_type or "application/octet-stream",
                    "x-upsert": "true"
                }
            )
            logger.info(f"✅ Storage upload successful: {upload_result}")
            
            # AWAIT the get_public_url call since it's async
            storage_url_response = await db.service.storage.from_("documents").get_public_url(storage_path)
            
            # Handle the response properly
            if hasattr(storage_url_response, 'public_url'):
                public_url = storage_url_response.public_url
            elif isinstance(storage_url_response, dict):
                public_url = storage_url_response.get("publicURL") or storage_url_response.get("public_url")
            else:
                public_url = str(storage_url_response)
                
            logger.info(f"Generated public URL: {public_url}")
            
        except Exception as storage_error:
            logger.error(f"❌ Storage upload failed: {storage_error}")
            logger.error(f"Storage error traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload file to storage: {str(storage_error)}"
            )
        
        # Step 6: Create database record
        logger.info("Creating database record...")
        file_data = {
            "id": file_id,
            "user_id": user_id,
            "folder_id": folder_id,
            "original_filename": file.filename or "unknown",
            "file_size": len(file_content),
            "file_type": file.content_type or "application/octet-stream",
            "storage_path": storage_path,
            "storage_url": public_url,
            "status": "uploaded",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        try:
            db_response = await db.client.from_("files").insert(file_data).execute()
            if not db_response.data:
                logger.error("Database insert returned no data")
                raise HTTPException(status_code=400, detail="Failed to create file record")
            logger.info("✅ Database record created successfully")
        except Exception as db_error:
            logger.error(f"❌ Database insert failed: {db_error}")
            logger.error(f"Database error traceback: {traceback.format_exc()}")
            # Clean up storage on DB failure
            try:
                db.service.storage.from_("documents").remove([storage_path])
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
        
        # Step 7: Start background processing
        task_id = None
        job_id = None
        if auto_process:
            logger.info("Starting background processing...")
            try:
                # Update file status to 'queued'
                await db.client.from_("files").update({
                    "status": "queued",
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", file_id).execute()
                
                # Start processing pipeline
                task_result = start_file_processing_pipeline(file_id)
                task_id = task_result.get("chain_id")
                job_id = task_result.get("job_id")
                logger.info(f"✅ Background processing started: task_id={task_id}, job_id={job_id}")
            except Exception as celery_error:
                logger.error(f"❌ Celery processing failed: {celery_error}")
                # Don't fail the upload for Celery issues, just log
                logger.warning("File uploaded but background processing failed to start")
        
        logger.info(f"✅ Upload completed successfully: file_id={file_id}")
        return ApiResponse(
            success=True,
            message="File uploaded successfully and processing pipeline started.",
            data={
                "file_id": file_id,
                "task_id": task_id,
                "job_id": job_id,
                "storage_url": public_url,
                "status": "queued" if auto_process else "uploaded"
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors with full traceback
        logger.error(f"❌ Unexpected error in file upload: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/process/{file_id}", response_model=ApiResponse)
async def start_file_processing(
    file_id: str,
    user_id: str = Query(..., description="User ID"),
    db: DatabaseService = Depends(get_database_service)
):
    """
    Manually start or restart the full processing pipeline for a specific file.
    """
    try:
        # Verify file exists and belongs to user
        file_response = await db.client.from_("files").select(
            "storage_path, original_filename"
        ).eq("id", file_id).eq("user_id", user_id).single().execute()

        if not file_response.data:
            raise HTTPException(status_code=404, detail="File not found")

        # Clear any previous errors and set status to queued
        await db.client.from_("files").update({
            "status": "queued",
            "error_message": None,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", file_id).execute()

        # Start the full processing pipeline using the chain
        task_result = start_file_processing_pipeline(file_id)
        task_id = task_result.get("chain_id")
        job_id = task_result.get("job_id")

        return ApiResponse(
            success=True,
            message="File processing pipeline started successfully.",
            data={"file_id": file_id, "task_id": task_id, "job_id": job_id, "status": "queued"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error starting file processing: {str(e)}"
        )

@router.post("/batch-process", response_model=ApiResponse)
async def batch_process_files(
    folder_id: str = Form(...),
    user_id: str = Form(...),
    process_type: str = Form("full"),
    file_ids: Optional[List[str]] = Form(None),  # If None, process all files in folder
    db: DatabaseService = Depends(get_database_service)
):
    """
    Start batch processing for multiple files in a folder
    """
    try:
        # Verify folder exists and belongs to user
        folder_response = await (
            db.client.from_("folders")
            .select("*")
            .eq("id", folder_id)
            .eq("user_id", user_id)
            .execute()
        )

        if not folder_response.data:
            raise HTTPException(status_code=404, detail="Folder not found")

        # Get file IDs to process
        if file_ids is None:
            # Process all files in folder
            files_response = await (
                db.client.from_("files")
                .select("id")
                .eq("folder_id", folder_id)
                .eq("user_id", user_id)
                .execute()
            )
            file_ids = [f["id"] for f in files_response.data] if files_response.data else []

        if not file_ids:
            raise HTTPException(status_code=400, detail="No files found to process")

        # Start batch processing task
        task_result = batch_process_files_task.delay(
            file_ids, user_id, process_type
        )

        return ApiResponse(
            success=True,
            message=f"Batch processing started for {len(file_ids)} files",
            data={
                "batch_task_id": task_result.id,
                "file_count": len(file_ids),
                "process_type": process_type,
                "folder_id": folder_id
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error starting batch processing: {str(e)}"
        )

@router.get("/task/{task_id}/status")
async def get_task_status_endpoint(task_id: str):
    """Get the status of a processing task"""
    try:
        status = get_task_status(task_id)
        return status
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting task status: {str(e)}"
        )

@router.delete("/task/{task_id}/cancel")
async def cancel_task_endpoint(task_id: str):
    """Cancel a running processing task"""
    try:
        result = cancel_task(task_id)
        return {"message": result}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error cancelling task: {str(e)}"
        )

@router.get("/processing-stats/{user_id}")
async def get_processing_stats(
    user_id: str,
    db: DatabaseService = Depends(get_database_service)
):
    """Get processing statistics for a user"""
    try:
        # Get file counts by status
        all_files = await db.client.from_("files").select("status").eq("user_id", user_id).execute()
        
        status_counts = {}
        if all_files.data:
            for file_data in all_files.data:
                status = file_data.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1

        # Get markdown content stats
        markdown_stats = await db.client.from_("markdown_content").select("word_count").eq("user_id", user_id).execute()
        total_words = sum(item.get("word_count", 0) for item in markdown_stats.data) if markdown_stats.data else 0

        # Get clause counts
        clause_stats = await db.client.from_("clause_library").select("id", count="exact").eq("user_id", user_id).execute()
        total_clauses = clause_stats.count or 0

        return {
            "user_id": user_id,
            "file_status_counts": status_counts,
            "total_files": len(all_files.data) if all_files.data else 0,
            "total_words_processed": total_words,
            "total_clauses_extracted": total_clauses,
            "processing_pipeline_stages": {
                "uploaded": status_counts.get("uploaded", 0),
                "queued": status_counts.get("queued", 0),
                "converting": status_counts.get("converting", 0),
                "markdown_ready": status_counts.get("markdown_ready", 0),
                "extracting_metadata": status_counts.get("extracting_metadata", 0),
                "metadata_ready": status_counts.get("metadata_ready", 0),
                "extracting_clauses": status_counts.get("extracting_clauses", 0),
                "clauses_ready": status_counts.get("clauses_ready", 0),
                "processed": status_counts.get("processed", 0),
                "error": status_counts.get("error", 0)
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting processing stats: {str(e)}"
        )

@router.delete("/{file_id}", response_model=ApiResponse)
async def delete_file(
    file_id: str, 
    user_id: str = Query(..., description="User ID"),
    db: DatabaseService = Depends(get_database_service)
):
    """Delete a file and all its related data"""
    try:
        # Get file info first
        file_response = await (
            db.client.from_("files")
            .select("*")
            .eq("id", file_id)
            .eq("user_id", user_id)
            .execute()
        )

        if not file_response.data:
            raise HTTPException(status_code=404, detail="File not found")

        file_data = file_response.data[0]

        # Delete from storage if exists
        if file_data.get("storage_path"):
            try:
                db.service.storage.from_("documents").remove([
                    file_data["storage_path"]
                ])
            except Exception as storage_error:
                print(f"Warning: Failed to delete file from storage: {storage_error}")

        # Delete related records (will cascade due to foreign keys)
        # markdown_content, clause_library will be deleted automatically
        
        # Delete file record from database
        await db.client.from_("files").delete().eq(
            "id", file_id
        ).eq("user_id", user_id).execute()

        return ApiResponse(
            success=True,
            message="File and all related data deleted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting file: {str(e)}"
        )

@router.get("/{file_id}")
async def get_file_by_id(
    file_id: str, 
    user_id: str = Query(..., description="User ID"),
    include_content: bool = Query(False, description="Include markdown content"),
    db: DatabaseService = Depends(get_database_service)
):
    """Get a specific file by ID with optional content"""
    try:
        select_fields = "*, markdown_content(word_count, created_at)"
        if include_content:
            select_fields = "*, markdown_content(*)"

        response = await (
            db.client.from_("files")
            .select(select_fields)
            .eq("id", file_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="File not found")

        file_data = response.data
        
        # Add processing statistics
        markdown_info = file_data.get("markdown_content")
        if markdown_info:
            file_data["word_count"] = markdown_info[0].get("word_count") if markdown_info else None
            file_data["markdown_created_at"] = markdown_info[0].get("created_at") if markdown_info else None
            if include_content:
                file_data["markdown_content_text"] = markdown_info[0].get("content") if markdown_info else None

        return file_data

    except Exception as e:
        if "No rows" in str(e) or "PGRST116" in str(e):
            raise HTTPException(status_code=404, detail="File not found")
        raise HTTPException(
            status_code=500, detail=f"Error fetching file: {str(e)}"
        )

@router.get("/{file_id}/markdown")
async def get_file_markdown(
    file_id: str,
    user_id: str = Query(..., description="User ID"),
    db: DatabaseService = Depends(get_database_service)
):
    """Get markdown content for a specific file"""
    try:
        # Verify file belongs to user
        file_check = await (
            db.client.from_("files")
            .select("id")
            .eq("id", file_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        if not file_check.data:
            raise HTTPException(status_code=404, detail="File not found")

        # Get markdown content
        markdown_response = await (
            db.client.from_("markdown_content")
            .select("*")
            .eq("file_id", file_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        if not markdown_response.data:
            raise HTTPException(status_code=404, detail="Markdown content not found")

        return markdown_response.data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching markdown content: {str(e)}"
        )

@router.get("/{file_id}/clauses")
async def get_file_clauses(
    file_id: str,
    user_id: str = Query(..., description="User ID"),
    clause_type: Optional[str] = Query(None, description="Filter by clause type"),
    template_id: Optional[str] = Query(None, description="Filter by template ID"),
    db: DatabaseService = Depends(get_database_service)
):
    """Get extracted clauses for a specific file"""
    try:
        # Verify file belongs to user
        file_check = await (
            db.client.from_("files")
            .select("id")
            .eq("id", file_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        if not file_check.data:
            raise HTTPException(status_code=404, detail="File not found")

        # Build query for clauses
        query = (
            db.client.from_("clause_library")
            .select("*")
            .eq("file_id", file_id)  # Changed from source_file_id to file_id
            .eq("user_id", user_id)
        )

        if clause_type:
            query = query.eq("clause_type", clause_type)
        
        if template_id:
            query = query.eq("template_id", template_id)

        clauses_response = await query.execute()

        return {
            "file_id": file_id,
            "clauses": clauses_response.data or [],
            "total_clauses": len(clauses_response.data) if clauses_response.data else 0
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching clauses: {str(e)}"
        )

@router.patch("/{file_id}/status", response_model=ApiResponse)
async def update_file_status(
    file_id: str,
    status: str = Form(...),
    user_id: str = Query(..., description="User ID"),
    error_message: str = Form(None),
    db: DatabaseService = Depends(get_database_service)
):
    """Update file processing status (for manual intervention)"""
    try:
        # Verify file exists and belongs to user
        file_response = await (
            db.client.from_("files")
            .select("*")
            .eq("id", file_id)
            .eq("user_id", user_id)
            .execute()
        )

        if not file_response.data:
            raise HTTPException(status_code=404, detail="File not found")

        # Valid status values
        valid_statuses = [
            "uploaded", "queued", "converting", "markdown_ready",
            "extracting_metadata", "metadata_ready", "extracting_clauses",
            "clauses_ready", "processed", "template_ready", "error"
        ]

        if status not in valid_statuses:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )

        # Update data
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }

        if status == "processed":
            update_data["processed_at"] = datetime.utcnow().isoformat()

        if error_message:
            update_data["error_message"] = error_message
        elif status != "error":
            update_data["error_message"] = None  # Clear error message for non-error statuses

        # Update file status
        await db.client.from_("files").update(update_data).eq(
            "id", file_id
        ).eq("user_id", user_id).execute()

        return ApiResponse(
            success=True,
            message=f"File status updated to {status}"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating file status: {str(e)}"
        )

@router.get("/health/pipeline")
async def check_pipeline_health():
    """Health check for the processing pipeline"""
    try:
        # Import here to avoid circular imports
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from celery_tasks import health_check
        
        try:
            task_result = health_check.delay()
            # Wait briefly for result
            result = task_result.get(timeout=5)
            celery_status = "healthy"
            celery_error = None
        except Exception as e:
            celery_status = "unhealthy"
            celery_error = str(e)

        # Test database connection
        try:
            db = get_database_service()
            await db.client.from_("files").select("id").limit(1).execute()
            db_status = "healthy"
            db_error = None
        except Exception as e:
            db_status = "unhealthy"
            db_error = str(e)

        # Test storage connection
        try:
            db = get_database_service()
            db.service.storage.from_("documents").list(limit=1)
            storage_status = "healthy"
            storage_error = None
        except Exception as e:
            storage_status = "unhealthy"
            storage_error = str(e)

        overall_status = "healthy" if all([
            celery_status == "healthy",
            db_status == "healthy", 
            storage_status == "healthy"
        ]) else "unhealthy"

        return {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "celery": {
                    "status": celery_status,
                    "error": celery_error
                },
                "database": {
                    "status": db_status,
                    "error": db_error
                },
                "storage": {
                    "status": storage_status,
                    "error": storage_error
                }
            }
        }

    except Exception as e:
        return {
            "overall_status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }