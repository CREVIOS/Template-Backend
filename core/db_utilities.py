"""
Database utility functions for reuse across the application.
Centralizes common database operations to avoid code duplication.
"""

# todo organize this file, bring everything into the utility folder, and make it a proper module, use them in the other files

from typing import List, Dict, Any, Optional, Union
from uuid import uuid4
from datetime import datetime
from loguru import logger
from .database import get_database_service, DatabaseService

logger.add("logs/db_utilities.log", rotation="10 MB", level="DEBUG")

# ============================================================================
# FILE OPERATIONS
# ============================================================================

async def get_file_by_id(file_id: str, user_id: str, include_content: bool = False) -> Optional[Dict[str, Any]]:
    """Get a file by ID with optional markdown content"""
    try:
        db = get_database_service()
        select_fields = "*, markdown_content(word_count, created_at)"
        if include_content:
            select_fields = "*, markdown_content(*)"

        response = await db.client.from_("files").select(select_fields).eq("id", file_id).eq("user_id", user_id).single().execute()
        
        if not response.data:
            return None
            
        file_data = response.data
        
        # Enrich with markdown info
        markdown_info = file_data.get("markdown_content")
        if markdown_info and len(markdown_info) > 0:
            file_data["word_count"] = markdown_info[0].get("word_count")
            file_data["markdown_created_at"] = markdown_info[0].get("created_at")
            if include_content:
                file_data["markdown_content_text"] = markdown_info[0].get("content")
        
        return file_data
        
    except Exception as e:
        logger.error(f"Error getting file {file_id}: {e}")
        return None

async def get_files_by_folder(folder_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Get all files in a folder"""
    try:
        db = get_database_service()
        response = await db.client.from_("files").select("*, markdown_content(word_count, created_at)").eq("folder_id", folder_id).eq("user_id", user_id).order("created_at", desc=True).execute()
        
        if not response.data:
            return []
            
        # Enrich file data with processing information
        files = []
        for file_data in response.data:
            markdown_info = file_data.get("markdown_content")
            if markdown_info and len(markdown_info) > 0:
                file_data["word_count"] = markdown_info[0].get("word_count")
                file_data["markdown_created_at"] = markdown_info[0].get("created_at")
            files.append(file_data)
            
        return files
        
    except Exception as e:
        logger.error(f"Error getting files for folder {folder_id}: {e}")
        return []

async def update_file_status(file_id: str, status: str, user_id: str, error_message: Optional[str] = None) -> bool:
    """Update file processing status"""
    try:
        db = get_database_service()
        
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if status == "processed":
            update_data["processed_at"] = datetime.utcnow().isoformat()
            
        if error_message:
            update_data["error_message"] = error_message
        elif status != "error":
            update_data["error_message"] = None
            
        await db.client.from_("files").update(update_data).eq("id", file_id).eq("user_id", user_id).execute()
        logger.debug(f"Updated file {file_id} status to {status}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating file {file_id} status: {e}")
        return False

async def delete_file_with_cleanup(file_id: str, user_id: str) -> bool:
    """Delete file and all related data with storage cleanup"""
    try:
        db = get_database_service()
        
        # Get file info first
        file_data = await get_file_by_id(file_id, user_id)
        if not file_data:
            return False
            
        # Delete from storage if exists
        if file_data.get("storage_path"):
            try:
                await db.service.storage.from_("documents").remove([file_data["storage_path"]])
            except Exception as storage_error:
                logger.warning(f"Failed to delete file from storage: {storage_error}")
                
        # Delete file record (cascades to related tables)
        await db.client.from_("files").delete().eq("id", file_id).eq("user_id", user_id).execute()
        logger.info(f"Deleted file {file_id} and related data")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {e}")
        return False

async def create_file_record(file_data: Dict[str, Any]) -> Optional[str]:
    """Create a new file record in the database"""
    try:
        db = get_database_service()
        
        file_record = {
            "id": file_data.get("id") or str(uuid4()),
            "user_id": file_data["user_id"],
            "folder_id": file_data["folder_id"],
            "original_filename": file_data["original_filename"],
            "file_size": file_data["file_size"],
            "file_type": file_data["file_type"],
            "storage_path": file_data["storage_path"],
            "storage_url": file_data["storage_url"],
            "status": file_data.get("status", "uploaded"),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("files").insert(file_record).execute()
        if response.data:
            logger.info(f"Created file record {file_record['id']}")
            return file_record["id"]
        return None
        
    except Exception as e:
        logger.error(f"Error creating file record: {e}")
        return None

# ============================================================================
# FOLDER OPERATIONS  
# ============================================================================

async def get_folder_by_id(folder_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get folder by ID"""
    try:
        db = get_database_service()
        response = await db.client.from_("folders").select("*").eq("id", folder_id).eq("user_id", user_id).single().execute()
        return response.data if response.data else None
        
    except Exception as e:
        logger.error(f"Error getting folder {folder_id}: {e}")
        return None

async def get_folders_by_user(user_id: str) -> List[Dict[str, Any]]:
    """Get all folders for a user"""
    try:
        db = get_database_service()
        response = await db.client.from_("folders").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting folders for user {user_id}: {e}")
        return []

async def create_folder(name: str, user_id: str, color: str = "blue") -> Optional[str]:
    """Create a new folder"""
    try:
        db = get_database_service()
        
        folder_data = {
            "id": str(uuid4()),
            "name": name,
            "user_id": user_id,
            "color": color,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("folders").insert(folder_data).execute()
        if response.data:
            logger.info(f"Created folder {folder_data['id']}")
            return folder_data["id"]
        return None
        
    except Exception as e:
        logger.error(f"Error creating folder: {e}")
        return None

async def update_folder(folder_id: str, user_id: str, updates: Dict[str, Any]) -> bool:
    """Update folder data"""
    try:
        db = get_database_service()
        
        updates["updated_at"] = datetime.utcnow().isoformat()
        await db.client.from_("folders").update(updates).eq("id", folder_id).eq("user_id", user_id).execute()
        logger.debug(f"Updated folder {folder_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating folder {folder_id}: {e}")
        return False

async def delete_folder(folder_id: str, user_id: str) -> bool:
    """Delete folder and all its files"""
    try:
        db = get_database_service()
        
        # Get all files in folder first for cleanup
        files = await get_files_by_folder(folder_id, user_id)
        
        # Delete all files in folder
        for file_data in files:
            await delete_file_with_cleanup(file_data["id"], user_id)
            
        # Delete folder
        await db.client.from_("folders").delete().eq("id", folder_id).eq("user_id", user_id).execute()
        logger.info(f"Deleted folder {folder_id} and {len(files)} files")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting folder {folder_id}: {e}")
        return False

# ============================================================================
# TEMPLATE OPERATIONS
# ============================================================================

async def get_template_by_id(template_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get template by ID with folder info"""
    try:
        db = get_database_service()
        response = await db.client.from_("templates").select("*, folders(name, color, user_id)").eq("id", template_id).single().execute()
        
        if not response.data:
            return None
            
        template_data = response.data
        folder = template_data.get("folders", {})
        
        # Check user access through folder
        if folder.get("user_id") != user_id:
            return None
            
        return template_data
        
    except Exception as e:
        logger.error(f"Error getting template {template_id}: {e}")
        return None

async def get_templates_by_user(user_id: str, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all templates for a user, optionally filtered by folder"""
    try:
        db = get_database_service()
        
        query = db.client.from_("templates").select("*, folders(name, color), template_usage_stats(action_type, created_at)").eq("folders.user_id", user_id)
        
        if folder_id:
            query = query.eq("folder_id", folder_id)
            
        response = await query.order("created_at", desc=True).execute()
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting templates for user {user_id}: {e}")
        return []

async def save_template(template_data: Dict[str, Any]) -> Optional[str]:
    """Save a new template to the database"""
    try:
        db = get_database_service()
        
        template_record = {
            "id": template_data.get("id") or str(uuid4()),
            "folder_id": template_data["folder_id"],
            "name": template_data["name"],
            "content": template_data["content"],
            "template_type": template_data.get("template_type", "general"),
            "file_extension": template_data.get("file_extension", ".docx"),
            "formatting_data": template_data.get("formatting_data", {}),
            "content_json": template_data.get("content_json"),
            "word_compatible": template_data.get("word_compatible", True),
            "is_active": template_data.get("is_active", True),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("templates").insert(template_record).execute()
        if response.data:
            logger.info(f"Saved template {template_record['id']}")
            return template_record["id"]
        return None
        
    except Exception as e:
        logger.error(f"Error saving template: {e}")
        return None

async def update_template(template_id: str, user_id: str, updates: Dict[str, Any]) -> bool:
    """Update template data"""
    try:
        db = get_database_service()
        
        # Verify user has access to template
        template = await get_template_by_id(template_id, user_id)
        if not template:
            return False
            
        updates["updated_at"] = datetime.utcnow().isoformat()
        await db.client.from_("templates").update(updates).eq("id", template_id).execute()
        logger.debug(f"Updated template {template_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating template {template_id}: {e}")
        return False

# ============================================================================
# MARKDOWN CONTENT OPERATIONS
# ============================================================================

async def save_markdown_content(file_id: str, user_id: str, content: str, word_count: int) -> bool:
    """Save markdown content for a file"""
    try:
        db = get_database_service()
        
        markdown_data = {
            "file_id": file_id,
            "user_id": user_id,
            "content": content,
            "word_count": word_count,
            "created_at": datetime.utcnow().isoformat()
        }
        
        await db.client.from_("markdown_content").insert(markdown_data).execute()
        logger.debug(f"Saved markdown content for file {file_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving markdown content for file {file_id}: {e}")
        return False

async def get_markdown_content(file_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get markdown content for a file"""
    try:
        db = get_database_service()
        response = await db.client.from_("markdown_content").select("*").eq("file_id", file_id).eq("user_id", user_id).single().execute()
        return response.data if response.data else None
        
    except Exception as e:
        logger.error(f"Error getting markdown content for file {file_id}: {e}")
        return None

# ============================================================================
# CLAUSE LIBRARY OPERATIONS
# ============================================================================

async def save_clause_to_library(clause_data: Dict[str, Any]) -> bool:
    """Save a clause to the clause library"""
    try:
        db = get_database_service()
        
        clause_record = {
            "id": clause_data.get("id") or str(uuid4()),
            "user_id": clause_data["user_id"],
            "file_id": clause_data["file_id"],
            "folder_id": clause_data["folder_id"],
            "clause_type": clause_data["clause_type"],
            "clause_text": clause_data["clause_text"],
            "clause_metadata": clause_data.get("clause_metadata", {}),
            "created_at": datetime.utcnow().isoformat()
        }
        
        await db.client.from_("clause_library").insert(clause_record).execute()
        logger.debug(f"Saved clause to library for file {clause_data['file_id']}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving clause to library: {e}")
        return False

async def get_clauses_by_file(file_id: str, user_id: str, clause_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all clauses for a specific file"""
    try:
        db = get_database_service()
        
        query = db.client.from_("clause_library").select("*").eq("file_id", file_id).eq("user_id", user_id)
        
        if clause_type:
            query = query.eq("clause_type", clause_type)
            
        response = await query.execute()
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting clauses for file {file_id}: {e}")
        return []

async def get_clauses_by_template(template_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Get all clauses associated with a template"""
    try:
        db = get_database_service()
        response = await db.client.from_("clause_library").select("*").eq("template_id", template_id).eq("user_id", user_id).execute()
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting clauses for template {template_id}: {e}")
        return []

# ============================================================================
# JOB MANAGEMENT OPERATIONS
# ============================================================================

async def create_job(user_id: str, job_type: str, metadata: Optional[Dict[str, Any]] = None, total_steps: int = 1) -> Optional[str]:
    """Create a new job record"""
    try:
        db = get_database_service()
        
        job_data = {
            "id": str(uuid4()),
            "user_id": user_id,
            "job_type": job_type,
            "status": "pending",
            "total_steps": total_steps,
            "current_step": 0,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("jobs").insert(job_data).execute()
        if response.data:
            logger.info(f"Created job {job_data['id']} of type {job_type}")
            return job_data["id"]
        return None
        
    except Exception as e:
        logger.error(f"Error creating job: {e}")
        return None

async def update_job_status(job_id: str, status: str, error_message: Optional[str] = None, result: Optional[Dict[str, Any]] = None) -> bool:
    """Update job status"""
    try:
        db = get_database_service()
        
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if status in ["completed", "failed"]:
            update_data["completed_at"] = datetime.utcnow().isoformat()
            
        if error_message:
            update_data["error_message"] = error_message
            
        if result:
            update_data["result"] = result
            
        await db.client.from_("jobs").update(update_data).eq("id", job_id).execute()
        logger.debug(f"Updated job {job_id} status to {status}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating job {job_id} status: {e}")
        return False

async def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job status and details"""
    try:
        db = get_database_service()
        response = await db.client.from_("jobs").select("*").eq("id", job_id).single().execute()
        return response.data if response.data else None
        
    except Exception as e:
        logger.error(f"Error getting job {job_id} status: {e}")
        return None

# ============================================================================
# USAGE TRACKING OPERATIONS
# ============================================================================

async def track_template_usage(template_id: str, user_id: str, action_type: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Track template usage for analytics"""
    try:
        db = get_database_service()
        
        usage_data = {
            "template_id": template_id,
            "user_id": user_id,
            "action_type": action_type,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        await db.client.from_("template_usage_stats").insert(usage_data).execute()
        logger.debug(f"Tracked template usage: {action_type} for template {template_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error tracking template usage: {e}")
        return False

async def get_processing_stats(user_id: str) -> Dict[str, Any]:
    """Get comprehensive processing statistics for a user"""
    try:
        db = get_database_service()
        
        # Get file counts by status
        files_response = await db.client.from_("files").select("status").eq("user_id", user_id).execute()
        
        status_counts = {}
        if files_response.data:
            for file_data in files_response.data:
                status = file_data.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                
        # Get markdown stats
        markdown_response = await db.client.from_("markdown_content").select("word_count").eq("user_id", user_id).execute()
        total_words = sum(item.get("word_count", 0) for item in markdown_response.data) if markdown_response.data else 0
        
        # Get clause counts
        clause_response = await db.client.from_("clause_library").select("id", count="exact").eq("user_id", user_id).execute()
        total_clauses = clause_response.count or 0
        
        # Get template counts
        template_response = await db.client.from_("templates").select("id", count="exact").eq("folders.user_id", user_id).execute()
        total_templates = template_response.count or 0
        
        return {
            "user_id": user_id,
            "file_status_counts": status_counts,
            "total_files": len(files_response.data) if files_response.data else 0,
            "total_words_processed": total_words,
            "total_clauses_extracted": total_clauses,
            "total_templates": total_templates,
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
        logger.error(f"Error getting processing stats for user {user_id}: {e}")
        return {
            "user_id": user_id,
            "error": str(e),
            "file_status_counts": {},
            "total_files": 0,
            "total_words_processed": 0,
            "total_clauses_extracted": 0,
            "total_templates": 0,
            "processing_pipeline_stages": {}
        }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def check_user_access_to_file(file_id: str, user_id: str) -> bool:
    """Check if user has access to a specific file"""
    file_data = await get_file_by_id(file_id, user_id)
    return file_data is not None

async def check_user_access_to_folder(folder_id: str, user_id: str) -> bool:
    """Check if user has access to a specific folder"""
    folder_data = await get_folder_by_id(folder_id, user_id)
    return folder_data is not None

async def check_user_access_to_template(template_id: str, user_id: str) -> bool:
    """Check if user has access to a specific template"""
    template_data = await get_template_by_id(template_id, user_id)
    return template_data is not None

async def cleanup_orphaned_records(user_id: str) -> Dict[str, int]:
    """Clean up orphaned records for a user"""
    try:
        db = get_database_service()
        cleanup_counts = {"files": 0, "markdown": 0, "clauses": 0}
        
        # Find files without folders
        files_response = await db.client.from_("files").select("id, folder_id").eq("user_id", user_id).execute()
        
        if files_response.data:
            for file_data in files_response.data:
                folder_exists = await check_user_access_to_folder(file_data["folder_id"], user_id)
                if not folder_exists:
                    await delete_file_with_cleanup(file_data["id"], user_id)
                    cleanup_counts["files"] += 1
                    
        logger.info(f"Cleaned up {cleanup_counts['files']} orphaned files for user {user_id}")
        return cleanup_counts
        
    except Exception as e:
        logger.error(f"Error cleaning up orphaned records for user {user_id}: {e}")
        return {"files": 0, "markdown": 0, "clauses": 0}