from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from loguru import logger
import json

from core.database import get_database_service, DatabaseService
from core.api_config import APIConfiguration
# from core.template_generator import TemplateGenerator
from core.document_exporter import DocumentExporter
from core.models import (
    TemplatesResponse,
    TemplatePreviewResponse,
    TemplateWithDetails,
    TemplatePreview,
    ApiResponse
)
from celery_tasks import template_generation_task
from core.db_utilities import create_job, get_job_status as get_job_status_util

logger.add("logs/template.log", rotation="10 MB", level="DEBUG")
router = APIRouter(tags=["template"])

# ============================================================================
# DEPENDENCIES
# ============================================================================

def get_api_config() -> APIConfiguration:
    """Get API configuration instance"""
    return APIConfiguration()

def get_document_exporter() -> DocumentExporter:
    """Get document exporter instance"""
    return DocumentExporter()

# ============================================================================
# TEMPLATE LISTING & RETRIEVAL
# ============================================================================

@router.get("/", response_model=TemplatesResponse)
async def get_templates(
    user_id: str = Query(...),
    sort_field: str = Query(
        "name",
        pattern="^(name|folder_name|files_count|last_action_type|last_action_date)$"
    ),
    sort_direction: str = Query("asc", pattern="^(asc|desc)$"),
    search: Optional[str] = Query(None),
    folder_id: Optional[str] = Query(None),
    template_type: Optional[str] = Query(None),
    db: DatabaseService = Depends(get_database_service)
):
    """Get all templates with details for a user"""
    try:
        # Build the query
        query = db.client.from_("templates").select(
            "*, folders(name, color), template_usage_stats(action_type, created_at)"
        ).eq("folders.user_id", user_id)

        # Apply filters
        if search:
            query = query.ilike("name", f"%{search}%")
        if folder_id:
            query = query.eq("folder_id", folder_id)
        if template_type:
            query = query.eq("template_type", template_type)

        # Apply sorting for database fields
        if sort_field in ["name", "folder_name"]:
            field_to_sort = sort_field if sort_field == "name" else "folders.name"
            query = query.order(field_to_sort, desc=(sort_direction == "desc"))
        
        result = await query.execute()

        if not result.data:
            return TemplatesResponse(templates=[], total=0)

        # Process results
        templates = []
        for item in result.data:
            folder = item.get("folders", {})
            usage_stats = item.get("template_usage_stats", [])

            # Get most recent action
            last_action = None
            last_action_date = None
            if usage_stats:
                sorted_stats = sorted(usage_stats, key=lambda x: x.get("created_at", ""), reverse=True)
                if sorted_stats:
                    last_action = sorted_stats[0].get("action_type")
                    last_action_date = sorted_stats[0].get("created_at")

            # Count files for this folder
            files_response = await db.client.from_("files").select("*", count="exact").eq("folder_id", item.get("folder_id")).execute()
            files_count = files_response.count or 0

            template = TemplateWithDetails(
                id=item.get("id"),
                folder_id=item.get("folder_id"),
                name=item.get("name"),
                content=item.get("content"),
                template_type=item.get("template_type"),
                file_extension=item.get("file_extension"),
                formatting_data=item.get("formatting_data"),
                word_compatible=item.get("word_compatible"),
                is_active=item.get("is_active"),
                created_at=item.get("created_at"),
                updated_at=item.get("updated_at"),
                folder_name=folder.get("name"),
                folder_color=folder.get("color"),
                files_count=files_count,
                last_action_type=last_action,
                last_action_date=last_action_date
            )
            templates.append(template)
        
        # Apply post-processing sorting
        if sort_field in ["last_action_type", "last_action_date", "files_count"]:
            if sort_field == "last_action_type":
                templates.sort(
                    key=lambda t: (t.last_action_type is None, t.last_action_type or ""),
                    reverse=(sort_direction == "desc")
                )
            elif sort_field == "last_action_date":
                templates.sort(
                    key=lambda t: (t.last_action_date is None, t.last_action_date or ""),
                    reverse=(sort_direction == "desc")
                )
            elif sort_field == "files_count":
                templates.sort(
                    key=lambda t: t.files_count,
                    reverse=(sort_direction == "desc")
                )

        return TemplatesResponse(templates=templates, total=len(templates))

    except Exception as e:
        logger.error(f"Error fetching templates: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching templates: {str(e)}")

@router.get("/{template_id}", response_model=TemplatePreviewResponse)
async def get_template_preview(
    template_id: str,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service)
):
    """Get template preview by ID"""
    try:
        result = await db.client.from_("templates").select(
            "*, folders(name)"
        ).eq("id", template_id).single().execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Template not found")

        # Track view action
        await track_template_usage(template_id, user_id, "viewed")

        item = result.data
        folder = item.get("folders", {})

        # Handle content_json vs content properly
        content_json = item.get("content_json")
        content = item.get("content")
        
        if content_json and isinstance(content_json, dict):
            final_content = json.dumps(content_json)
        elif content_json and isinstance(content_json, str):
            final_content = content_json
        else:
            final_content = content or ""
            
        template = TemplatePreview(
            id=item.get("id"),
            folder_id=item.get("folder_id"),
            name=item.get("name"),
            content=final_content,
            template_type=item.get("template_type"),
            file_extension=item.get("file_extension"),
            formatting_data=item.get("formatting_data"),
            folder_name=folder.get("name"),
            created_at=item.get("created_at"),
            updated_at=item.get("updated_at")
        )

        return TemplatePreviewResponse(template=template)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching template: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching template: {str(e)}")

# ============================================================================
# TEMPLATE GENERATION
# ============================================================================

@router.post("/generate", response_model=ApiResponse)
async def start_template_generation(
    user_id: str = Query(...),
    folder_id: str = Query(...),
    priority_template_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service)
):
    """Start template generation from files in a folder"""
    try:
        # Verify folder exists and belongs to user
        folder_response = await db.client.from_("folders").select(
            "*").eq("id", folder_id).eq("user_id", user_id).execute()

        if not folder_response.data:
            raise HTTPException(status_code=404, detail="Folder not found")

        # Check for existing active jobs for this folder
        existing_jobs = await db.client.from_("jobs").select(
            "id, status, metadata, celery_task_id"
        ).eq("user_id", user_id).eq("job_type", "template_generation").in_(
            "status", ["pending", "processing"]
        ).execute()
        
        if existing_jobs.data:
            for job in existing_jobs.data:
                job_metadata = job.get("metadata", {})
                if job_metadata.get("folder_id") == folder_id:
                    logger.info(f"Found existing active job {job['id']} for folder {folder_id}")
                    return ApiResponse(
                        success=True,
                        message="Template generation already in progress for this folder",
                        data={
                            "generation_job_id": job["id"],
                            "status": job["status"],
                            "files_ready": 0,
                            "files_need_processing": 0,
                            "estimated_duration": "2-5 minutes",
                            "existing_job": True
                        }
                    )

        # Get files from the folder
        files_response = await db.client.from_("files").select(
            "*").eq("folder_id", folder_id).eq("user_id", user_id).execute()

        if not files_response.data:
            raise HTTPException(
                status_code=400, detail="No files found in the folder"
            )
        
        # Check for recent template generation
        existing_templates = await db.client.from_("templates").select(
            "id, created_at"
        ).eq("folder_id", folder_id).order("created_at", desc=True).limit(1).execute()
        
        if existing_templates.data:
            last_template = existing_templates.data[0]
            created_at = datetime.fromisoformat(last_template["created_at"].replace('Z', '+00:00'))
            time_diff = (datetime.utcnow() - created_at.replace(tzinfo=None)).total_seconds()
            
            if time_diff < 300:  # 5 minutes
                logger.warning(f"Template was created {time_diff} seconds ago for folder {folder_id}")
                return ApiResponse(
                    success=False,
                    message=f"A template was generated for this folder {int(time_diff)} seconds ago. Please wait before generating another.",
                    data={
                        "existing_template_id": last_template["id"],
                        "seconds_since_last_generation": int(time_diff)
                    }
                )

        # Generate template name
        folder_name = folder_response.data[0].get('name', 'Template')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        template_name = f"{folder_name}_Template_{timestamp}"
        
        # Create job record
        generation_job_id = await create_job(
            user_id=user_id,
            job_type="template_generation",
            metadata={
                "folder_id": folder_id,
                "priority_template_id": priority_template_id,
                "template_name": template_name,
                "file_ids": [f["id"] for f in files_response.data],
                "total_files": len(files_response.data)
            },
            total_steps=3
        )
        
        # Start Celery task
        task_result = template_generation_task.delay(generation_job_id)
        
        # Update job with Celery task ID
        await db.client.from_("jobs").update({
            "celery_task_id": task_result.id,
            "status": "processing",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", generation_job_id).execute()

        return ApiResponse(
            success=True,
            message="Template generation job created successfully",
            data={
                "generation_job_id": generation_job_id,
                "status": "pending",
                "files_ready": sum(1 for f in files_response.data if f["status"] == "processed"),
                "files_need_processing": sum(1 for f in files_response.data if f["status"] != "processed"),
                "total_files": len(files_response.data),
                "estimated_duration": "2-5 minutes"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting template generation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error starting template generation: {str(e)}")

@router.get("/generation/{generation_job_id}/status")
async def get_generation_status(
    generation_job_id: str,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service)
):
    """Get the status of a template generation job"""
    try:
        job_status = await get_job_status_util(generation_job_id)
        
        if "error" in job_status:
            return {
                "generation_job_id": generation_job_id,
                "status": "error",
                "progress": 0,
                "message": job_status.get("error", "Unknown error"),
                "files_processed": 0,
                "total_files": 0,
                "estimated_completion": ""
            }
        
        # Extract metadata
        metadata = job_status.get("metadata", {})
        total_files = metadata.get("total_files", 0)
        
        # Get files processed count
        files_processed = 0
        if job_status.get("steps"):
            for step in job_status["steps"]:
                if step.get("step_name") == "process_files" and step.get("metadata"):
                    files_processed = step["metadata"].get("processed_files", 0)
                    break
        
        # Generate status message
        status = job_status.get("status", "pending")
        current_step_name = job_status.get("current_step_name", "")
        
        if status == "completed":
            message = "Template generation completed successfully"
        elif status == "failed":
            message = job_status.get("error_message", "Template generation failed")
        elif current_step_name == "process_files":
            message = f"Processing files ({files_processed}/{total_files})"
        elif current_step_name == "generate_template":
            message = "Generating template with AI..."
        elif current_step_name == "save_template":
            message = "Saving template..."
        else:
            message = "Preparing template generation..."
        
        # Estimate completion
        estimated_completion = ""
        if status == "processing":
            progress = job_status.get("progress", 0)
            if progress > 0:
                if progress < 50:
                    estimated_completion = "3-5 minutes"
                elif progress < 80:
                    estimated_completion = "1-2 minutes"
                else:
                    estimated_completion = "Less than 1 minute"
        
        return {
            "generation_job_id": generation_job_id,
            "status": status,
            "progress": job_status.get("progress", 0),
            "message": message,
            "files_processed": files_processed,
            "total_files": total_files,
            "estimated_completion": estimated_completion
        }
        
    except Exception as e:
        logger.error(f"Error getting generation status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error getting generation status: {str(e)}")

# ============================================================================
# TEMPLATE OPERATIONS
# ============================================================================

@router.put("/{template_id}")
async def update_template_content(
    template_id: str,
    request: dict,
    user_id: str = Query(...),
    auto_extract_clauses: bool = Query(False, description="Automatically extract and update clauses in content_json"),
    db: DatabaseService = Depends(get_database_service)
):
    """Update template content"""
    try:
        # Verify template exists and user has access
        template_response = await db.client.from_("templates").select(
            "*, folders(user_id, name)"
        ).eq("id", template_id).single().execute()
        
        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")
        
        folder = template_response.data.get("folders", {})
        if folder.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Extract content from request
        content = request.get("content", "")
        name = request.get("name")
        
        # Determine content type
        content_json = None
        content_text = None
        
        try:
            parsed_content = json.loads(content)
            if isinstance(parsed_content, dict) and 'clauses' in parsed_content:
                content_json = parsed_content
            else:
                content_text = content
        except (json.JSONDecodeError, TypeError):
            content_text = content
        
        # Auto-extract clauses if requested
        if auto_extract_clauses and content_text:
            template_name = name or template_response.data.get("name", "Template")
            extracted_content_json = await _update_content_json_from_template(content_text, template_name)
            if extracted_content_json:
                content_json = extracted_content_json
        
        # Build update data
        update_data = {"updated_at": datetime.utcnow().isoformat()}
        
        if content_json is not None:
            update_data["content_json"] = content_json
            if content_text:
                update_data["content"] = content_text
        else:
            update_data["content"] = content_text
        
        if name:
            update_data["name"] = name
        
        # Update template
        update_result = await db.client.from_("templates").update(update_data).eq(
            "id", template_id
        ).execute()
        
        if not update_result.data:
            raise HTTPException(status_code=500, detail="Failed to update template")
        
        # Track update
        await track_template_usage(template_id, user_id, "edited", {
            "content_length": len(content),
            "updated_fields": list(update_data.keys()),
            "clauses_count": len(content_json.get("clauses", [])) if content_json else 0
        })
        
        # Return updated template
        updated_template = update_result.data[0]
        response_content = updated_template.get("content", "")
        if updated_template.get("content_json") and isinstance(updated_template.get("content_json"), dict):
            response_content = json.dumps(updated_template.get("content_json"))
        
        template = TemplatePreview(
            id=updated_template.get("id"),
            folder_id=updated_template.get("folder_id"),
            name=updated_template.get("name"),
            content=response_content,
            template_type=updated_template.get("template_type"),
            file_extension=updated_template.get("file_extension"),
            formatting_data=updated_template.get("formatting_data"),
            folder_name=folder.get("name"),
            created_at=updated_template.get("created_at"),
            updated_at=updated_template.get("updated_at")
        )
        
        return {
            "success": True,
            "template": template,
            "clauses_extracted": len(content_json.get("clauses", [])) if content_json else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating template: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error updating template: {str(e)}")

@router.get("/{template_id}/export/{format}")
async def export_template(
    template_id: str,
    format: str,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service),
    exporter: DocumentExporter = Depends(get_document_exporter)
):
    """Export template in specified format (html, docx, pdf)"""
    try:
        # Validate format
        allowed_formats = ['html', 'docx', 'pdf']
        if format not in allowed_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid format. Allowed formats: {', '.join(allowed_formats)}"
            )
        
        # Get template data
        result = await db.client.from_("templates").select(
            "*, folders!inner(user_id)"
        ).eq("id", template_id).eq("folders.user_id", user_id).single().execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Template not found")

        template_data = result.data
        
        # Track download
        await track_template_usage(template_id, user_id, "downloaded", {
            "export_format": format,
            "export_timestamp": datetime.utcnow().isoformat()
        })
        
        # Export based on format
        if format == 'html':
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{template_data['name']}</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
                    .metadata {{ background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                    .content {{ white-space: pre-wrap; }}
                </style>
            </head>
            <body>
                <h1>{template_data['name']}</h1>
                <div class="metadata">
                    <p><strong>Type:</strong> {template_data['template_type']}</p>
                    <p><strong>Created:</strong> {template_data['created_at']}</p>
                    <p><strong>Updated:</strong> {template_data['updated_at']}</p>
                </div>
                <div class="content">{template_data['content']}</div>
            </body>
            </html>
            """
            
            return Response(
                content=html_content,
                media_type="text/html",
                headers={
                    "Content-Disposition": f"attachment; filename={template_data['name']}.html"
                }
            )
            
        elif format == 'docx':
            try:
                docx_content = exporter.generate_docx(template_data['content'])
                return Response(
                    content=docx_content,
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    headers={
                        "Content-Disposition": f"attachment; filename={template_data['name']}.docx"
                    }
                )
            except Exception as docx_error:
                logger.error(f"DOCX export failed: {docx_error}")
                # Fallback to text
                return Response(
                    content=template_data['content'],
                    media_type="text/plain",
                    headers={
                        "Content-Disposition": f"attachment; filename={template_data['name']}.txt"
                    }
                )
                
        elif format == 'pdf':
            try:
                pdf_content = exporter.export_to_pdf(
                    template_data['content'],
                    template_data['name'],
                    template_data.get('formatting_data', {})
                )
                return Response(
                    content=pdf_content,
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f"attachment; filename={template_data['name']}.pdf"
                    }
                )
            except Exception as pdf_error:
                logger.error(f"PDF export failed: {pdf_error}")
                # Fallback to HTML
                return export_template(template_id, 'html', user_id, db, exporter)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting template: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error exporting template: {str(e)}")

# ============================================================================
# CLAUSE OPERATIONS
# ============================================================================

@router.get("/{template_id}/clauses")
async def get_template_clauses(
    template_id: str,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service)
):
    """Get all clauses from clause_library for a template's folder"""
    try:
        # Get template and verify access
        template_result = await db.client.from_("templates").select(
            "folder_id, folders(user_id)"
        ).eq("id", template_id).single().execute()
        
        if not template_result.data:
            raise HTTPException(status_code=404, detail="Template not found")
        
        folder_data = template_result.data.get("folders", {})
        if folder_data.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        folder_id = template_result.data["folder_id"]
        
        # Get clauses from clause_library
        clauses_result = await db.client.from_("clause_library").select(
            "clause_type, clause_text, clause_metadata"
        ).eq("user_id", user_id).eq("folder_id", folder_id).order("clause_type").execute()
        
        if not clauses_result.data:
            return {"clauses": []}
        
        # Format clauses
        formatted_clauses = []
        for clause in clauses_result.data:
            metadata = clause.get("clause_metadata", {}) or {}
            
            formatted_clauses.append({
                "clause_type": clause["clause_type"],
                "clause_text": clause["clause_text"],
                "clause_purpose": metadata.get("purpose", f"Standard {clause['clause_type']} clause"),
                "position_context": metadata.get("position_context", "General use"),
                "relevance_assessment": {
                    "when_to_include": metadata.get("when_to_include", ["Standard contracts"]),
                    "when_to_exclude": metadata.get("when_to_exclude", []),
                    "industry_considerations": metadata.get("industry_considerations", []),
                    "risk_implications": metadata.get("risk_implications", ["Standard risk"]),
                    "compliance_requirements": metadata.get("compliance_requirements", []),
                    "best_practices": metadata.get("best_practices", ["Review with legal counsel"])
                }
            })
        
        return {"clauses": formatted_clauses}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching template clauses: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching template clauses: {str(e)}")

@router.get("/{template_id}/clauses-from-content")
async def get_template_clauses_from_content(
    template_id: str,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service)
):
    """Get clauses from template's content_json field"""
    try:
        # Verify access
        template_response = await db.client.from_("templates").select(
            "content_json, folders(user_id, name)"
        ).eq("id", template_id).single().execute()
        
        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")
        
        folder = template_response.data.get("folders", {})
        if folder.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        content_json = template_response.data.get("content_json")
        
        if not content_json:
            return {
                "success": True,
                "clauses": [],
                "message": "No clauses data found in template"
            }
        
        clauses = content_json.get("clauses", [])
        extraction_metadata = content_json.get("extraction_metadata", {})
        
        return {
            "success": True,
            "clauses": clauses,
            "extraction_metadata": extraction_metadata,
            "total_clauses": len(clauses)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching template clauses from content: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching template clauses: {str(e)}")

# ============================================================================
# FILE & FOLDER OPERATIONS
# ============================================================================

@router.get("/folder/{folder_id}/files")
async def get_folder_files(
    folder_id: str,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service)
):
    """Get files for a specific folder"""
    try:
        # Verify folder access
        folder_response = await db.client.from_("folders").select(
            "name, color"
        ).eq("id", folder_id).eq("user_id", user_id).single().execute()
        
        if not folder_response.data:
            raise HTTPException(status_code=404, detail="Folder not found")
        
        # Get files
        files_response = await db.client.from_("files").select(
            "*, file_info(*)"
        ).eq("folder_id", folder_id).eq("user_id", user_id).order("created_at", desc=True).execute()
        
        files = files_response.data or []
        folder_data = folder_response.data
        
        return {
            "files": files,
            "folder_name": folder_data.get("name"),
            "folder_color": folder_data.get("color"),
            "total_files": len(files)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching folder files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching folder files: {str(e)}")

@router.get("/file/{file_id}/info")
async def get_file_info(
    file_id: str,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service)
):
    """Get file_info data for a specific file"""
    try:
        # Verify file access
        file_response = await db.client.from_("files").select(
            "id, original_filename, user_id"
        ).eq("id", file_id).eq("user_id", user_id).single().execute()
        
        if not file_response.data:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get file_info
        file_info_response = await db.client.from_("file_info").select(
            "*"
        ).eq("file_id", file_id).eq("user_id", user_id).execute()
        
        file_info = file_info_response.data or []
        
        return {
            "file_id": file_id,
            "file_name": file_response.data.get("original_filename"),
            "file_info": file_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching file info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching file info: {str(e)}")

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.get("/{template_id}/content-info")
async def get_template_content_info(
    template_id: str,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service)
):
    """Get information about template content structure"""
    try:
        # Verify access
        template_response = await db.client.from_("templates").select(
            "name, content, content_json, created_at, updated_at, folders(user_id, name)"
        ).eq("id", template_id).single().execute()
        
        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")
        
        folder = template_response.data.get("folders", {})
        if folder.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        template_data = template_response.data
        content_json = template_data.get("content_json")
        content = template_data.get("content", "")
        
        # Analyze content
        info = {
            "template_id": template_id,
            "template_name": template_data.get("name"),
            "has_text_content": bool(content),
            "text_content_length": len(content) if content else 0,
            "has_json_content": bool(content_json),
            "created_at": template_data.get("created_at"),
            "updated_at": template_data.get("updated_at")
        }
        
        if content_json:
            clauses = content_json.get("clauses", [])
            extraction_metadata = content_json.get("extraction_metadata", {})
            
            # Analyze clause types
            clause_types = {}
            for clause in clauses:
                clause_type = clause.get("clause_type", "unknown")
                clause_types[clause_type] = clause_types.get(clause_type, 0) + 1
            
            info.update({
                "json_content_info": {
                    "total_clauses": len(clauses),
                    "clause_types": clause_types,
                    "extraction_metadata": extraction_metadata
                }
            })
        else:
            info["json_content_info"] = None
        
        return {
            "success": True,
            "info": info,
            "capabilities": {
                "can_extract_clauses": True,
                "supports_auto_extraction": True,
                "supported_formats": ["text", "json"],
                "extraction_endpoint": f"/api/templates/{template_id}/clauses-from-content",
                "update_with_extraction": f"/api/templates/{template_id}?auto_extract_clauses=true"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching template content info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching template content info: {str(e)}")

@router.post("/admin/cleanup-jobs")
async def cleanup_stuck_jobs(
    user_id: str = Query(...),
    max_age_hours: int = Query(24, description="Maximum age in hours for jobs to be considered stuck"),
    db: DatabaseService = Depends(get_database_service)
):
    """Admin endpoint to clean up stuck template generation jobs"""
    try:
        from core.db_utilities import cleanup_orphaned_records
        cleanup_result = await cleanup_orphaned_records(user_id)
        total_cleaned = sum(cleanup_result.values())
        
        return ApiResponse(
            success=True,
            message=f"Successfully cleaned up {total_cleaned} orphaned records",
            data={
                "cleanup_result": cleanup_result,
                "total_cleaned": total_cleaned,
                "max_age_hours": max_age_hours
            }
        )
        
    except Exception as e:
        logger.error(f"Error during manual job cleanup: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during job cleanup: {str(e)}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def track_template_usage(
    template_id: str,
    user_id: Optional[str],
    action_type: str,
    metadata: Optional[dict] = None
):
    """Helper function to track template usage"""
    try:
        db = get_database_service()

        usage_data = {
            "template_id": template_id,
            "action_type": action_type,
            "metadata": metadata or {}
        }
        
        if user_id:
            usage_data["user_id"] = user_id

        result = await db.client.from_("template_usage_stats").insert(usage_data).execute()
        
        if result.data:
            logger.info(f"Tracked usage: {action_type} for template {template_id}")
        else:
            logger.warning(f"Failed to track usage: {action_type} for template {template_id}")

    except Exception as e:
        logger.error(f"Error tracking template usage: {str(e)}", exc_info=True)

async def _update_content_json_from_template(template_content: str, template_name: str) -> Optional[Dict[str, Any]]:
    """Extract clauses from template content and format for content_json"""
    try:
        from core.template_generator import TemplateGenerator
        from core.api_config import APIConfiguration
        
        api_config = APIConfiguration()
        template_generator = TemplateGenerator(api_config)
        
        # Extract clauses used in the template
        extracted_clauses = template_generator.extract_used_clauses(template_content)
        
        if extracted_clauses:
            return {
                "clauses": extracted_clauses,
                "extraction_metadata": {
                    "extracted_at": datetime.utcnow().isoformat(),
                    "total_clauses": len(extracted_clauses),
                    "template_name": template_name,
                    "last_updated": datetime.utcnow().isoformat()
                }
            }
        return None
        
    except Exception as e:
        logger.warning(f"Failed to extract clauses for content_json update: {e}")
        return None