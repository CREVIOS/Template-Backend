from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from loguru import logger
import json
import asyncio

from core.database import get_database_service, DatabaseService
from core.redis_cache import get_cache_service, RedisCacheService
from core.api_config import APIConfiguration
# from core.template_generator import TemplateGenerator
from core.document_exporter import DocumentExporter, ExportStyleConfig
from core.models import (
    TemplatesResponse,
    TemplatePreviewResponse,
    TemplateWithDetails,
    TemplatePreview,
    ApiResponse,
    RenameTemplateRequest
)
from celery_tasks import template_generation_task, template_update_task
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

def get_cache_service_dep() -> RedisCacheService:
    """FastAPI dependency to get cache service"""
    return get_cache_service()

async def fetch_templates_from_db(
    user_id: str,
    sort_field: str = "created_at",
    sort_direction: str = "desc",
    search: Optional[str] = None,
    folder_id: Optional[str] = None,
    template_type: Optional[str] = None,
    db: DatabaseService = None
) -> List[Dict[str, Any]]:
    """Fetch templates from database using RPC and return processed data sorted by latest created_at"""
    if db is None:
        db = get_database_service()
    
    try:
        logger.debug(f"Making RPC call to get_user_templates for user: {user_id}")
        
        # Use RPC to get user templates (properly filtered by user_id)
        result = await db.client.rpc("get_user_templates",
                             {"p_user_id": user_id}).execute()
        
        logger.debug(f"RPC call completed. Result: {result}")
        logger.debug(f"Result data: {result.data}")
        logger.debug(f"Result count: {result.count}")
        
        # Check for RPC errors
        if hasattr(result, 'error') and result.error:
            logger.error(f"RPC call returned error: {result.error}")
            return []
        
        logger.debug(f"Result data type: {type(result.data)}, Length: {len(result.data) if result.data else 0}")

        if not result.data:
            logger.debug("No templates returned from RPC call")
            # Try a direct query as a fallback test
            logger.debug("Attempting direct query as fallback...")
            try:
                direct_result = await db.client.from_("templates").select(
                    "*, folders!inner(user_id, name as folder_name)"
                ).eq("folders.user_id", user_id).execute()
                logger.debug(f"Direct query result: {len(direct_result.data) if direct_result.data else 0} templates")
                if direct_result.data:
                    logger.debug(f"Sample direct result: {direct_result.data[0]}")
            except Exception as direct_error:
                logger.debug(f"Direct query also failed: {direct_error}")
            return []

        # Process and sort results by created_at (latest first)
        templates = []
        for i, item in enumerate(result.data):
            try:
                logger.debug(f"Processing template {i+1}: ID={item.get('id')}, Name={item.get('name')}")
                
                template_data = {
                    "id": item.get("id"),
                    "folder_id": item.get("folder_id"),
                    "name": item.get("name"),
                    "content": item.get("content"),
                    "template_type": item.get("template_type"),
                    "file_extension": item.get("file_extension"),
                    "formatting_data": item.get("formatting_data"),
                    "word_compatible": item.get("word_compatible"),
                    "is_active": item.get("is_active"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "folder_name": item.get("folder_name"),
                    "files_count": item.get("files_count", 0),
                    "status": item.get("status")
                }
                templates.append(template_data)
                
            except Exception as e:
                safe_log_error(f"Skipping template {item.get('id', 'unknown')} due to data error", e)
                continue

        if templates:
            template_ids = [t.get("id") for t in templates if t.get("id")]
            if template_ids:
                # Get latest usage stats for each template using a more efficient query
                usage_stats_result = await db.client.from_("template_usage_stats").select(
                    "template_id, action_type, created_at"
                ).in_("template_id", template_ids).order(
                    "created_at", desc=True
                ).execute()
                
                # Create a map of template_id to latest usage stats
                latest_usage_stats = {}
                if usage_stats_result.data:
                    for stat in usage_stats_result.data:
                        template_id = stat.get("template_id")
                        if template_id not in latest_usage_stats:
                            # First occurrence is the latest due to desc order
                            latest_usage_stats[template_id] = {
                                "last_action_type": stat.get("action_type"),
                                "last_action_date": stat.get("created_at")
                            }
                
                # Enrich template data with usage stats
                for template_data in templates:
                    template_id = template_data.get("id")
                    if template_id in latest_usage_stats:
                        stats = latest_usage_stats[template_id]
                        template_data["last_action_type"] = stats["last_action_type"]
                        template_data["last_action_date"] = stats["last_action_date"]
                    else:
                        template_data["last_action_type"] = None
                        template_data["last_action_date"] = None

        def parse_timestamp(timestamp_str):
            if not timestamp_str:
                return datetime.min
            try:
                # Handle Supabase timestamp format: "2025-07-20 17:57:46.680676+00"
                if timestamp_str.endswith('+00'):
                    timestamp_str = timestamp_str[:-3] + '+00:00'
                return datetime.fromisoformat(timestamp_str)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not parse timestamp: {timestamp_str} - {e}")
                return datetime.min
        
        templates.sort(
            key=lambda t: parse_timestamp(t.get("created_at")), 
            reverse=True
        )
        
        logger.debug(f"Successfully processed {len(templates)} templates")
        return templates

    except Exception as e:
        # FIXED: Use safe logging instead of f-string
        safe_log_error("Error fetching templates from DB", e, exc_info=True)
        return []

# Fix for get_folder_files function (around line 999)
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
        # FIXED: Use safe logging instead of f-string
        safe_log_error("Error fetching folder files", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching folder files: {str(e)}")


# ============================================================================
# TEMPLATE LISTING & RETRIEVAL
# ============================================================================

@router.get("/active-job-step/{user_id}")
async def get_active_job_step(
    user_id: str,
    db: DatabaseService = Depends(get_database_service)
):
    """Get active job step for a user"""
    try:
        # For now, return a simple response to test the endpoint
        return {
            "active_job": None,
            "active_step": None,
            "user_id": user_id,
            "message": "Endpoint working - no active jobs found"
        }
        
    except Exception as e:
        logger.error(f"Error getting active job step for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error getting active job step: {str(e)}")

def safe_log_error(message: str, exception: Exception, **kwargs):
    logger.opt(exception=exception).error("{}: {}", message, exception, **kwargs)


@router.get("/", response_model=TemplatesResponse)
async def get_templates(
    user_id: str = Query(...),
    sort_field: str = Query("created_at", pattern="^(name|folder_name|created_at)$"),
    sort_direction: str = Query("desc", pattern="^(asc|desc)$"),
    search: Optional[str] = Query(None),
    folder_id: Optional[str] = Query(None),
    template_type: Optional[str] = Query(None),
    force_refresh: bool = Query(False, description="Force refresh from database bypassing cache"),
    db: DatabaseService = Depends(get_database_service),
    cache: RedisCacheService = Depends(get_cache_service_dep)
):
    """Get all templates with details for a user - cache-first approach"""
    try:
        templates_data = []
        cache_used = False
        
        # Always try cache first unless force_refresh is True
        if not force_refresh:
            cached_templates = await cache.get_user_templates(user_id)
            if cached_templates:
                templates_data = cached_templates
                cache_used = True
                logger.debug(f"Cache HIT for user {user_id}: {len(templates_data)} templates")
            else:
                logger.debug(f"Cache MISS for user {user_id}, fetching from DB")
        
        # If cache miss or force refresh, fetch from database
        if not templates_data:
            templates_data = await fetch_templates_from_db(
                user_id=user_id,
                db=db
            )
            
            # Cache the unfiltered results for future use
            if templates_data:
                await cache.set_user_templates(user_id, templates_data)
                logger.debug(f"Cached {len(templates_data)} templates for user {user_id}")
        
        # Apply filtering and sorting to data (from cache or DB)
        if templates_data:
            # Apply filters
            if search:
                templates_data = [t for t in templates_data if search.lower() in t.get("name", "").lower()]
                logger.debug(f"Applied search filter '{search}': {len(templates_data)} templates remain")
            if folder_id:
                templates_data = [t for t in templates_data if t.get("folder_id") == folder_id]
                logger.debug(f"Applied folder filter '{folder_id}': {len(templates_data)} templates remain")
            if template_type:
                templates_data = [t for t in templates_data if t.get("template_type") == template_type]
                logger.debug(f"Applied type filter '{template_type}': {len(templates_data)} templates remain")
            
            # Data is already sorted by created_at (latest first) from RPC
            # Apply additional sorting if requested  
            if sort_field != "created_at":
                if sort_field == "name":
                    templates_data.sort(
                        key=lambda t: t.get("name", "").lower(),
                        reverse=(sort_direction == "desc")
                    )
                elif sort_field == "folder_name":
                    templates_data.sort(
                        key=lambda t: t.get("folder_name", "").lower(),
                        reverse=(sort_direction == "desc")
                    )
                logger.debug(f"Applied custom sorting: {sort_field} {sort_direction}")
            else:
                # If sorting by created_at and direction is asc, reverse the default desc order
                if sort_direction == "asc":
                    templates_data.reverse()
                    logger.debug("Applied created_at ascending sort")
        
        # Convert to TemplateWithDetails objects
        templates = []
        for template_data in templates_data:
            template = TemplateWithDetails(
                id=template_data.get("id"),
                folder_id=template_data.get("folder_id"),
                name=template_data.get("name"),
                content=template_data.get("content"),
                template_type=template_data.get("template_type"),
                file_extension=template_data.get("file_extension"),
                formatting_data=template_data.get("formatting_data"),
                word_compatible=template_data.get("word_compatible"),
                is_active=template_data.get("is_active"),
                created_at=template_data.get("created_at"),
                updated_at=template_data.get("updated_at"),
                folder_name=template_data.get("folder_name"),
                folder_color=None,
                files_count=template_data.get("files_count", 0),  
                last_action_type=template_data.get("last_action_type"),
                last_action_date=template_data.get("last_action_date"),
            )
            templates.append(template)

        filter_info = []
        if search:
            filter_info.append(f"search='{search}'")
        if folder_id:
            filter_info.append(f"folder_id='{folder_id}'")
        if template_type:
            filter_info.append(f"type='{template_type}'")
        if sort_field != "created_at" or sort_direction != "desc":
            filter_info.append(f"sort={sort_field}_{sort_direction}")
        
        filter_str = f" with filters: {', '.join(filter_info)}" if filter_info else ""
        logger.info(f"Returned {len(templates)} templates for user {user_id} (cache_used: {cache_used}){filter_str}")
        return TemplatesResponse(templates=templates, total=len(templates))

    except Exception as e:
        # FIXED: Use safe logging instead of f-string
        safe_log_error("Error fetching templates", e, exc_info=True)
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
        folder = item.get("folders") or {}  # Handle None case

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
# CACHE MANAGEMENT
# ============================================================================

@router.post("/cache/refresh")
async def refresh_template_cache(
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service),
    cache: RedisCacheService = Depends(get_cache_service_dep)
):
    """Refresh template cache for a user"""
    try:
        # Invalidate existing cache
        await cache.invalidate_user_cache(user_id)
        
        # Fetch fresh data from database
        fresh_templates = await fetch_templates_from_db(user_id=user_id, db=db)
        
        # Cache the fresh data
        success = await cache.set_user_templates(user_id, fresh_templates)
        
        if success:
            logger.info(f"Cache refreshed for user {user_id}: {len(fresh_templates)} templates")
            return ApiResponse(
                success=True,
                message=f"Cache refreshed successfully with {len(fresh_templates)} templates",
                data={"template_count": len(fresh_templates)}
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to refresh cache")
            
    except Exception as e:
        logger.error(f"Error refreshing cache for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error refreshing cache: {str(e)}")

@router.get("/cache/info")
async def get_cache_info(
    user_id: str = Query(...),
    cache: RedisCacheService = Depends(get_cache_service_dep)
):
    """Get cache information for a user"""
    try:
        cache_info = await cache.get_cache_info(user_id)
        cache_valid = await cache.is_cache_valid(user_id)
        cache_stats = await cache.get_cache_stats()
        
        return {
            "user_cache_info": cache_info,
            "cache_valid": cache_valid,
            "global_stats": {
                "hit_count": cache_stats.hit_count,
                "miss_count": cache_stats.miss_count,
                "hit_rate": cache_stats.hit_count / (cache_stats.hit_count + cache_stats.miss_count) * 100 if (cache_stats.hit_count + cache_stats.miss_count) > 0 else 0,
                "last_refresh": cache_stats.last_refresh.isoformat() if cache_stats.last_refresh else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting cache info for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting cache info: {str(e)}")

@router.get("/cache/health")
async def cache_health_check(
    cache: RedisCacheService = Depends(get_cache_service_dep)
):
    """Health check for cache service"""
    try:
        health_info = await cache.health_check()
        return health_info
    except Exception as e:
        logger.error(f"Cache health check failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }

# ============================================================================
# TEMPLATE GENERATION
# ============================================================================
@router.post("/generate", response_model=ApiResponse)
async def start_template_generation(
    user_id: str = Query(...),
    folder_id: str = Query(...),
    priority_template_id: str = Query(...),
    template_name: Optional[str] = Query(None),  # ← ADD THIS PARAMETER
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

        # Log the received template name for debugging
        logger.info(f"   Template generation request:")
        logger.info(f"   User ID: {user_id}")
        logger.info(f"   Folder ID: {folder_id}")
        logger.info(f"   Priority Template ID: {priority_template_id}")
        logger.info(f"   Custom Template Name: {template_name}")  
        
        # Determine final template name
        if template_name and template_name.strip():
            # Use the custom name provided by the user
            final_template_name = template_name.strip()
            logger.info(f"✅ Using custom template name: '{final_template_name}'")
        else:
            # Generate automatic name if no custom name provided
            folder_name = folder_response.data[0].get('name', 'Template')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            final_template_name = f"{folder_name}_Template_{timestamp}"
            logger.info(f"✅ Using auto-generated name: '{final_template_name}'")
        
        # Create job record with the final template name
        generation_job_id = await create_job(
            user_id=user_id,
            job_type="template_generation",
            metadata={
                "folder_id": folder_id,
                "priority_template_id": priority_template_id,
                "template_name": final_template_name,  # ← USE FINAL NAME HERE
                "custom_name_provided": bool(template_name and template_name.strip()),  # ← TRACK IF CUSTOM NAME WAS PROVIDED
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
            message=f"Template generation job created successfully with name: '{final_template_name}'",  # ← INCLUDE NAME IN MESSAGE
            data={
                "generation_job_id": generation_job_id,
                "template_name": final_template_name,  # ← RETURN THE USED NAME
                "custom_name_provided": bool(template_name and template_name.strip()),  # ← INDICATE IF CUSTOM NAME WAS USED
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


# @router.get("/folder/{folder_id}/clauses")
# async def get_folder_clauses(
#     folder_id: str,
#     user_id: str = Query(...),
#     db: DatabaseService = Depends(get_database_service),
#     cache: RedisCacheService = Depends(get_cache_service)
# ):
#     """
#     Get all clauses from clause_library for a folder, with Redis caching.
#     """
#     cache_key = f"clause_library:folder:{folder_id}:user:{user_id}"
#     # Try cache first
#     cached = await asyncio.to_thread(cache.client.get, cache_key)
#     if cached:
#         return {"clauses": json.loads(cached)}
#     # Fetch from DB
#     clauses_result = await db.client.from_("clause_library").select("*").eq("user_id", user_id).eq("folder_id", folder_id).order("clause_type").execute()
#     clauses = clauses_result.data or []
#     # Cache result
#     await asyncio.to_thread(cache.client.setex, cache_key, 300, json.dumps(clauses))  # 5 min TTL
#     return {"clauses": clauses}


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


# ============================================================================
# TEMPLATE RENAMING AND DELETION
# ============================================================================

@router.delete("/{template_id}", response_model=ApiResponse)
async def delete_template(
    template_id: str,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service),
    cache: RedisCacheService = Depends(get_cache_service_dep)
):
    """Delete a template by ID"""
    try:
        # Verify template exists and user has access using an inner join
        select_query = await db.client.from_("templates").select(
            "id, folders!inner(user_id)"
        ).eq("id", template_id).eq("folders.user_id", user_id).single().execute()

        if not select_query.data:
            raise HTTPException(status_code=404, detail="Template not found or access denied")

        # Delete the template
        delete_result = await db.client.from_("templates").delete().eq("id", template_id).execute()

        if not delete_result.data:
            raise HTTPException(status_code=500, detail="Failed to delete template")

        # Invalidate cache for the user to reflect the change
        await cache.invalidate_user_cache(user_id)
        logger.info(f"Deleted template {template_id} for user {user_id} and invalidated cache.")
        
        # Track delete action
        await track_template_usage(template_id, user_id, "deleted")

        return ApiResponse(success=True, message="Template deleted successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting template {template_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting template: {str(e)}")

@router.patch("/{template_id}/rename", response_model=ApiResponse)
async def rename_template(
    template_id: str,
    request: RenameTemplateRequest,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service),
    cache: RedisCacheService = Depends(get_cache_service_dep)
):
    """Rename a template"""
    try:
        # Verify template exists and user has access
        select_query = await db.client.from_("templates").select(
            "id, name, folders!inner(user_id)"
        ).eq("id", template_id).eq("folders.user_id", user_id).single().execute()

        if not select_query.data:
            raise HTTPException(status_code=404, detail="Template not found or access denied")
            
        old_name = select_query.data.get("name")

        # Update the template name
        update_result = await db.client.from_("templates").update(
            {"name": request.name, "updated_at": datetime.utcnow().isoformat()}
        ).eq("id", template_id).execute()

        if not update_result.data:
            raise HTTPException(status_code=500, detail="Failed to rename template")
            
        # Invalidate cache
        await cache.invalidate_user_cache(user_id)
        logger.info(f"Renamed template {template_id} from '{old_name}' to '{request.name}' for user {user_id} and invalidated cache.")

        # Track rename action
        await track_template_usage(template_id, user_id, "renamed", {
            "old_name": old_name,
            "new_name": request.name
        })

        return ApiResponse(success=True, message="Template renamed successfully", data={"new_name": request.name})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming template {template_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error renaming template: {str(e)}")


# ============================================================================
# TEMPLATE CONTENT UPDATES
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
                
                
                exporter = DocumentExporter()
                cfg = ExportStyleConfig(
                    numbering_style='numeric',          # or 'roman', 'numeric'
                    drafting_note_as_comment=True,    # comments instead of inline notes
                )
                docx_content = exporter.export_json_to_docx(template_data['content'], cfg)
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
        # FIXED: Use safe logging instead of f-string
        safe_log_error("Error fetching folder files", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching folder files: {str(e)}")


# @router.get("/file/{file_id}/info")
# async def get_file_info(
#     file_id: str,
#     user_id: str = Query(...),
#     db: DatabaseService = Depends(get_database_service)
# ):
#     """Get file_info data for a specific file"""
#     try:
#         # Verify file access
#         file_response = await db.client.from_("files").select(
#             "id, original_filename, user_id"
#         ).eq("id", file_id).eq("user_id", user_id).single().execute()
        
#         if not file_response.data:
#             raise HTTPException(status_code=404, detail="File not found")
        
#         # Get file_info
#         file_info_response = await db.client.from_("file_info").select(
#             "*"
#         ).eq("file_id", file_id).eq("user_id", user_id).execute()
        
#         file_info = file_info_response.data or []
        
#         return {
#             "file_id": file_id,
#             "file_name": file_response.data.get("original_filename"),
#             "file_info": file_info
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error fetching file info: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error fetching file info: {str(e)}")

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
# Template Update
# ============================================================================
@router.post("/update", response_model=ApiResponse)
async def update_template_with_files(
    template_id: str = Query(...),
    user_id: str = Query(...),
    file_ids: List[str] = Query(..., description="List of file IDs to update template with"),
    db: DatabaseService = Depends(get_database_service)
):
    """Update existing template with new files"""
    try:
        logger.info(f"🔄 Starting template update: {template_id} with {len(file_ids)} files")
        
        # Verify template exists and user has access
        template_response = await db.client.from_("templates").select(
            "*, folders(user_id, name)"
        ).eq("id", template_id).single().execute()
        
        if not template_response.data:
            raise HTTPException(status_code=404, detail="Template not found")
        
        template_data = template_response.data
        folder = template_data.get("folders", {})
        
        if folder.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Verify all files exist and belong to user
        files_response = await db.client.from_("files").select(
            "id, original_filename, status, folder_id, user_id"
        ).in_("id", file_ids).eq("user_id", user_id).execute()
        
        if not files_response.data or len(files_response.data) != len(file_ids):
            raise HTTPException(status_code=400, detail="Some files not found or access denied")
        
        # Check file status for information (but don't reject unprocessed files)
        unprocessed_files = [f for f in files_response.data if f["status"] not in ["processed", "clauses_ready"]]
        processed_files = [f for f in files_response.data if f["status"] in ["processed", "clauses_ready"]]
        
        logger.info(f"Template update: {len(processed_files)} files ready, {len(unprocessed_files)} files need processing")
        
        # Check for existing active update jobs for this template
        existing_jobs = await db.client.from_("jobs").select(
            "id, status, metadata, celery_task_id"
        ).eq("user_id", user_id).eq("job_type", "template_update").in_(
            "status", ["pending", "processing"]
        ).execute()
        
        if existing_jobs.data:
            for job in existing_jobs.data:
                job_metadata = job.get("metadata", {})
                if job_metadata.get("template_id") == template_id:
                    logger.info(f"Found existing active update job {job['id']} for template {template_id}")
                    return ApiResponse(
                        success=True,
                        message="Template update already in progress",
                        data={
                            "update_job_id": job["id"],
                            "status": job["status"],
                            "existing_job": True
                        }
                    )
        
        # Create job record
        update_job_id = await create_job(
            user_id=user_id,
            job_type="template_update",
            metadata={
                "template_id": template_id,
                "template_name": template_data.get("name"),
                "file_ids": file_ids,
                "total_files": len(file_ids),
                "folder_id": template_data.get("folder_id")
            },
            total_steps=3
        )
        
        # Start Celery task
        task_result = template_update_task.delay(update_job_id)
        
        # Update job with Celery task ID
        await db.client.from_("jobs").update({
            "celery_task_id": task_result.id,
            "status": "processing",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", update_job_id).execute()
        
        return ApiResponse(
            success=True,
            message="Template update job created successfully",
            data={
                "update_job_id": update_job_id,
                "status": "pending",
                "template_id": template_id,
                "files_count": len(file_ids),
                "files_ready": len(processed_files),
                "files_need_processing": len(unprocessed_files),
                "estimated_duration": "2-3 minutes" if len(unprocessed_files) == 0 else "3-5 minutes"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting template update: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error starting template update: {str(e)}")

@router.get("/update/{update_job_id}/status")
async def get_update_status(
    update_job_id: str,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service)
):
    """Get the status of a template update job"""
    try:
        job_status = await get_job_status_util(update_job_id)
        
        if "error" in job_status:
            return {
                "update_job_id": update_job_id,
                "status": "error",
                "progress": 0,
                "message": job_status.get("error", "Unknown error"),
                "estimated_completion": ""
            }
        
        # Extract metadata
        metadata = job_status.get("metadata", {})
        template_name = metadata.get("template_name", "Unknown Template")
        
        # Generate status message
        status = job_status.get("status", "pending")
        current_step_name = job_status.get("current_step_name", "")
        
        if status == "completed":
            message = f"Template '{template_name}' updated successfully"
        elif status == "failed":
            message = job_status.get("error_message", "Template update failed")
        elif current_step_name == "process_files":
            message = "Ensuring files are ready for update..."
        elif current_step_name == "update_template":
            message = f"Updating template '{template_name}' with AI..."
        elif current_step_name == "extract_clauses":
            message = "Extracting clauses from updated template..."
        else:
            message = "Preparing template update..."
        
        # Estimate completion
        estimated_completion = ""
        if status == "processing":
            progress = job_status.get("progress", 0)
            if progress < 50:
                estimated_completion = "2-3 minutes"
            elif progress < 80:
                estimated_completion = "1-2 minutes"
            else:
                estimated_completion = "Less than 1 minute"
        
        return {
            "update_job_id": update_job_id,
            "status": status,
            "progress": job_status.get("progress", 0),
            "message": message,
            "template_name": template_name,
            "estimated_completion": estimated_completion
        }
        
    except Exception as e:
        logger.error(f"Error getting update status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error getting update status: {str(e)}")


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

