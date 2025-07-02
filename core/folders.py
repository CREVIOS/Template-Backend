from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional
from uuid import uuid4
from datetime import datetime, timedelta
from loguru import logger

from core.database import DatabaseService, get_database_service
from core.models import FolderCreate, FolderUpdate, FolderWithCount, FolderStats, ApiResponse

router = APIRouter()
logger.add("logs/folders.log", rotation="1 MB", level="DEBUG")

# Dependency to inject DatabaseService
async def get_db(
    db: DatabaseService = Depends(get_database_service)
) -> DatabaseService:
    return db

@router.get("/", response_model=List[FolderWithCount])
async def get_folders(
    user_id: str = Query(..., description="User ID"),  
    db: DatabaseService = Depends(get_db)
):
    """Get all folders for a user with file counts"""
    try:
        # Fetch folders using anon client
        folders_resp = await db.anon.table("folders") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("updated_at", desc=True) \
            .execute()

        if not folders_resp.data:
            return []

        # Build result with file counts via anon
        result: List[dict] = []
        for folder in folders_resp.data:
            files_resp = await db.anon.table("files") \
                .select("id", count="exact") \
                .eq("folder_id", folder["id"]) \
                .eq("user_id", user_id) \
                .execute()
            count = files_resp.count or 0
            result.append({
                **folder,
                "file_count": count
            })

        return result

    except Exception as e:
        logger.error(f"Error in get_folders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{folder_id}")
async def get_folder_by_id(
    folder_id: str,
    user_id: str = Query(..., description="User ID"),
    db: DatabaseService = Depends(get_database_service)
):
    """Get folder by ID with file counts"""
    try:
        result = await db.client.from_("folders").select(
            "*, files(count)"
        ).eq("id", folder_id).eq("user_id", user_id).single().execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Folder not found")
        
        return result.data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=ApiResponse)
async def create_folder(
    folder: FolderCreate,
    user_id: str = Query(..., description="User ID"),
    db: DatabaseService = Depends(get_db)
):
    """Create a new folder"""
    folder_data = {
        "id": str(uuid4()),
        "user_id": user_id,
        "name": folder.name,
        "description": folder.description,
        "color": folder.color,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    resp = await db.anon.table("folders").insert(folder_data).execute()
    if not resp.data:
        raise HTTPException(400, "Failed to create folder")
    return ApiResponse(success=True, message="Folder created", data={"folder_id": folder_data["id"]})

@router.delete("/{folder_id}", response_model=ApiResponse)
async def delete_folder(
    folder_id: str,
    user_id: str = Query(..., description="User ID"),
    db: DatabaseService = Depends(get_db)
):
    """Delete a folder and its files"""
    # Verify folder exists using anon
    folder_resp = await db.anon.table("folders") \
        .select("id") \
        .eq("id", folder_id) \
        .eq("user_id", user_id) \
        .single() \
        .execute()
    if not folder_resp.data:
        raise HTTPException(404, "Folder not found")

    # Fetch files to delete
    files_resp = await db.anon.table("files") \
        .select("id, storage_path") \
        .eq("folder_id", folder_id) \
        .eq("user_id", user_id) \
        .execute()
    # Remove files from storage using service role
    if files_resp.data:
        paths = [f["storage_path"] for f in files_resp.data if f.get("storage_path")]
        if paths:
            try:
                await db.service.storage.from_("documents").remove(paths)
            except Exception as storage_err:
                logger.warning(f"Failed to remove files from storage: {storage_err}")
        # Delete file records via anon
        await db.anon.table("files") \
            .delete() \
            .eq("folder_id", folder_id) \
            .eq("user_id", user_id) \
            .execute()

    # Delete folder via anon
    await db.anon.table("folders") \
        .delete() \
        .eq("id", folder_id) \
        .eq("user_id", user_id) \
        .execute()

    return ApiResponse(success=True, message="Folder and files deleted")

@router.get("/stats", response_model=FolderStats)
async def get_user_stats(
    user_id: str = Query(..., description="User ID"),
    db: DatabaseService = Depends(get_db)
):
    """Get overall user stats"""
    # Folder count via anon
    fr = await db.anon.table("folders").select("id", count="exact").eq("user_id", user_id).execute()
    # Files counts via anon
    file_q = db.anon.table("files").select("id", count="exact").eq("user_id", user_id)
    r_all = await file_q.execute()
    r_processed = await file_q.eq("status", "processed").execute()
    r_pending   = await file_q.eq("status", "pending").execute()
    r_error     = await file_q.eq("status", "error").execute()

    return FolderStats(
        totalFolders=fr.count or 0,
        totalFiles=r_all.count or 0,
        totalProcessed=r_processed.count or 0,
        totalPending=r_pending.count or 0,
        totalErrors=r_error.count or 0
    )

@router.put("/{folder_id}", response_model=ApiResponse)
async def update_folder(
    folder_id: str,
    folder: FolderUpdate,
    user_id: str = Query(..., description="User ID"),
    db: DatabaseService = Depends(get_db)
):
    """Update folder fields"""
    # Verify existence via anon
    existing = await db.anon.table("folders").select("id").eq("id", folder_id).eq("user_id", user_id).single().execute()
    if not existing.data:
        raise HTTPException(404, "Folder not found")
    updates = {"updated_at": datetime.utcnow().isoformat()}
    if folder.name is not None: updates["name"] = folder.name
    if folder.description is not None: updates["description"] = folder.description
    if folder.color is not None: updates["color"] = folder.color
    await db.anon.table("folders").update(updates).eq("id", folder_id).eq("user_id", user_id).execute()
    return ApiResponse(success=True, message="Folder updated")
