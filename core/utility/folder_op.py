"""
Folder operation utilities for the Legal Template Generator.
Centralizes all folder-related database operations and management.
"""

from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime
from loguru import logger
from core.database import get_database_service

logger.add("logs/folder_utilities.log", rotation="10 MB", level="DEBUG")

# ============================================================================
# FOLDER CRUD OPERATIONS
# ============================================================================

async def create_folder(
    user_id: str,
    name: str,
    description: Optional[str] = None,
    color: Optional[str] = None,
    folder_id: Optional[str] = None
) -> Optional[str]:
    """Create a new folder"""
    try:
        db = get_database_service()
        
        folder_data = {
            "id": folder_id or str(uuid4()),
            "user_id": user_id,
            "name": name,
            "description": description,
            "color": color,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("folders").insert(folder_data).execute()
        
        if response.data:
            logger.info(f"Created folder {folder_data['id']}: {name}")
            return folder_data["id"]
        return None
        
    except Exception as e:
        logger.error(f"Error creating folder '{name}': {e}")
        return None

async def get_folder_by_id(folder_id: str, user_id: str, include_file_count: bool = False) -> Optional[Dict[str, Any]]:
    """Get folder by ID with optional file count"""
    try:
        db = get_database_service()
        
        response = await db.client.from_("folders").select("*").eq("id", folder_id).eq("user_id", user_id).single().execute()
        
        if not response.data:
            return None
            
        folder_data = response.data
        
        if include_file_count:
            # Get file count for this folder
            files_response = await db.client.from_("files").select("id", count="exact").eq("folder_id", folder_id).eq("user_id", user_id).execute()
            folder_data["file_count"] = files_response.count or 0
        
        return folder_data
        
    except Exception as e:
        logger.error(f"Error getting folder {folder_id}: {e}")
        return None

async def get_folders_by_user(user_id: str, include_file_counts: bool = True) -> List[Dict[str, Any]]:
    """Get all folders for a user with optional file counts"""
    try:
        db = get_database_service()
        
        folders_response = await db.client.from_("folders").select("*").eq("user_id", user_id).order("updated_at", desc=True).execute()
        
        if not folders_response.data:
            return []
        
        folders = []
        for folder in folders_response.data:
            folder_data = dict(folder)
            
            if include_file_counts:
                # Get file count for each folder
                files_response = await db.client.from_("files").select("id", count="exact").eq("folder_id", folder["id"]).eq("user_id", user_id).execute()
                folder_data["file_count"] = files_response.count or 0
            
            folders.append(folder_data)
        
        return folders
        
    except Exception as e:
        logger.error(f"Error getting folders for user {user_id}: {e}")
        return []

async def update_folder(
    folder_id: str,
    user_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    color: Optional[str] = None
) -> bool:
    """Update folder information"""
    try:
        db = get_database_service()
        
        # Build update data
        update_data = {"updated_at": datetime.utcnow().isoformat()}
        
        if name is not None:
            update_data["name"] = name
        if description is not None:
            update_data["description"] = description
        if color is not None:
            update_data["color"] = color
        
        response = await db.client.from_("folders").update(update_data).eq("id", folder_id).eq("user_id", user_id).execute()
        
        if response.data:
            logger.info(f"Updated folder {folder_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error updating folder {folder_id}: {e}")
        return False

async def delete_folder_with_files(folder_id: str, user_id: str) -> bool:
    """Delete folder and all its files with storage cleanup"""
    try:
        db = get_database_service()
        
        # Verify folder exists
        folder_data = await get_folder_by_id(folder_id, user_id)
        if not folder_data:
            logger.warning(f"Folder {folder_id} not found for deletion")
            return False
        
        # Get all files in the folder
        files_response = await db.client.from_("files").select("id, storage_path").eq("folder_id", folder_id).eq("user_id", user_id).execute()
        
        # Delete files from storage
        if files_response.data:
            storage_paths = [file_data["storage_path"] for file_data in files_response.data if file_data.get("storage_path")]
            
            if storage_paths:
                try:
                    await db.service.storage.from_("documents").remove(storage_paths)
                    logger.debug(f"Deleted {len(storage_paths)} files from storage")
                except Exception as storage_error:
                    logger.warning(f"Failed to delete files from storage: {storage_error}")
            
            # Delete file records (this will cascade to related tables)
            await db.client.from_("files").delete().eq("folder_id", folder_id).eq("user_id", user_id).execute()
            logger.info(f"Deleted {len(files_response.data)} files from folder {folder_id}")
        
        # Delete folder
        await db.client.from_("folders").delete().eq("id", folder_id).eq("user_id", user_id).execute()
        logger.info(f"Deleted folder {folder_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error deleting folder {folder_id}: {e}")
        return False

# ============================================================================
# FOLDER ACCESS AND VALIDATION
# ============================================================================

async def check_folder_exists(folder_id: str, user_id: str) -> bool:
    """Check if folder exists for user"""
    try:
        db = get_database_service()
        response = await db.client.from_("folders").select("id").eq("id", folder_id).eq("user_id", user_id).execute()
        return bool(response.data)
        
    except Exception as e:
        logger.error(f"Error checking folder existence {folder_id}: {e}")
        return False

async def check_user_folder_access(folder_id: str, user_id: str) -> bool:
    """Check if user has access to folder"""
    return await check_folder_exists(folder_id, user_id)

async def get_folder_name(folder_id: str, user_id: str) -> Optional[str]:
    """Get folder name"""
    try:
        db = get_database_service()
        response = await db.client.from_("folders").select("name").eq("id", folder_id).eq("user_id", user_id).single().execute()
        return response.data.get("name") if response.data else None
        
    except Exception as e:
        logger.error(f"Error getting folder name {folder_id}: {e}")
        return None

async def validate_folder_name(name: str, user_id: str, exclude_folder_id: Optional[str] = None) -> bool:
    """Check if folder name is unique for user"""
    try:
        db = get_database_service()
        query = db.client.from_("folders").select("id").eq("user_id", user_id).eq("name", name)
        
        if exclude_folder_id:
            query = query.neq("id", exclude_folder_id)
            
        response = await query.execute()
        return len(response.data) == 0
        
    except Exception as e:
        logger.error(f"Error validating folder name '{name}': {e}")
        return False

# ============================================================================
# FOLDER STATISTICS AND ANALYTICS
# ============================================================================

async def get_folder_statistics(folder_id: str, user_id: str) -> Dict[str, Any]:
    """Get comprehensive folder statistics"""
    try:
        db = get_database_service()
        
        # Get basic folder info
        folder_data = await get_folder_by_id(folder_id, user_id)
        if not folder_data:
            return {}
        
        # Get file statistics
        files_response = await db.client.from_("files").select("file_size, file_type, status, created_at").eq("folder_id", folder_id).eq("user_id", user_id).execute()
        
        files_data = files_response.data or []
        
        # Calculate statistics
        total_files = len(files_data)
        total_size = sum(file_data.get("file_size", 0) for file_data in files_data)
        
        # Status counts
        status_counts = {}
        for file_data in files_data:
            status = file_data.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # File type counts
        type_counts = {}
        for file_data in files_data:
            file_type = file_data.get("file_type", "unknown")
            type_counts[file_type] = type_counts.get(file_type, 0) + 1
        
        return {
            "folder_id": folder_id,
            "folder_name": folder_data.get("name"),
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "status_counts": status_counts,
            "type_counts": type_counts,
            "average_file_size": round(total_size / total_files, 2) if total_files > 0 else 0,
            "created_at": folder_data.get("created_at"),
            "updated_at": folder_data.get("updated_at")
        }
        
    except Exception as e:
        logger.error(f"Error getting folder statistics for {folder_id}: {e}")
        return {}

async def get_user_folder_stats(user_id: str) -> Dict[str, Any]:
    """Get overall folder statistics for a user"""
    try:
        db = get_database_service()
        
        # Get folder count
        folders_response = await db.client.from_("folders").select("id", count="exact").eq("user_id", user_id).execute()
        total_folders = folders_response.count or 0
        
        # Get file statistics across all folders
        files_response = await db.client.from_("files").select("status, file_size, file_type").eq("user_id", user_id).execute()
        
        files_data = files_response.data or []
        
        # Calculate file statistics
        total_files = len(files_data)
        total_size = sum(file_data.get("file_size", 0) for file_data in files_data)
        
        # Status counts
        status_counts = {}
        for file_data in files_data:
            status = file_data.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # File type counts
        type_counts = {}
        for file_data in files_data:
            file_type = file_data.get("file_type", "unknown")
            type_counts[file_type] = type_counts.get(file_type, 0) + 1
        
        return {
            "total_folders": total_folders,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "status_counts": status_counts,
            "type_counts": type_counts,
            "average_file_size": round(total_size / total_files, 2) if total_files > 0 else 0,
            "files_per_folder": round(total_files / total_folders, 2) if total_folders > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting user folder stats for {user_id}: {e}")
        return {
            "total_folders": 0,
            "total_files": 0,
            "total_size_bytes": 0,
            "total_size_mb": 0,
            "status_counts": {},
            "type_counts": {},
            "average_file_size": 0,
            "files_per_folder": 0
        }

async def get_folder_activity_summary(folder_id: str, user_id: str, days: int = 30) -> Dict[str, Any]:
    """Get folder activity summary for the last N days"""
    try:
        db = get_database_service()
        
        # Calculate date threshold
        from datetime import timedelta
        threshold_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Get recent files
        files_response = await db.client.from_("files").select("created_at, status, file_size").eq("folder_id", folder_id).eq("user_id", user_id).gte("created_at", threshold_date).execute()
        
        files_data = files_response.data or []
        
        # Calculate activity metrics
        recent_uploads = len(files_data)
        recent_size = sum(file_data.get("file_size", 0) for file_data in files_data)
        
        # Status distribution of recent files
        recent_status_counts = {}
        for file_data in files_data:
            status = file_data.get("status", "unknown")
            recent_status_counts[status] = recent_status_counts.get(status, 0) + 1
        
        return {
            "folder_id": folder_id,
            "activity_period_days": days,
            "recent_uploads": recent_uploads,
            "recent_size_bytes": recent_size,
            "recent_size_mb": round(recent_size / 1024 / 1024, 2),
            "recent_status_counts": recent_status_counts,
            "activity_rate": round(recent_uploads / days, 2) if days > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting folder activity summary for {folder_id}: {e}")
        return {
            "folder_id": folder_id,
            "activity_period_days": days,
            "recent_uploads": 0,
            "recent_size_bytes": 0,
            "recent_size_mb": 0,
            "recent_status_counts": {},
            "activity_rate": 0
        }

# ============================================================================
# FOLDER MAINTENANCE AND CLEANUP
# ============================================================================

async def cleanup_empty_folders(user_id: str) -> int:
    """Remove empty folders for a user"""
    try:
        db = get_database_service()
        
        # Get all folders for the user
        folders_response = await db.client.from_("folders").select("id, name").eq("user_id", user_id).execute()
        
        if not folders_response.data:
            return 0
        
        empty_folders = []
        
        # Check each folder for files
        for folder in folders_response.data:
            files_response = await db.client.from_("files").select("id", count="exact").eq("folder_id", folder["id"]).eq("user_id", user_id).execute()
            
            if (files_response.count or 0) == 0:
                empty_folders.append(folder["id"])
        
        # Delete empty folders
        if empty_folders:
            await db.client.from_("folders").delete().in_("id", empty_folders).eq("user_id", user_id).execute()
            logger.info(f"Cleaned up {len(empty_folders)} empty folders for user {user_id}")
        
        return len(empty_folders)
        
    except Exception as e:
        logger.error(f"Error cleaning up empty folders for user {user_id}: {e}")
        return 0

async def get_folders_by_name_pattern(user_id: str, pattern: str) -> List[Dict[str, Any]]:
    """Get folders matching a name pattern"""
    try:
        db = get_database_service()
        
        # Use PostgreSQL ILIKE for case-insensitive pattern matching
        response = await db.client.from_("folders").select("*").eq("user_id", user_id).ilike("name", f"%{pattern}%").execute()
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting folders by pattern '{pattern}': {e}")
        return []

async def get_recently_updated_folders(user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get recently updated folders"""
    try:
        db = get_database_service()
        
        response = await db.client.from_("folders").select("*").eq("user_id", user_id).order("updated_at", desc=True).limit(limit).execute()
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting recently updated folders: {e}")
        return []

async def bulk_update_folder_color(folder_ids: List[str], user_id: str, color: str) -> int:
    """Update color for multiple folders"""
    try:
        db = get_database_service()
        
        update_data = {
            "color": color,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("folders").update(update_data).in_("id", folder_ids).eq("user_id", user_id).execute()
        
        updated_count = len(response.data) if response.data else 0
        logger.info(f"Bulk updated color for {updated_count} folders")
        return updated_count
        
    except Exception as e:
        logger.error(f"Error bulk updating folder colors: {e}")
        return 0
