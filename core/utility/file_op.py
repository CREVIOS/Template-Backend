"""
File operation utilities for the Legal Template Generator.
Centralizes all file-related database operations and storage management.
"""

import os
import hashlib
from typing import List, Dict, Any, Optional, Union
from uuid import uuid4
from datetime import datetime
from loguru import logger
from core.database import get_database_service, DatabaseService

logger.add("logs/file_utilities.log", rotation="10 MB", level="DEBUG")

# ============================================================================
# FILE CRUD OPERATIONS
# ============================================================================

async def create_file_record(
    user_id: str,
    folder_id: str,
    original_filename: str,
    file_size: int,
    file_type: str,
    storage_path: str,
    storage_url: str,
    file_id: Optional[str] = None,
    status: str = "uploaded"
) -> Optional[str]:
    """Create a new file record in the database"""
    try:
        db = get_database_service()
        
        file_record = {
            "id": file_id or str(uuid4()),
            "user_id": user_id,
            "folder_id": folder_id,
            "original_filename": original_filename,
            "file_size": file_size,
            "file_type": file_type,
            "storage_path": storage_path,
            "storage_url": storage_url,
            "status": status,
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

async def get_files_by_folder(folder_id: str, user_id: str, include_stats: bool = True) -> List[Dict[str, Any]]:
    """Get all files in a folder with optional statistics"""
    try:
        db = get_database_service()
        select_fields = "*"
        if include_stats:
            select_fields = "*, markdown_content(word_count, created_at)"
        
        response = await db.client.from_("files").select(select_fields).eq("folder_id", folder_id).eq("user_id", user_id).order("created_at", desc=True).execute()
        
        if not response.data:
            return []
            
        files = []
        for file_data in response.data:
            if include_stats:
                markdown_info = file_data.get("markdown_content")
                if markdown_info and len(markdown_info) > 0:
                    file_data["word_count"] = markdown_info[0].get("word_count")
                    file_data["markdown_created_at"] = markdown_info[0].get("created_at")
            files.append(file_data)
            
        return files
        
    except Exception as e:
        logger.error(f"Error getting files for folder {folder_id}: {e}")
        return []

async def get_files_by_ids(file_ids: List[str], user_id: str) -> List[Dict[str, Any]]:
    """Get multiple files by their IDs"""
    try:
        db = get_database_service()
        response = await db.client.from_("files").select("*").in_("id", file_ids).eq("user_id", user_id).execute()
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting files by IDs: {e}")
        return []

async def update_file_record(file_id: str, user_id: str, update_data: Dict[str, Any]) -> bool:
    """Update file record with provided data"""
    try:
        db = get_database_service()
        
        # Always update the updated_at timestamp
        update_data["updated_at"] = datetime.utcnow().isoformat()
        
        response = await db.client.from_("files").update(update_data).eq("id", file_id).eq("user_id", user_id).execute()
        
        if response.data:
            logger.debug(f"Updated file {file_id} with data: {list(update_data.keys())}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error updating file {file_id}: {e}")
        return False

async def delete_file_with_cleanup(file_id: str, user_id: str) -> bool:
    """Delete file and all related data with storage cleanup"""
    try:
        db = get_database_service()
        
        # Get file info first
        file_data = await get_file_by_id(file_id, user_id)
        if not file_data:
            logger.warning(f"File {file_id} not found for deletion")
            return False
            
        # Delete from storage if exists
        if file_data.get("storage_path"):
            try:
                await db.service.storage.from_("documents").remove([file_data["storage_path"]])
                logger.debug(f"Deleted file from storage: {file_data['storage_path']}")
            except Exception as storage_error:
                logger.warning(f"Failed to delete file from storage: {storage_error}")
                
        # Delete file record (cascades to related tables)
        await db.client.from_("files").delete().eq("id", file_id).eq("user_id", user_id).execute()
        logger.info(f"Deleted file {file_id} and related data")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting file {file_id}: {e}")
        return False

# ============================================================================
# FILE STATUS MANAGEMENT
# ============================================================================

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
            
        response = await db.client.from_("files").update(update_data).eq("id", file_id).eq("user_id", user_id).execute()
        
        if response.data:
            logger.debug(f"Updated file {file_id} status to {status}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error updating file {file_id} status: {e}")
        return False

async def get_files_by_status(user_id: str, status: str, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get files filtered by status"""
    try:
        db = get_database_service()
        query = db.client.from_("files").select("*").eq("user_id", user_id).eq("status", status)
        
        if folder_id:
            query = query.eq("folder_id", folder_id)
            
        response = await query.execute()
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting files by status {status}: {e}")
        return []

async def get_file_status_counts(user_id: str, folder_id: Optional[str] = None) -> Dict[str, int]:
    """Get file counts grouped by status"""
    try:
        db = get_database_service()
        query = db.client.from_("files").select("status").eq("user_id", user_id)
        
        if folder_id:
            query = query.eq("folder_id", folder_id)
            
        response = await query.execute()
        
        status_counts = {}
        if response.data:
            for file_data in response.data:
                status = file_data.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                
        return status_counts
        
    except Exception as e:
        logger.error(f"Error getting file status counts: {e}")
        return {}

# ============================================================================
# FILE VALIDATION AND UTILITIES
# ============================================================================

def validate_file_type(filename: str, allowed_types: List[str]) -> bool:
    """Validate if file type is allowed"""
    if not filename:
        return False
        
    file_extension = os.path.splitext(filename)[1].lower()
    return file_extension in allowed_types

def generate_file_hash(file_content: bytes) -> str:
    """Generate SHA256 hash of file content for deduplication"""
    return hashlib.sha256(file_content).hexdigest()

def generate_storage_path(user_id: str, folder_id: str, file_id: str, filename: str) -> str:
    """Generate consistent storage path for file"""
    file_extension = os.path.splitext(filename)[1] if filename else ""
    return f"{user_id}/{folder_id}/{file_id}{file_extension}"

async def check_file_exists(file_id: str, user_id: str) -> bool:
    """Check if file exists"""
    try:
        db = get_database_service()
        response = await db.client.from_("files").select("id").eq("id", file_id).eq("user_id", user_id).execute()
        return bool(response.data)
        
    except Exception as e:
        logger.error(f"Error checking file existence {file_id}: {e}")
        return False

async def check_user_file_access(file_id: str, user_id: str) -> bool:
    """Check if user has access to file"""
    return await check_file_exists(file_id, user_id)

# ============================================================================
# FILE STATISTICS AND ANALYTICS
# ============================================================================

async def get_file_statistics(user_id: str, folder_id: Optional[str] = None) -> Dict[str, Any]:
    """Get comprehensive file statistics"""
    try:
        db = get_database_service()
        
        # Base query
        query = db.client.from_("files").select("file_size, file_type, status, created_at").eq("user_id", user_id)
        
        if folder_id:
            query = query.eq("folder_id", folder_id)
            
        response = await query.execute()
        
        files_data = response.data or []
        
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
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "status_counts": status_counts,
            "type_counts": type_counts,
            "average_file_size": round(total_size / total_files, 2) if total_files > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting file statistics: {e}")
        return {
            "total_files": 0,
            "total_size_bytes": 0,
            "total_size_mb": 0,
            "status_counts": {},
            "type_counts": {},
            "average_file_size": 0
        }

# ============================================================================
# FILE CLEANUP AND MAINTENANCE
# ============================================================================

async def cleanup_orphaned_files(user_id: str) -> int:
    """Clean up files that reference non-existent folders"""
    try:
        db = get_database_service()
        
        # Get all files for the user
        files_response = await db.client.from_("files").select("id, folder_id").eq("user_id", user_id).execute()
        
        if not files_response.data:
            return 0
        
        # Get all valid folder IDs
        folders_response = await db.client.from_("folders").select("id").eq("user_id", user_id).execute()
        valid_folder_ids = set(folder["id"] for folder in folders_response.data) if folders_response.data else set()
        
        # Find orphaned files
        orphaned_files = []
        for file_data in files_response.data:
            if file_data["folder_id"] not in valid_folder_ids:
                orphaned_files.append(file_data["id"])
        
        # Delete orphaned files
        cleanup_count = 0
        for file_id in orphaned_files:
            if await delete_file_with_cleanup(file_id, user_id):
                cleanup_count += 1
        
        logger.info(f"Cleaned up {cleanup_count} orphaned files for user {user_id}")
        return cleanup_count
        
    except Exception as e:
        logger.error(f"Error cleaning up orphaned files for user {user_id}: {e}")
        return 0

async def get_files_needing_processing(user_id: str) -> List[Dict[str, Any]]:
    """Get files that need processing"""
    try:
        db = get_database_service()
        response = await db.client.from_("files").select("*").eq("user_id", user_id).in_("status", ["uploaded", "queued", "processing"]).order("created_at", desc=False).execute()
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting files needing processing: {e}")
        return []

async def bulk_update_file_status(file_ids: List[str], user_id: str, status: str) -> int:
    """Update status for multiple files"""
    try:
        db = get_database_service()
        
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("files").update(update_data).in_("id", file_ids).eq("user_id", user_id).execute()
        
        updated_count = len(response.data) if response.data else 0
        logger.info(f"Bulk updated {updated_count} files to status {status}")
        return updated_count
        
    except Exception as e:
        logger.error(f"Error bulk updating file status: {e}")
        return 0
