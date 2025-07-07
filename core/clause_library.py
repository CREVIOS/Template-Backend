import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from core.database import DatabaseService, get_database_service
from core.redis_cache import RedisCacheService, get_cache_service
from core.models import ClauseLibrary
from core.api_config import APIConfiguration
from core.db_utilities import create_job, get_job_status as get_job_status_util
from fastapi.responses import JSONResponse
import asyncio
import json
from datetime import datetime



router = APIRouter()

@router.get("/folder/{folder_id}/clauses")
async def get_folder_clauses(
    folder_id: str,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service),
    cache: RedisCacheService = Depends(get_cache_service)
):
    """
    Get all clauses from clause_library for a folder with associated file information, with Redis caching.
    """
    print(f"[DEBUG] get_folder_clauses called with folder_id={folder_id} (type={type(folder_id)}), user_id={user_id} (type={type(user_id)})")
    
    # Try cache first (if available)
    try:
        cache_key = f"clause_library:folder:{folder_id}:user:{user_id}"
        cached = await asyncio.to_thread(cache.client.get, cache_key)
        if cached:
            print(f"[DEBUG] Returning {len(json.loads(cached))} clauses from cache.")
            return {"clauses": json.loads(cached)}
    except Exception as e:
        print(f"[DEBUG] Cache not available, skipping cache: {e}")
    
    # Fetch from DB with file information
    try:
        print(f"[DEBUG] Querying DB: folder_id={folder_id}, user_id={user_id}")
        
        # First get all clauses for the folder
        clauses_query = db.client.from_("clause_library").select("*").eq("user_id", user_id).eq("folder_id", folder_id).order("clause_type")
        clauses_result = await clauses_query.execute()
        clauses = clauses_result.data or []
        print(f"[DEBUG] DB returned {len(clauses)} clauses for folder_id={folder_id} and user_id={user_id}")
        
        # Get unique file_ids from clauses
        file_ids = list(set([clause.get("file_id") for clause in clauses if clause.get("file_id")]))
        print(f"[DEBUG] Found {len(file_ids)} unique file_ids: {file_ids}")
        
        # Fetch file information for all file_ids
        file_info_map = {}
        if file_ids:
            files_query = db.client.from_("files").select("id, original_filename, file_size, file_type, status, created_at, extracted_metadata").in_("id", file_ids).eq("user_id", user_id)
            files_result = await files_query.execute()
            files = files_result.data or []
            
            # Create a map of file_id -> file_info
            for file in files:
                file_info_map[file["id"]] = file
        
        # Enhance clauses with file information
        enhanced_clauses = []
        for clause in clauses:
            enhanced_clause = dict(clause)
            file_id = clause.get("file_id")
            if file_id and file_id in file_info_map:
                enhanced_clause["file_info"] = file_info_map[file_id]
                # print(f"[DEBUG] Added file info for clause {clause.get('id')}: {file_info_map[file_id].get('original_filename')}")
            else:
                enhanced_clause["file_info"] = None
                if file_id:
                    print(f"[DEBUG] No file info found for clause {clause.get('id')} with file_id {file_id}")
            enhanced_clauses.append(enhanced_clause)
        
        
        # Cache result (if available)
        try:
            await asyncio.to_thread(cache.client.setex, cache_key, 300, json.dumps(enhanced_clauses))  # 5 min TTL
        except Exception as e:
            print(f"[DEBUG] Failed to cache result: {e}")
        
        return {"clauses": enhanced_clauses}
    except Exception as e:
        import traceback
        print(f"[ERROR] Database query failed: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to fetch clauses: {str(e)}")


@router.put("/{clause_id}")
async def update_clause(
    clause_id: str,
    data: dict,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service),
    cache: RedisCacheService = Depends(get_cache_service)
):
    """
    Update a clause by ID.
    Only the clause_type and clause_text fields can be updated.
    """
    try:
        print(f"====== UPDATE CLAUSE API CALLED ======")
        print(f"Trying to update clause: id={clause_id}, user_id={user_id}")
        
        # Check if clause exists - first check only for ID match to debug
        raw_check = await db.client.from_('clause_library').select('*').eq('id', clause_id).execute()
        
        # Now check with proper user ID filter
        existing = await db.client.from_('clause_library').select('*').eq('id', clause_id).eq('user_id', user_id).execute()
        
        if not existing.data or len(existing.data) == 0:
            print(f"ERROR: Clause not found with id={clause_id} and user_id={user_id}")
            raise HTTPException(status_code=404, detail="Clause not found")
        
        print(f"Existing clause: {existing.data[0]}")
        
        # Get the folder_id for cache invalidation
        folder_id = existing.data[0].get("folder_id")
        print(f"Clause belongs to folder: {folder_id}")
        
        # Only update allowed fields
        update_data = {
            "clause_type": data.get("clause_type", existing.data[0]["clause_type"]),
            "clause_text": data.get("clause_text", existing.data[0]["clause_text"]),
        }
        
        
        # Update the clause
        result = await db.client.from_('clause_library').update(update_data).eq('id', clause_id).eq('user_id', user_id).execute()
        
        
        # Invalidate clause library cache for this specific folder
        if folder_id:
            try:
                cache_key = f"clause_library:folder:{folder_id}:user:{user_id}"
                await asyncio.to_thread(cache.client.delete, cache_key)
                print(f"Invalidated clause cache for folder: {folder_id}")
            except Exception as e:
                print(f"Warning: Failed to invalidate cache: {e}")
        else:
            print("Warning: No folder_id found, cannot invalidate specific cache")
        
        return {"message": "Clause updated successfully", "id": clause_id, "folder_id": folder_id}
    except HTTPException as he:
        print(f"HTTP Exception: {str(he)}")
        raise
    except Exception as e:
        print(f"Error updating clause: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update clause: {str(e)}")

@router.get("/{clause_id}")
async def get_clause(
    clause_id: str,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service)
):
    """
    Get a single clause by ID.
    This is useful for debugging and checking if a clause exists.
    """
    print(f"====== GET CLAUSE API CALLED ======")
    print(f"Checking if clause exists: id={clause_id}, user_id={user_id}")
    
    try:
        # First check without user filter
        raw_check = await db.client.from_('clause_library').select('*').eq('id', clause_id).execute()
        if raw_check.data and len(raw_check.data) > 0:
            print(f"Found clause with ID {clause_id} - User ID: {raw_check.data[0]['user_id']}")
            
        # Use execute() instead of single().execute() to avoid APIError
        result = await db.client.from_('clause_library').select('*').eq('id', clause_id).eq('user_id', user_id).execute()
        
        # Check if any results were found
        if not result.data or len(result.data) == 0:
            print(f"ERROR: Clause not found with id={clause_id} and user_id={user_id}")
            raise HTTPException(status_code=404, detail="Clause not found")
        
        # Verify user_id matches (extra security check)
        if result.data[0]["user_id"] != user_id:
            print(f"Matches requested user_id: False")
            raise HTTPException(status_code=403, detail="Not authorized to access this clause")
            
        print(f"Clause user_id: {result.data[0]['user_id']}")
        print(f"Matches requested user_id: True")
        print(f"Returning clause data: {result.data[0]}")
        
        # Return the first matching clause
        return result.data[0]
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error getting clause: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get clause: {str(e)}")

@router.delete("/{clause_id}")
async def delete_clause(
    clause_id: str,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service),
    cache: RedisCacheService = Depends(get_cache_service)
):
    """
    Delete a clause by ID.
    """
    try:
        print(f"====== DELETE CLAUSE API CALLED ======")
        print(f"Trying to delete clause: id={clause_id}, user_id={user_id}")
        
        # Check if clause exists and get folder_id for cache invalidation
        existing = await db.client.from_('clause_library').select('*').eq('id', clause_id).eq('user_id', user_id).execute()
        if not existing.data or len(existing.data) == 0:
            print(f"ERROR: Clause not found with id={clause_id} and user_id={user_id}")
            raise HTTPException(status_code=404, detail="Clause not found")
        
        # Get the folder_id for cache invalidation
        folder_id = existing.data[0].get("folder_id")
        print(f"Clause belongs to folder: {folder_id}")
        
        # Delete the clause
        result = await db.client.from_('clause_library').delete().eq('id', clause_id).eq('user_id', user_id).execute()
        
        # Invalidate clause library cache for this specific folder
        if folder_id:
            try:
                cache_key = f"clause_library:folder:{folder_id}:user:{user_id}"
                await asyncio.to_thread(cache.client.delete, cache_key)
                print(f"Invalidated clause cache for folder: {folder_id}")
            except Exception as e:
                print(f"Warning: Failed to invalidate cache: {e}")
        else:
            print("Warning: No folder_id found, cannot invalidate specific cache")
        
        return {"message": "Clause deleted successfully", "id": clause_id, "folder_id": folder_id}
    except HTTPException as he:
        print(f"HTTP Exception: {str(he)}")
        raise
    except Exception as e:
        print(f"Error deleting clause: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete clause: {str(e)}")

# Add the correct endpoint for folder clauses that the frontend expects
@router.get("/clause-library/folder/{folder_id}/clauses")
async def get_folder_clauses_alt(
    folder_id: str,
    user_id: str = Query(...),
    refresh: bool = Query(False),
    db: DatabaseService = Depends(get_database_service),
    cache: RedisCacheService = Depends(get_cache_service)
):
    """
    Alternative endpoint that matches the frontend's expected path.
    This calls the same implementation as the original endpoint with file information.
    """
    print(f"====== GET FOLDER CLAUSES API CALLED (Alternative path) ======")
    print(f"Folder ID: {folder_id}, User ID: {user_id}, Refresh: {refresh}")
    
    # If refresh is requested, invalidate cache first
    if refresh:
        try:
            cache_key = f"clause_library:folder:{folder_id}:user:{user_id}"
            await asyncio.to_thread(cache.client.delete, cache_key)
            print(f"[DEBUG] Cache invalidated for refresh request")
        except Exception as e:
            print(f"[DEBUG] Failed to invalidate cache: {e}")
    
    return await get_folder_clauses(folder_id, user_id, db, cache)

# Debug endpoint to check clause library status
@router.get("/debug/status")
async def debug_clause_library_status(
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service),
    cache: RedisCacheService = Depends(get_cache_service)
):
    """
    Debug endpoint to check clause library status and cache.
    """
    try:
        print(f"====== DEBUG CLAUSE LIBRARY STATUS ======")
        print(f"User ID: {user_id}")
        
        # Get all clauses for this user
        all_clauses = await db.client.from_('clause_library').select('*').eq('user_id', user_id).execute()
        clauses = all_clauses.data or []
        
        # Group by folder
        folders = {}
        for clause in clauses:
            folder_id = clause.get('folder_id', 'unknown')
            if folder_id not in folders:
                folders[folder_id] = []
            folders[folder_id].append(clause)
        
        # Check cache status for each folder
        cache_status = {}
        for folder_id in folders.keys():
            try:
                cache_key = f"clause_library:folder:{folder_id}:user:{user_id}"
                cached = await asyncio.to_thread(cache.client.get, cache_key)
                cache_status[folder_id] = {
                    "cached": cached is not None,
                    "clause_count": len(folders[folder_id])
                }
            except Exception as e:
                cache_status[folder_id] = {
                    "cached": False,
                    "clause_count": len(folders[folder_id]),
                    "cache_error": str(e)
                }
        
        return {
            "user_id": user_id,
            "total_clauses": len(clauses),
            "folders": folders,
            "cache_status": cache_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"Error in debug endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")
    

@router.get("/file/{file_id}/info")
async def get_file_info(
    file_id: str,
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service)
):
    """Get extracted_metadata JSON for a specific file"""
    try:
        print(f"[INFO] Getting file info for file_id: {file_id}, user_id: {user_id}")
        
        # Verify file access and fetch extracted_metadata
        # Use .execute() instead of .single().execute() to avoid APIError
        file_response = await db.client.from_("files").select(
            "id, original_filename, user_id, extracted_metadata, file_size, file_type"
        ).eq("id", file_id).eq("user_id", user_id).execute()
        
        if not file_response.data or len(file_response.data) == 0:
            print(f"[ERROR] File not found: file_id={file_id}, user_id={user_id}")
            raise HTTPException(status_code=404, detail="File not found")
        
        file_data = file_response.data[0]
        extracted_metadata = file_data.get("extracted_metadata")
        
        print(f"[INFO] Successfully fetched file info for: {file_id}")
        
        return {
            "file_id": file_id,
            "file_name": file_data.get("original_filename"),
            "original_filename": file_data.get("original_filename"),
            "file_size": file_data.get("file_size"),
            "file_type": file_data.get("file_type"),
            "extracted_metadata": extracted_metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Error fetching file info: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error fetching file info: {str(e)}")


# Debug endpoint to list all files for a user
@router.get("/debug/files")
async def debug_list_user_files(
    user_id: str = Query(...),
    db: DatabaseService = Depends(get_database_service)
):
    """Debug endpoint to list all files for a user"""
    try:
        print(f"[DEBUG] Listing all files for user: {user_id}")
        
        # Get all files for this user
        files_response = await db.client.from_("files").select(
            "id, original_filename, user_id, folder_id, file_size, file_type, status, created_at"
        ).eq("user_id", user_id).execute()
        
        files = files_response.data or []
        
        print(f"[DEBUG] Found {len(files)} files for user: {user_id}")
        
        return {
            "user_id": user_id,
            "total_files": len(files),
            "files": files,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"[ERROR] Error in debug files endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")

