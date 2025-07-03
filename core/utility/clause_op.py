"""
Clause operation utilities for the Legal Template Generator.
Centralizes all clause library operations and management.
"""

import hashlib
import json
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime
from loguru import logger
from core.database import get_database_service

logger.add("logs/clause_utilities.log", rotation="10 MB", level="DEBUG")

# ============================================================================
# CLAUSE LIBRARY OPERATIONS
# ============================================================================

async def save_clause_to_library(
    user_id: str,
    file_id: str,
    folder_id: str,
    clause_type: str,
    clause_text: str,
    clause_metadata: Optional[Dict[str, Any]] = None,
    clause_id: Optional[str] = None
) -> bool:
    """Save a clause to the clause library"""
    try:
        db = get_database_service()
        
        # Generate clause hash for deduplication
        clause_hash = hashlib.sha256(clause_text.encode('utf-8')).hexdigest()
        
        clause_record = {
            "id": clause_id or str(uuid4()),
            "user_id": user_id,
            "file_id": file_id,
            "folder_id": folder_id,
            "clause_type": clause_type,
            "clause_text": clause_text,
            "clause_metadata": clause_metadata or {},
            "clause_hash": clause_hash,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Add hash to metadata
        if clause_record["clause_metadata"]:
            clause_record["clause_metadata"]["clause_hash"] = clause_hash
        
        response = await db.client.from_("clause_library").insert(clause_record).execute()
        
        if response.data:
            logger.debug(f"Saved clause to library for file {file_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error saving clause to library: {e}")
        return False

async def save_clauses_bulk(clauses_data: List[Dict[str, Any]]) -> int:
    """Save multiple clauses to the library in bulk"""
    try:
        db = get_database_service()
        
        # Prepare clause records
        clause_records = []
        for clause_data in clauses_data:
            clause_text = clause_data.get("clause_text", "")
            if not clause_text:
                continue
                
            clause_hash = hashlib.sha256(clause_text.encode('utf-8')).hexdigest()
            
            clause_record = {
                "id": clause_data.get("id") or str(uuid4()),
                "user_id": clause_data["user_id"],
                "file_id": clause_data["file_id"],
                "folder_id": clause_data["folder_id"],
                "clause_type": clause_data.get("clause_type", "general"),
                "clause_text": clause_text,
                "clause_metadata": clause_data.get("clause_metadata", {}),
                "clause_hash": clause_hash,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Add hash to metadata
            if clause_record["clause_metadata"]:
                clause_record["clause_metadata"]["clause_hash"] = clause_hash
            
            clause_records.append(clause_record)
        
        if not clause_records:
            return 0
        
        response = await db.client.from_("clause_library").insert(clause_records).execute()
        
        saved_count = len(response.data) if response.data else 0
        logger.info(f"Bulk saved {saved_count} clauses to library")
        return saved_count
        
    except Exception as e:
        logger.error(f"Error bulk saving clauses: {e}")
        return 0

async def get_clauses_by_file(file_id: str, user_id: str, clause_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all clauses for a specific file"""
    try:
        db = get_database_service()
        
        query = db.client.from_("clause_library").select("*").eq("file_id", file_id).eq("user_id", user_id)
        
        if clause_type:
            query = query.eq("clause_type", clause_type)
            
        response = await query.order("clause_type", "created_at").execute()
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting clauses for file {file_id}: {e}")
        return []

async def get_clauses_by_folder(folder_id: str, user_id: str, clause_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all clauses for a specific folder"""
    try:
        db = get_database_service()
        
        query = db.client.from_("clause_library").select("*").eq("folder_id", folder_id).eq("user_id", user_id)
        
        if clause_type:
            query = query.eq("clause_type", clause_type)
            
        response = await query.order("clause_type", "created_at").execute()
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting clauses for folder {folder_id}: {e}")
        return []

async def get_clauses_by_type(user_id: str, clause_type: str, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all clauses of a specific type"""
    try:
        db = get_database_service()
        
        query = db.client.from_("clause_library").select("*").eq("user_id", user_id).eq("clause_type", clause_type)
        
        if folder_id:
            query = query.eq("folder_id", folder_id)
            
        response = await query.order("created_at", desc=True).execute()
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting clauses by type {clause_type}: {e}")
        return []

async def get_clause_by_id(clause_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific clause by ID"""
    try:
        db = get_database_service()
        
        response = await db.client.from_("clause_library").select("*").eq("id", clause_id).eq("user_id", user_id).single().execute()
        
        return response.data if response.data else None
        
    except Exception as e:
        logger.error(f"Error getting clause {clause_id}: {e}")
        return None

async def update_clause_metadata(clause_id: str, user_id: str, metadata: Dict[str, Any]) -> bool:
    """Update clause metadata"""
    try:
        db = get_database_service()
        
        response = await db.client.from_("clause_library").update({
            "clause_metadata": metadata,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", clause_id).eq("user_id", user_id).execute()
        
        if response.data:
            logger.debug(f"Updated metadata for clause {clause_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error updating clause metadata {clause_id}: {e}")
        return False

async def delete_clause(clause_id: str, user_id: str) -> bool:
    """Delete a clause from the library"""
    try:
        db = get_database_service()
        
        response = await db.client.from_("clause_library").delete().eq("id", clause_id).eq("user_id", user_id).execute()
        
        if response.data:
            logger.info(f"Deleted clause {clause_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error deleting clause {clause_id}: {e}")
        return False

async def delete_clauses_by_file(file_id: str, user_id: str) -> int:
    """Delete all clauses for a specific file"""
    try:
        db = get_database_service()
        
        response = await db.client.from_("clause_library").delete().eq("file_id", file_id).eq("user_id", user_id).execute()
        
        deleted_count = len(response.data) if response.data else 0
        logger.info(f"Deleted {deleted_count} clauses for file {file_id}")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error deleting clauses for file {file_id}: {e}")
        return 0

# ============================================================================
# CLAUSE ANALYSIS AND STATISTICS
# ============================================================================

async def get_clause_statistics(user_id: str, folder_id: Optional[str] = None) -> Dict[str, Any]:
    """Get comprehensive clause statistics"""
    try:
        db = get_database_service()
        
        query = db.client.from_("clause_library").select("clause_type, created_at").eq("user_id", user_id)
        
        if folder_id:
            query = query.eq("folder_id", folder_id)
            
        response = await query.execute()
        
        clauses_data = response.data or []
        
        # Calculate statistics
        total_clauses = len(clauses_data)
        
        # Clause type distribution
        type_counts = {}
        for clause in clauses_data:
            clause_type = clause.get("clause_type", "unknown")
            type_counts[clause_type] = type_counts.get(clause_type, 0) + 1
        
        # Most common clause types
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total_clauses": total_clauses,
            "clause_type_counts": type_counts,
            "most_common_types": sorted_types[:10],
            "unique_clause_types": len(type_counts),
            "folder_id": folder_id
        }
        
    except Exception as e:
        logger.error(f"Error getting clause statistics: {e}")
        return {
            "total_clauses": 0,
            "clause_type_counts": {},
            "most_common_types": [],
            "unique_clause_types": 0,
            "folder_id": folder_id
        }

async def find_similar_clauses(clause_text: str, user_id: str, folder_id: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """Find similar clauses using text similarity"""
    try:
        db = get_database_service()
        
        # Generate hash for exact match
        clause_hash = hashlib.sha256(clause_text.encode('utf-8')).hexdigest()
        
        query = db.client.from_("clause_library").select("*").eq("user_id", user_id)
        
        if folder_id:
            query = query.eq("folder_id", folder_id)
            
        response = await query.execute()
        
        clauses_data = response.data or []
        
        # Find exact hash matches first
        exact_matches = []
        similar_clauses = []
        
        for clause in clauses_data:
            clause_metadata = clause.get("clause_metadata", {})
            if clause_metadata.get("clause_hash") == clause_hash:
                exact_matches.append(clause)
            else:
                # Simple similarity check - could be enhanced with more sophisticated algorithms
                clause_words = set(clause.get("clause_text", "").lower().split())
                input_words = set(clause_text.lower().split())
                
                if clause_words and input_words:
                    intersection = clause_words & input_words
                    union = clause_words | input_words
                    similarity = len(intersection) / len(union) if union else 0
                    
                    if similarity > 0.3:  # 30% similarity threshold
                        clause["similarity_score"] = similarity
                        similar_clauses.append(clause)
        
        # Sort by similarity score
        similar_clauses.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        # Return exact matches first, then similar ones
        result = exact_matches + similar_clauses[:limit]
        
        return result[:limit]
        
    except Exception as e:
        logger.error(f"Error finding similar clauses: {e}")
        return []

async def get_clause_usage_stats(user_id: str, days: int = 30) -> Dict[str, Any]:
    """Get clause usage statistics for the last N days"""
    try:
        db = get_database_service()
        
        # Calculate date threshold
        from datetime import timedelta
        threshold_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        response = await db.client.from_("clause_library").select("clause_type, created_at").eq("user_id", user_id).gte("created_at", threshold_date).execute()
        
        clauses_data = response.data or []
        
        # Calculate statistics
        recent_clauses = len(clauses_data)
        
        # Type distribution for recent clauses
        type_counts = {}
        for clause in clauses_data:
            clause_type = clause.get("clause_type", "unknown")
            type_counts[clause_type] = type_counts.get(clause_type, 0) + 1
        
        return {
            "period_days": days,
            "recent_clauses": recent_clauses,
            "recent_type_counts": type_counts,
            "activity_rate": round(recent_clauses / days, 2) if days > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting clause usage stats: {e}")
        return {
            "period_days": days,
            "recent_clauses": 0,
            "recent_type_counts": {},
            "activity_rate": 0
        }

# ============================================================================
# CLAUSE FORMATTING AND EXPORT
# ============================================================================

def format_clause_for_display(clause: Dict[str, Any]) -> Dict[str, Any]:
    """Format clause data for display purposes"""
    try:
        metadata = clause.get("clause_metadata", {}) or {}
        
        formatted_clause = {
            "id": clause.get("id"),
            "clause_type": clause.get("clause_type", "general"),
            "clause_text": clause.get("clause_text", ""),
            "clause_purpose": metadata.get("clause_purpose", f"Standard {clause.get('clause_type', 'general')} clause"),
            "position_context": metadata.get("position_context", "General use"),
            "created_at": clause.get("created_at"),
            "file_id": clause.get("file_id"),
            "folder_id": clause.get("folder_id"),
            "relevance_assessment": {
                "when_to_include": metadata.get("when_to_include", ["Standard contracts"]),
                "when_to_exclude": metadata.get("when_to_exclude", []),
                "industry_considerations": metadata.get("industry_considerations", []),
                "risk_implications": metadata.get("risk_implications", ["Standard risk"]),
                "compliance_requirements": metadata.get("compliance_requirements", []),
                "best_practices": metadata.get("best_practices", ["Review with legal counsel"])
            }
        }
        
        return formatted_clause
        
    except Exception as e:
        logger.error(f"Error formatting clause: {e}")
        return clause

def format_clauses_for_export(clauses: List[Dict[str, Any]], export_format: str = "json") -> str:
    """Format clauses for export in various formats"""
    try:
        if export_format.lower() == "json":
            return json.dumps(clauses, indent=2, ensure_ascii=False)
        
        elif export_format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(["ID", "Type", "Text", "Purpose", "Position", "Created"])
            
            # Write clause data
            for clause in clauses:
                metadata = clause.get("clause_metadata", {}) or {}
                writer.writerow([
                    clause.get("id", ""),
                    clause.get("clause_type", ""),
                    clause.get("clause_text", ""),
                    metadata.get("clause_purpose", ""),
                    metadata.get("position_context", ""),
                    clause.get("created_at", "")
                ])
            
            return output.getvalue()
        
        elif export_format.lower() == "markdown":
            output = ["# Clause Library Export\n"]
            
            for clause in clauses:
                metadata = clause.get("clause_metadata", {}) or {}
                output.append(f"## {clause.get('clause_type', 'General').title()} Clause\n")
                output.append(f"**ID:** {clause.get('id', 'N/A')}\n")
                output.append(f"**Purpose:** {metadata.get('clause_purpose', 'N/A')}\n")
                output.append(f"**Position:** {metadata.get('position_context', 'N/A')}\n")
                output.append(f"**Created:** {clause.get('created_at', 'N/A')}\n")
                output.append(f"\n**Clause Text:**\n{clause.get('clause_text', '')}\n")
                output.append("---\n")
            
            return "\n".join(output)
        
        else:
            # Default to JSON
            return json.dumps(clauses, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Error formatting clauses for export: {e}")
        return json.dumps(clauses, indent=2, ensure_ascii=False)

# ============================================================================
# CLAUSE DEDUPLICATION AND CLEANUP
# ============================================================================

async def find_duplicate_clauses(user_id: str, folder_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find duplicate clauses based on text similarity"""
    try:
        db = get_database_service()
        
        query = db.client.from_("clause_library").select("*").eq("user_id", user_id)
        
        if folder_id:
            query = query.eq("folder_id", folder_id)
            
        response = await query.execute()
        
        clauses_data = response.data or []
        
        # Group clauses by hash
        hash_groups = {}
        for clause in clauses_data:
            metadata = clause.get("clause_metadata", {})
            clause_hash = metadata.get("clause_hash")
            
            if clause_hash:
                if clause_hash not in hash_groups:
                    hash_groups[clause_hash] = []
                hash_groups[clause_hash].append(clause)
        
        # Find groups with duplicates
        duplicates = []
        for hash_value, clauses in hash_groups.items():
            if len(clauses) > 1:
                duplicates.append({
                    "hash": hash_value,
                    "count": len(clauses),
                    "clauses": clauses
                })
        
        logger.info(f"Found {len(duplicates)} duplicate clause groups")
        return duplicates
        
    except Exception as e:
        logger.error(f"Error finding duplicate clauses: {e}")
        return []

async def remove_duplicate_clauses(user_id: str, folder_id: Optional[str] = None, keep_latest: bool = True) -> int:
    """Remove duplicate clauses, keeping either the latest or earliest"""
    try:
        duplicate_groups = await find_duplicate_clauses(user_id, folder_id)
        
        if not duplicate_groups:
            return 0
        
        db = get_database_service()
        deleted_count = 0
        
        for group in duplicate_groups:
            clauses = group["clauses"]
            
            # Sort by creation date
            clauses.sort(key=lambda x: x.get("created_at", ""), reverse=keep_latest)
            
            # Keep the first one (latest or earliest based on sort), delete the rest
            clauses_to_delete = clauses[1:]
            
            for clause in clauses_to_delete:
                clause_id = clause.get("id")
                if clause_id:
                    response = await db.client.from_("clause_library").delete().eq("id", clause_id).eq("user_id", user_id).execute()
                    
                    if response.data:
                        deleted_count += 1
        
        logger.info(f"Removed {deleted_count} duplicate clauses")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error removing duplicate clauses: {e}")
        return 0

async def cleanup_orphaned_clauses(user_id: str) -> int:
    """Remove clauses that reference non-existent files or folders"""
    try:
        db = get_database_service()
        
        # Get all clauses for the user
        clauses_response = await db.client.from_("clause_library").select("id, file_id, folder_id").eq("user_id", user_id).execute()
        
        if not clauses_response.data:
            return 0
        
        # Get all valid file and folder IDs
        files_response = await db.client.from_("files").select("id").eq("user_id", user_id).execute()
        folders_response = await db.client.from_("folders").select("id").eq("user_id", user_id).execute()
        
        valid_file_ids = set(file["id"] for file in files_response.data) if files_response.data else set()
        valid_folder_ids = set(folder["id"] for folder in folders_response.data) if folders_response.data else set()
        
        # Find orphaned clauses
        orphaned_clauses = []
        for clause in clauses_response.data:
            file_id = clause.get("file_id")
            folder_id = clause.get("folder_id")
            
            if (file_id and file_id not in valid_file_ids) or (folder_id and folder_id not in valid_folder_ids):
                orphaned_clauses.append(clause["id"])
        
        # Delete orphaned clauses
        if orphaned_clauses:
            await db.client.from_("clause_library").delete().in_("id", orphaned_clauses).eq("user_id", user_id).execute()
            logger.info(f"Cleaned up {len(orphaned_clauses)} orphaned clauses")
        
        return len(orphaned_clauses)
        
    except Exception as e:
        logger.error(f"Error cleaning up orphaned clauses: {e}")
        return 0

# ============================================================================
# CLAUSE SEARCH AND FILTERING
# ============================================================================

async def search_clauses(
    user_id: str,
    search_term: str,
    clause_type: Optional[str] = None,
    folder_id: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Search clauses by text content"""
    try:
        db = get_database_service()
        
        query = db.client.from_("clause_library").select("*").eq("user_id", user_id)
        
        if clause_type:
            query = query.eq("clause_type", clause_type)
        
        if folder_id:
            query = query.eq("folder_id", folder_id)
            
        response = await query.limit(limit).execute()
        
        clauses_data = response.data or []
        
        # Filter by search term (case-insensitive)
        search_term_lower = search_term.lower()
        matching_clauses = []
        
        for clause in clauses_data:
            clause_text = clause.get("clause_text", "").lower()
            clause_type_text = clause.get("clause_type", "").lower()
            
            if search_term_lower in clause_text or search_term_lower in clause_type_text:
                matching_clauses.append(clause)
        
        logger.debug(f"Found {len(matching_clauses)} clauses matching '{search_term}'")
        return matching_clauses
        
    except Exception as e:
        logger.error(f"Error searching clauses: {e}")
        return []

async def get_clause_types(user_id: str, folder_id: Optional[str] = None) -> List[str]:
    """Get all unique clause types for a user"""
    try:
        db = get_database_service()
        
        query = db.client.from_("clause_library").select("clause_type").eq("user_id", user_id)
        
        if folder_id:
            query = query.eq("folder_id", folder_id)
            
        response = await query.execute()
        
        clause_types = set()
        for clause in response.data or []:
            clause_type = clause.get("clause_type")
            if clause_type:
                clause_types.add(clause_type)
        
        return sorted(list(clause_types))
        
    except Exception as e:
        logger.error(f"Error getting clause types: {e}")
        return []
