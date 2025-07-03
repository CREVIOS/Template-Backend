"""
Template operation utilities for the Legal Template Generator.
Centralizes all template-related database operations and management.
"""

import json
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime, timedelta
from loguru import logger
from core.database import get_database_service

logger.add("logs/template_utilities.log", rotation="10 MB", level="DEBUG")

# ============================================================================
# TEMPLATE CRUD OPERATIONS
# ============================================================================

async def create_template(
    folder_id: str,
    name: str,
    content: str,
    template_type: str = "general",
    file_extension: str = ".docx",
    formatting_data: Optional[Dict[str, Any]] = None,
    content_json: Optional[Dict[str, Any]] = None,
    word_compatible: bool = True,
    is_active: bool = True,
    template_id: Optional[str] = None
) -> Optional[str]:
    """Create a new template"""
    try:
        db = get_database_service()
        
        template_data = {
            "id": template_id or str(uuid4()),
            "folder_id": folder_id,
            "name": name,
            "content": content,
            "template_type": template_type,
            "file_extension": file_extension,
            "formatting_data": formatting_data or {},
            "content_json": content_json,
            "word_compatible": word_compatible,
            "is_active": is_active,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("templates").insert(template_data).execute()
        
        if response.data:
            logger.info(f"Created template {template_data['id']}: {name}")
            return template_data["id"]
        return None
        
    except Exception as e:
        logger.error(f"Error creating template '{name}': {e}")
        return None

async def get_template_by_id(template_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Get template by ID with folder info and user access validation"""
    try:
        db = get_database_service()
        
        response = await db.client.from_("templates").select(
            "*, folders(name, color, user_id)"
        ).eq("id", template_id).single().execute()
        
        if not response.data:
            return None
            
        template_data = response.data
        folder = template_data.get("folders", {})
        
        # Check user access through folder
        if folder.get("user_id") != user_id:
            logger.warning(f"Access denied to template {template_id} for user {user_id}")
            return None
            
        return template_data
        
    except Exception as e:
        logger.error(f"Error getting template {template_id}: {e}")
        return None

async def get_templates_by_user(
    user_id: str,
    folder_id: Optional[str] = None,
    template_type: Optional[str] = None,
    is_active: Optional[bool] = None,
    include_stats: bool = True
) -> List[Dict[str, Any]]:
    """Get all templates for a user with optional filters"""
    try:
        db = get_database_service()
        
        # Build query with joins
        select_fields = "*, folders(name, color)"
        if include_stats:
            select_fields += ", template_usage_stats(action_type, created_at)"
        
        query = db.client.from_("templates").select(select_fields).eq("folders.user_id", user_id)
        
        # Apply filters
        if folder_id:
            query = query.eq("folder_id", folder_id)
        if template_type:
            query = query.eq("template_type", template_type)
        if is_active is not None:
            query = query.eq("is_active", is_active)
            
        response = await query.order("created_at", desc=True).execute()
        
        templates = response.data or []
        
        # Enrich templates with statistics if requested
        if include_stats:
            for template in templates:
                usage_stats = template.get("template_usage_stats", [])
                
                # Get most recent action
                last_action = None
                last_action_date = None
                if usage_stats:
                    sorted_stats = sorted(usage_stats, key=lambda x: x.get("created_at", ""), reverse=True)
                    if sorted_stats:
                        last_action = sorted_stats[0].get("action_type")
                        last_action_date = sorted_stats[0].get("created_at")
                
                template["last_action_type"] = last_action
                template["last_action_date"] = last_action_date
        
        return templates
        
    except Exception as e:
        logger.error(f"Error getting templates for user {user_id}: {e}")
        return []

async def update_template(
    template_id: str,
    user_id: str,
    name: Optional[str] = None,
    content: Optional[str] = None,
    template_type: Optional[str] = None,
    file_extension: Optional[str] = None,
    formatting_data: Optional[Dict[str, Any]] = None,
    content_json: Optional[Dict[str, Any]] = None,
    word_compatible: Optional[bool] = None,
    is_active: Optional[bool] = None
) -> bool:
    """Update template data"""
    try:
        db = get_database_service()
        
        # Verify user has access to template
        template = await get_template_by_id(template_id, user_id)
        if not template:
            logger.warning(f"Template {template_id} not found or access denied for user {user_id}")
            return False
        
        # Build update data
        update_data = {"updated_at": datetime.utcnow().isoformat()}
        
        if name is not None:
            update_data["name"] = name
        if content is not None:
            update_data["content"] = content
        if template_type is not None:
            update_data["template_type"] = template_type
        if file_extension is not None:
            update_data["file_extension"] = file_extension
        if formatting_data is not None:
            update_data["formatting_data"] = formatting_data
        if content_json is not None:
            update_data["content_json"] = content_json
        if word_compatible is not None:
            update_data["word_compatible"] = word_compatible
        if is_active is not None:
            update_data["is_active"] = is_active
            
        response = await db.client.from_("templates").update(update_data).eq("id", template_id).execute()
        
        if response.data:
            logger.info(f"Updated template {template_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error updating template {template_id}: {e}")
        return False

async def delete_template(template_id: str, user_id: str) -> bool:
    """Delete template with access validation"""
    try:
        db = get_database_service()
        
        # Verify user has access to template
        template = await get_template_by_id(template_id, user_id)
        if not template:
            logger.warning(f"Template {template_id} not found or access denied for user {user_id}")
            return False
        
        # Delete template (cascades to related tables)
        response = await db.client.from_("templates").delete().eq("id", template_id).execute()
        
        if response.data:
            logger.info(f"Deleted template {template_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error deleting template {template_id}: {e}")
        return False

# ============================================================================
# TEMPLATE SEARCH AND FILTERING
# ============================================================================

async def search_templates(
    user_id: str,
    search_term: str,
    template_type: Optional[str] = None,
    folder_id: Optional[str] = None,
    is_active: Optional[bool] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Search templates by name or content"""
    try:
        db = get_database_service()
        
        # Build query
        query = db.client.from_("templates").select(
            "*, folders(name, color)"
        ).eq("folders.user_id", user_id)
        
        # Apply search
        query = query.or_(f"name.ilike.%{search_term}%,content.ilike.%{search_term}%")
        
        # Apply filters
        if template_type:
            query = query.eq("template_type", template_type)
        if folder_id:
            query = query.eq("folder_id", folder_id)
        if is_active is not None:
            query = query.eq("is_active", is_active)
            
        response = await query.order("updated_at", desc=True).limit(limit).execute()
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error searching templates for '{search_term}': {e}")
        return []

async def get_templates_by_type(
    user_id: str,
    template_type: str,
    folder_id: Optional[str] = None,
    is_active: bool = True
) -> List[Dict[str, Any]]:
    """Get templates by type"""
    try:
        db = get_database_service()
        
        query = db.client.from_("templates").select(
            "*, folders(name, color)"
        ).eq("folders.user_id", user_id).eq("template_type", template_type)
        
        if folder_id:
            query = query.eq("folder_id", folder_id)
        if is_active is not None:
            query = query.eq("is_active", is_active)
            
        response = await query.order("name").execute()
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting templates by type '{template_type}': {e}")
        return []

async def get_template_types(user_id: str, folder_id: Optional[str] = None) -> List[str]:
    """Get distinct template types for a user"""
    try:
        db = get_database_service()
        
        query = db.client.from_("templates").select("template_type").eq("folders.user_id", user_id)
        
        if folder_id:
            query = query.eq("folder_id", folder_id)
            
        response = await query.execute()
        
        # Extract unique types
        types = set()
        for template in response.data or []:
            template_type = template.get("template_type")
            if template_type:
                types.add(template_type)
        
        return sorted(list(types))
        
    except Exception as e:
        logger.error(f"Error getting template types: {e}")
        return []

# ============================================================================
# TEMPLATE VALIDATION AND UTILITIES
# ============================================================================

async def validate_template_access(template_id: str, user_id: str) -> bool:
    """Validate user has access to template"""
    template = await get_template_by_id(template_id, user_id)
    return template is not None

async def check_template_name_exists(folder_id: str, name: str, exclude_id: Optional[str] = None) -> bool:
    """Check if template name exists in folder"""
    try:
        db = get_database_service()
        
        query = db.client.from_("templates").select("id").eq("folder_id", folder_id).eq("name", name)
        
        if exclude_id:
            query = query.neq("id", exclude_id)
            
        response = await query.execute()
        
        return bool(response.data)
        
    except Exception as e:
        logger.error(f"Error checking template name existence: {e}")
        return False

async def duplicate_template(
    template_id: str,
    user_id: str,
    new_name: str,
    target_folder_id: Optional[str] = None
) -> Optional[str]:
    """Duplicate an existing template"""
    try:
        db = get_database_service()
        
        # Get original template
        original_template = await get_template_by_id(template_id, user_id)
        if not original_template:
            logger.warning(f"Template {template_id} not found or access denied for user {user_id}")
            return None
        
        # Use target folder or keep original folder
        folder_id = target_folder_id or original_template["folder_id"]
        
        # Check if name already exists
        if await check_template_name_exists(folder_id, new_name):
            logger.warning(f"Template name '{new_name}' already exists in folder {folder_id}")
            return None
        
        # Create duplicate
        new_template_id = await create_template(
            folder_id=folder_id,
            name=new_name,
            content=original_template.get("content", ""),
            template_type=original_template.get("template_type", "general"),
            file_extension=original_template.get("file_extension", ".docx"),
            formatting_data=original_template.get("formatting_data", {}),
            content_json=original_template.get("content_json"),
            word_compatible=original_template.get("word_compatible", True),
            is_active=original_template.get("is_active", True)
        )
        
        if new_template_id:
            logger.info(f"Duplicated template {template_id} to {new_template_id}")
            return new_template_id
        return None
        
    except Exception as e:
        logger.error(f"Error duplicating template {template_id}: {e}")
        return None

# ============================================================================
# TEMPLATE STATISTICS AND ANALYTICS
# ============================================================================

async def get_template_statistics(template_id: str, user_id: str, days: int = 30) -> Dict[str, Any]:
    """Get comprehensive statistics for a template"""
    try:
        db = get_database_service()
        
        # Verify access
        template = await get_template_by_id(template_id, user_id)
        if not template:
            return {}
        
        # Calculate time threshold
        threshold_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Get usage statistics
        usage_response = await db.client.from_("template_usage_stats").select("*").eq("template_id", template_id).gte("created_at", threshold_date).execute()
        
        usage_data = usage_response.data or []
        
        # Calculate statistics
        total_actions = len(usage_data)
        
        # Action type counts
        action_counts = {}
        for usage in usage_data:
            action_type = usage.get("action_type", "unknown")
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        # Daily usage
        daily_usage = {}
        for usage in usage_data:
            date_str = usage.get("created_at", "")[:10]
            daily_usage[date_str] = daily_usage.get(date_str, 0) + 1
        
        # Unique users
        unique_users = set()
        for usage in usage_data:
            user = usage.get("user_id")
            if user:
                unique_users.add(user)
        
        # Get related clauses count
        clauses_response = await db.client.from_("clause_library").select("id", count="exact").eq("template_id", template_id).execute()
        clauses_count = clauses_response.count or 0
        
        return {
            "template_id": template_id,
            "template_name": template.get("name", "Unknown"),
            "period_days": days,
            "total_actions": total_actions,
            "action_counts": action_counts,
            "daily_usage": daily_usage,
            "unique_users": len(unique_users),
            "clauses_count": clauses_count,
            "average_daily_usage": round(total_actions / days, 2) if days > 0 else 0,
            "created_at": template.get("created_at"),
            "updated_at": template.get("updated_at")
        }
        
    except Exception as e:
        logger.error(f"Error getting template statistics for {template_id}: {e}")
        return {}

async def get_user_template_stats(user_id: str) -> Dict[str, Any]:
    """Get template statistics for a user"""
    try:
        db = get_database_service()
        
        # Get all templates for user
        templates = await get_templates_by_user(user_id, include_stats=False)
        
        # Calculate statistics
        total_templates = len(templates)
        
        # Count by type
        type_counts = {}
        active_count = 0
        inactive_count = 0
        
        for template in templates:
            template_type = template.get("template_type", "general")
            type_counts[template_type] = type_counts.get(template_type, 0) + 1
            
            if template.get("is_active", True):
                active_count += 1
            else:
                inactive_count += 1
        
        # Get usage statistics
        usage_response = await db.client.from_("template_usage_stats").select("action_type, created_at").eq("user_id", user_id).execute()
        
        usage_data = usage_response.data or []
        total_actions = len(usage_data)
        
        # Recent activity (last 7 days)
        recent_threshold = (datetime.utcnow() - timedelta(days=7)).isoformat()
        recent_activity = len([u for u in usage_data if u.get("created_at", "") >= recent_threshold])
        
        return {
            "user_id": user_id,
            "total_templates": total_templates,
            "active_templates": active_count,
            "inactive_templates": inactive_count,
            "type_counts": type_counts,
            "total_actions": total_actions,
            "recent_activity": recent_activity,
            "most_common_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        }
        
    except Exception as e:
        logger.error(f"Error getting user template stats for {user_id}: {e}")
        return {}

# ============================================================================
# TEMPLATE CONTENT MANAGEMENT
# ============================================================================

async def extract_template_clauses(template_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Extract clauses from template content"""
    try:
        db = get_database_service()
        
        # Get template
        template = await get_template_by_id(template_id, user_id)
        if not template:
            return []
        
        # Check if content_json exists and contains clauses
        content_json = template.get("content_json")
        if content_json and isinstance(content_json, dict):
            clauses = content_json.get("clauses", [])
            if clauses:
                return clauses
        
        # If no structured clauses, return empty list
        # TODO: Implement AI-based clause extraction from content text
        return []
        
    except Exception as e:
        logger.error(f"Error extracting clauses from template {template_id}: {e}")
        return []

async def update_template_content_json(
    template_id: str,
    user_id: str,
    content_json: Dict[str, Any]
) -> bool:
    """Update template content_json field"""
    try:
        db = get_database_service()
        
        # Verify access
        template = await get_template_by_id(template_id, user_id)
        if not template:
            return False
        
        # Update content_json
        update_data = {
            "content_json": content_json,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("templates").update(update_data).eq("id", template_id).execute()
        
        if response.data:
            logger.info(f"Updated content_json for template {template_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error updating content_json for template {template_id}: {e}")
        return False

# ============================================================================
# TEMPLATE MAINTENANCE AND CLEANUP
# ============================================================================

async def cleanup_inactive_templates(user_id: str, days: int = 90) -> int:
    """Clean up templates that haven't been used in specified days"""
    try:
        db = get_database_service()
        
        # Calculate threshold date
        threshold_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Get templates with no recent usage
        templates_response = await db.client.from_("templates").select(
            "id, name, updated_at"
        ).eq("folders.user_id", user_id).eq("is_active", True).lt("updated_at", threshold_date).execute()
        
        if not templates_response.data:
            return 0
        
        cleanup_count = 0
        for template in templates_response.data:
            template_id = template["id"]
            
            # Check if template has recent usage
            usage_response = await db.client.from_("template_usage_stats").select("id").eq("template_id", template_id).gte("created_at", threshold_date).execute()
            
            # If no recent usage, mark as inactive
            if not usage_response.data:
                await db.client.from_("templates").update({
                    "is_active": False,
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", template_id).execute()
                
                cleanup_count += 1
                logger.info(f"Marked template {template_id} as inactive due to no recent usage")
        
        logger.info(f"Cleaned up {cleanup_count} inactive templates for user {user_id}")
        return cleanup_count
        
    except Exception as e:
        logger.error(f"Error cleaning up inactive templates for user {user_id}: {e}")
        return 0

async def get_templates_needing_update(user_id: str) -> List[Dict[str, Any]]:
    """Get templates that might need updates (old format, missing data, etc.)"""
    try:
        db = get_database_service()
        
        # Get templates with potential issues
        templates_response = await db.client.from_("templates").select(
            "*, folders(name)"
        ).eq("folders.user_id", user_id).execute()
        
        templates_needing_update = []
        
        for template in templates_response.data or []:
            issues = []
            
            # Check for missing content_json
            if not template.get("content_json"):
                issues.append("missing_content_json")
            
            # Check for empty formatting_data
            if not template.get("formatting_data"):
                issues.append("empty_formatting_data")
            
            # Check for very old templates
            created_at = template.get("created_at")
            if created_at:
                created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if created_date < datetime.utcnow() - timedelta(days=365):
                    issues.append("old_template")
            
            if issues:
                template["issues"] = issues
                templates_needing_update.append(template)
        
        return templates_needing_update
        
    except Exception as e:
        logger.error(f"Error getting templates needing update for user {user_id}: {e}")
        return []

async def bulk_update_template_status(
    template_ids: List[str],
    user_id: str,
    is_active: bool
) -> int:
    """Bulk update template active status"""
    try:
        db = get_database_service()
        
        # Verify user has access to all templates
        accessible_templates = []
        for template_id in template_ids:
            if await validate_template_access(template_id, user_id):
                accessible_templates.append(template_id)
        
        if not accessible_templates:
            return 0
        
        # Update templates
        update_data = {
            "is_active": is_active,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("templates").update(update_data).in_("id", accessible_templates).execute()
        
        updated_count = len(response.data) if response.data else 0
        logger.info(f"Bulk updated {updated_count} templates status to {is_active}")
        return updated_count
        
    except Exception as e:
        logger.error(f"Error bulk updating template status: {e}")
        return 0
