"""
Usage tracking operation utilities for the Legal Template Generator.
Centralizes all usage analytics and tracking operations.
"""

from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime, timedelta
from loguru import logger
from core.database import get_database_service

logger.add("logs/usage_utilities.log", rotation="10 MB", level="DEBUG")

# ============================================================================
# TEMPLATE USAGE TRACKING
# ============================================================================

async def track_template_usage(
    template_id: str,
    user_id: str,
    action_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Track template usage for analytics"""
    try:
        db = get_database_service()
        
        usage_data = {
            "id": str(uuid4()),
            "template_id": template_id,
            "user_id": user_id,
            "action_type": action_type,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("template_usage_stats").insert(usage_data).execute()
        
        if response.data:
            logger.debug(f"Tracked template usage: {action_type} for template {template_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error tracking template usage: {e}")
        return False

async def track_file_usage(
    file_id: str,
    user_id: str,
    action_type: str,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Track file usage for analytics"""
    try:
        db = get_database_service()
        
        usage_data = {
            "id": str(uuid4()),
            "file_id": file_id,
            "user_id": user_id,
            "action_type": action_type,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("file_usage_stats").insert(usage_data).execute()
        
        if response.data:
            logger.debug(f"Tracked file usage: {action_type} for file {file_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error tracking file usage: {e}")
        return False

async def track_user_action(
    user_id: str,
    action_type: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Track general user actions for analytics"""
    try:
        db = get_database_service()
        
        action_data = {
            "id": str(uuid4()),
            "user_id": user_id,
            "action_type": action_type,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("user_actions").insert(action_data).execute()
        
        if response.data:
            logger.debug(f"Tracked user action: {action_type} on {resource_type}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error tracking user action: {e}")
        return False

# ============================================================================
# USAGE ANALYTICS AND REPORTING
# ============================================================================

async def get_template_usage_stats(
    template_id: str,
    user_id: Optional[str] = None,
    days: int = 30
) -> Dict[str, Any]:
    """Get usage statistics for a specific template"""
    try:
        db = get_database_service()
        
        # Calculate time threshold
        threshold_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        query = db.client.from_("template_usage_stats").select("*").eq("template_id", template_id).gte("created_at", threshold_date)
        
        if user_id:
            query = query.eq("user_id", user_id)
            
        response = await query.execute()
        
        usage_data = response.data or []
        
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
            date_str = usage.get("created_at", "")[:10]  # Extract date part
            daily_usage[date_str] = daily_usage.get(date_str, 0) + 1
        
        # Unique users
        unique_users = set()
        for usage in usage_data:
            user = usage.get("user_id")
            if user:
                unique_users.add(user)
        
        return {
            "template_id": template_id,
            "period_days": days,
            "total_actions": total_actions,
            "action_counts": action_counts,
            "daily_usage": daily_usage,
            "unique_users": len(unique_users),
            "average_daily_usage": round(total_actions / days, 2) if days > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting template usage stats for {template_id}: {e}")
        return {
            "template_id": template_id,
            "period_days": days,
            "total_actions": 0,
            "action_counts": {},
            "daily_usage": {},
            "unique_users": 0,
            "average_daily_usage": 0
        }

async def get_user_activity_summary(user_id: str, days: int = 30) -> Dict[str, Any]:
    """Get comprehensive user activity summary"""
    try:
        db = get_database_service()
        
        # Calculate time threshold
        threshold_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Get user actions
        actions_response = await db.client.from_("user_actions").select("*").eq("user_id", user_id).gte("created_at", threshold_date).execute()
        
        # Get template usage
        template_usage_response = await db.client.from_("template_usage_stats").select("*").eq("user_id", user_id).gte("created_at", threshold_date).execute()
        
        # Get file usage
        file_usage_response = await db.client.from_("file_usage_stats").select("*").eq("user_id", user_id).gte("created_at", threshold_date).execute()
        
        actions_data = actions_response.data or []
        template_usage_data = template_usage_response.data or []
        file_usage_data = file_usage_response.data or []
        
        # Calculate statistics
        total_actions = len(actions_data)
        total_template_actions = len(template_usage_data)
        total_file_actions = len(file_usage_data)
        
        # Action type distributions
        action_types = {}
        for action in actions_data:
            action_type = action.get("action_type", "unknown")
            action_types[action_type] = action_types.get(action_type, 0) + 1
        
        template_action_types = {}
        for usage in template_usage_data:
            action_type = usage.get("action_type", "unknown")
            template_action_types[action_type] = template_action_types.get(action_type, 0) + 1
        
        file_action_types = {}
        for usage in file_usage_data:
            action_type = usage.get("action_type", "unknown")
            file_action_types[action_type] = file_action_types.get(action_type, 0) + 1
        
        # Daily activity
        daily_activity = {}
        all_activities = actions_data + template_usage_data + file_usage_data
        
        for activity in all_activities:
            date_str = activity.get("created_at", "")[:10]
            daily_activity[date_str] = daily_activity.get(date_str, 0) + 1
        
        # Most active days
        sorted_days = sorted(daily_activity.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_activity": total_actions + total_template_actions + total_file_actions,
            "general_actions": total_actions,
            "template_actions": total_template_actions,
            "file_actions": total_file_actions,
            "action_type_distribution": action_types,
            "template_action_distribution": template_action_types,
            "file_action_distribution": file_action_types,
            "daily_activity": daily_activity,
            "most_active_days": sorted_days[:7],  # Top 7 days
            "average_daily_activity": round((total_actions + total_template_actions + total_file_actions) / days, 2) if days > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting user activity summary for {user_id}: {e}")
        return {
            "user_id": user_id,
            "period_days": days,
            "total_activity": 0,
            "general_actions": 0,
            "template_actions": 0,
            "file_actions": 0,
            "action_type_distribution": {},
            "template_action_distribution": {},
            "file_action_distribution": {},
            "daily_activity": {},
            "most_active_days": [],
            "average_daily_activity": 0
        }

async def get_system_usage_stats(days: int = 30) -> Dict[str, Any]:
    """Get system-wide usage statistics"""
    try:
        db = get_database_service()
        
        # Calculate time threshold
        threshold_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Get all user actions
        actions_response = await db.client.from_("user_actions").select("user_id, action_type, created_at").gte("created_at", threshold_date).execute()
        
        # Get template usage
        template_response = await db.client.from_("template_usage_stats").select("user_id, action_type, template_id, created_at").gte("created_at", threshold_date).execute()
        
        # Get file usage
        file_response = await db.client.from_("file_usage_stats").select("user_id, action_type, file_id, created_at").gte("created_at", threshold_date).execute()
        
        actions_data = actions_response.data or []
        template_data = template_response.data or []
        file_data = file_response.data or []
        
        # Calculate system statistics
        total_actions = len(actions_data) + len(template_data) + len(file_data)
        
        # Unique users
        unique_users = set()
        for data in actions_data + template_data + file_data:
            user_id = data.get("user_id")
            if user_id:
                unique_users.add(user_id)
        
        # Action type distribution
        action_distribution = {}
        for data in actions_data + template_data + file_data:
            action_type = data.get("action_type", "unknown")
            action_distribution[action_type] = action_distribution.get(action_type, 0) + 1
        
        # Daily system activity
        daily_activity = {}
        for data in actions_data + template_data + file_data:
            date_str = data.get("created_at", "")[:10]
            daily_activity[date_str] = daily_activity.get(date_str, 0) + 1
        
        # Most used templates
        template_usage = {}
        for usage in template_data:
            template_id = usage.get("template_id")
            if template_id:
                template_usage[template_id] = template_usage.get(template_id, 0) + 1
        
        most_used_templates = sorted(template_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "period_days": days,
            "total_actions": total_actions,
            "unique_active_users": len(unique_users),
            "action_distribution": action_distribution,
            "daily_activity": daily_activity,
            "most_used_templates": most_used_templates,
            "average_daily_actions": round(total_actions / days, 2) if days > 0 else 0,
            "average_actions_per_user": round(total_actions / len(unique_users), 2) if unique_users else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting system usage stats: {e}")
        return {
            "period_days": days,
            "total_actions": 0,
            "unique_active_users": 0,
            "action_distribution": {},
            "daily_activity": {},
            "most_used_templates": [],
            "average_daily_actions": 0,
            "average_actions_per_user": 0
        }

# ============================================================================
# PROCESSING STATISTICS
# ============================================================================

async def get_processing_stats(user_id: str) -> Dict[str, Any]:
    """Get comprehensive processing statistics for a user"""
    try:
        db = get_database_service()
        
        # Get file counts by status
        files_response = await db.client.from_("files").select("status, file_size, created_at").eq("user_id", user_id).execute()
        
        files_data = files_response.data or []
        
        # Calculate file statistics
        total_files = len(files_data)
        
        status_counts = {}
        total_size = 0
        
        for file_data in files_data:
            status = file_data.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            total_size += file_data.get("file_size", 0)
        
        # Get markdown stats
        markdown_response = await db.client.from_("markdown_content").select("word_count, created_at").eq("user_id", user_id).execute()
        
        markdown_data = markdown_response.data or []
        total_words = sum(item.get("word_count", 0) for item in markdown_data)
        total_documents_processed = len(markdown_data)
        
        # Get clause counts
        clause_response = await db.client.from_("clause_library").select("id, created_at", count="exact").eq("user_id", user_id).execute()
        total_clauses = clause_response.count or 0
        
        # Get template counts
        template_response = await db.client.from_("templates").select("id, created_at").eq("user_id", user_id).execute()
        template_data = template_response.data or []
        total_templates = len(template_data)
        
        # Get job statistics
        jobs_response = await db.client.from_("jobs").select("status, job_type, created_at").eq("user_id", user_id).execute()
        
        jobs_data = jobs_response.data or []
        
        job_status_counts = {}
        job_type_counts = {}
        
        for job in jobs_data:
            status = job.get("status", "unknown")
            job_type = job.get("job_type", "unknown")
            
            job_status_counts[status] = job_status_counts.get(status, 0) + 1
            job_type_counts[job_type] = job_type_counts.get(job_type, 0) + 1
        
        return {
            "user_id": user_id,
            "files": {
                "total_files": total_files,
                "status_counts": status_counts,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / 1024 / 1024, 2),
                "average_file_size": round(total_size / total_files, 2) if total_files > 0 else 0
            },
            "content": {
                "total_documents_processed": total_documents_processed,
                "total_words_processed": total_words,
                "average_words_per_document": round(total_words / total_documents_processed, 2) if total_documents_processed > 0 else 0
            },
            "clauses": {
                "total_clauses": total_clauses
            },
            "templates": {
                "total_templates": total_templates
            },
            "jobs": {
                "total_jobs": len(jobs_data),
                "job_status_counts": job_status_counts,
                "job_type_counts": job_type_counts
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting processing stats for user {user_id}: {e}")
        return {
            "user_id": user_id,
            "files": {"total_files": 0, "status_counts": {}, "total_size_bytes": 0, "total_size_mb": 0, "average_file_size": 0},
            "content": {"total_documents_processed": 0, "total_words_processed": 0, "average_words_per_document": 0},
            "clauses": {"total_clauses": 0},
            "templates": {"total_templates": 0},
            "jobs": {"total_jobs": 0, "job_status_counts": {}, "job_type_counts": {}}
        }

# ============================================================================
# USAGE TRENDS AND INSIGHTS
# ============================================================================

async def get_usage_trends(user_id: str, days: int = 30) -> Dict[str, Any]:
    """Get usage trends over time for a user"""
    try:
        db = get_database_service()
        
        # Calculate time threshold
        threshold_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Get files created over time
        files_response = await db.client.from_("files").select("created_at").eq("user_id", user_id).gte("created_at", threshold_date).execute()
        
        # Get templates created over time
        templates_response = await db.client.from_("templates").select("created_at").eq("user_id", user_id).gte("created_at", threshold_date).execute()
        
        # Get clauses extracted over time
        clauses_response = await db.client.from_("clause_library").select("created_at").eq("user_id", user_id).gte("created_at", threshold_date).execute()
        
        files_data = files_response.data or []
        templates_data = templates_response.data or []
        clauses_data = clauses_response.data or []
        
        # Create daily trends
        daily_files = {}
        daily_templates = {}
        daily_clauses = {}
        
        for file_data in files_data:
            date_str = file_data.get("created_at", "")[:10]
            daily_files[date_str] = daily_files.get(date_str, 0) + 1
        
        for template_data in templates_data:
            date_str = template_data.get("created_at", "")[:10]
            daily_templates[date_str] = daily_templates.get(date_str, 0) + 1
        
        for clause_data in clauses_data:
            date_str = clause_data.get("created_at", "")[:10]
            daily_clauses[date_str] = daily_clauses.get(date_str, 0) + 1
        
        # Calculate growth rates (simple comparison between first and second half of period)
        mid_point = days // 2
        mid_date = (datetime.utcnow() - timedelta(days=mid_point)).isoformat()[:10]
        
        recent_files = sum(count for date, count in daily_files.items() if date >= mid_date)
        early_files = sum(count for date, count in daily_files.items() if date < mid_date)
        
        recent_templates = sum(count for date, count in daily_templates.items() if date >= mid_date)
        early_templates = sum(count for date, count in daily_templates.items() if date < mid_date)
        
        recent_clauses = sum(count for date, count in daily_clauses.items() if date >= mid_date)
        early_clauses = sum(count for date, count in daily_clauses.items() if date < mid_date)
        
        file_growth_rate = ((recent_files - early_files) / early_files * 100) if early_files > 0 else 0
        template_growth_rate = ((recent_templates - early_templates) / early_templates * 100) if early_templates > 0 else 0
        clause_growth_rate = ((recent_clauses - early_clauses) / early_clauses * 100) if early_clauses > 0 else 0
        
        return {
            "user_id": user_id,
            "period_days": days,
            "daily_trends": {
                "files": daily_files,
                "templates": daily_templates,
                "clauses": daily_clauses
            },
            "totals": {
                "files": len(files_data),
                "templates": len(templates_data),
                "clauses": len(clauses_data)
            },
            "growth_rates": {
                "files_percent": round(file_growth_rate, 2),
                "templates_percent": round(template_growth_rate, 2),
                "clauses_percent": round(clause_growth_rate, 2)
            },
            "averages": {
                "files_per_day": round(len(files_data) / days, 2) if days > 0 else 0,
                "templates_per_day": round(len(templates_data) / days, 2) if days > 0 else 0,
                "clauses_per_day": round(len(clauses_data) / days, 2) if days > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting usage trends for user {user_id}: {e}")
        return {
            "user_id": user_id,
            "period_days": days,
            "daily_trends": {"files": {}, "templates": {}, "clauses": {}},
            "totals": {"files": 0, "templates": 0, "clauses": 0},
            "growth_rates": {"files_percent": 0, "templates_percent": 0, "clauses_percent": 0},
            "averages": {"files_per_day": 0, "templates_per_day": 0, "clauses_per_day": 0}
        }

# ============================================================================
# USAGE DATA CLEANUP
# ============================================================================

async def cleanup_old_usage_data(days: int = 90) -> int:
    """Clean up old usage tracking data"""
    try:
        db = get_database_service()
        
        # Calculate cutoff date
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Clean up old records from each table
        tables_to_clean = [
            "template_usage_stats",
            "file_usage_stats", 
            "user_actions"
        ]
        
        total_deleted = 0
        
        for table in tables_to_clean:
            try:
                response = await db.client.from_(table).delete().lt("created_at", cutoff_date).execute()
                deleted_count = len(response.data) if response.data else 0
                total_deleted += deleted_count
                logger.debug(f"Cleaned up {deleted_count} records from {table}")
            except Exception as table_error:
                logger.warning(f"Failed to clean up {table}: {table_error}")
        
        logger.info(f"Cleaned up {total_deleted} old usage records (older than {days} days)")
        return total_deleted
        
    except Exception as e:
        logger.error(f"Error cleaning up old usage data: {e}")
        return 0

async def get_popular_templates(limit: int = 10, days: int = 30) -> List[Dict[str, Any]]:
    """Get most popular templates based on usage"""
    try:
        db = get_database_service()
        
        # Calculate time threshold
        threshold_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Get template usage stats
        response = await db.client.from_("template_usage_stats").select("template_id, action_type").gte("created_at", threshold_date).execute()
        
        usage_data = response.data or []
        
        # Count usage per template
        template_usage = {}
        for usage in usage_data:
            template_id = usage.get("template_id")
            if template_id:
                template_usage[template_id] = template_usage.get(template_id, 0) + 1
        
        # Sort by usage count
        sorted_templates = sorted(template_usage.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        # Get template details
        popular_templates = []
        for template_id, usage_count in sorted_templates:
            # Get template info
            template_response = await db.client.from_("templates").select("name, created_at, folders(name)").eq("id", template_id).single().execute()
            
            if template_response.data:
                template_info = template_response.data
                folder_info = template_info.get("folders", {})
                
                popular_templates.append({
                    "template_id": template_id,
                    "template_name": template_info.get("name", "Unknown"),
                    "folder_name": folder_info.get("name", "Unknown"),
                    "usage_count": usage_count,
                    "created_at": template_info.get("created_at")
                })
        
        return popular_templates
        
    except Exception as e:
        logger.error(f"Error getting popular templates: {e}")
        return []
