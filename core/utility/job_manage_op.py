"""
Job management operation utilities for the Legal Template Generator.
Centralizes all job-related database operations and background task management.
"""

from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime, timedelta
from loguru import logger
from core.database import get_database_service

logger.add("logs/job_utilities.log", rotation="10 MB", level="DEBUG")

# ============================================================================
# JOB CREATION AND MANAGEMENT
# ============================================================================

async def create_job(
    user_id: str,
    job_type: str,
    metadata: Optional[Dict[str, Any]] = None,
    total_steps: int = 1,
    job_id: Optional[str] = None
) -> Optional[str]:
    """Create a new job record"""
    try:
        db = get_database_service()
        
        job_data = {
            "id": job_id or str(uuid4()),
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

async def get_job_by_id(job_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get job by ID with optional user validation"""
    try:
        db = get_database_service()
        
        query = db.client.from_("jobs").select("*").eq("id", job_id)
        
        if user_id:
            query = query.eq("user_id", user_id)
            
        response = await query.single().execute()
        
        return response.data if response.data else None
        
    except Exception as e:
        logger.error(f"Error getting job {job_id}: {e}")
        return None

async def update_job_status(job_id: str, status: str, error_message: Optional[str] = None) -> bool:
    """Update job status"""
    try:
        db = get_database_service()
        
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if status == "completed":
            update_data["completed_at"] = datetime.utcnow().isoformat()
        elif status == "failed" and error_message:
            update_data["error_message"] = error_message
        
        response = await db.client.from_("jobs").update(update_data).eq("id", job_id).execute()
        
        if response.data:
            logger.debug(f"Updated job {job_id} status to {status}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error updating job {job_id} status: {e}")
        return False

async def update_job_with_task_id(job_id: str, task_id: str) -> bool:
    """Update job with Celery task ID"""
    try:
        db = get_database_service()
        
        update_data = {
            "celery_task_id": task_id,
            "status": "processing",
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("jobs").update(update_data).eq("id", job_id).execute()
        
        if response.data:
            logger.debug(f"Updated job {job_id} with task_id {task_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error updating job {job_id} with task_id: {e}")
        return False

async def update_job_progress(job_id: str, current_step: int, progress_percentage: Optional[int] = None) -> bool:
    """Update job progress"""
    try:
        db = get_database_service()
        
        update_data = {
            "current_step": current_step,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if progress_percentage is not None:
            update_data["progress_percentage"] = progress_percentage
        
        response = await db.client.from_("jobs").update(update_data).eq("id", job_id).execute()
        
        if response.data:
            logger.debug(f"Updated job {job_id} progress to step {current_step}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error updating job {job_id} progress: {e}")
        return False

async def update_job_result(job_id: str, result: Dict[str, Any]) -> bool:
    """Update job result"""
    try:
        db = get_database_service()
        
        update_data = {
            "result": result,
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("jobs").update(update_data).eq("id", job_id).execute()
        
        if response.data:
            logger.info(f"Updated job {job_id} with result")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error updating job {job_id} result: {e}")
        return False

async def delete_job(job_id: str, user_id: Optional[str] = None) -> bool:
    """Delete a job record"""
    try:
        db = get_database_service()
        
        query = db.client.from_("jobs").delete().eq("id", job_id)
        
        if user_id:
            query = query.eq("user_id", user_id)
            
        response = await query.execute()
        
        if response.data:
            logger.info(f"Deleted job {job_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error deleting job {job_id}: {e}")
        return False

# ============================================================================
# JOB STEP MANAGEMENT
# ============================================================================

async def create_job_step(
    job_id: str,
    step_name: str,
    step_order: int,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Create a job step"""
    try:
        db = get_database_service()
        
        step_data = {
            "id": str(uuid4()),
            "job_id": job_id,
            "step_name": step_name,
            "step_order": step_order,
            "status": "pending",
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        response = await db.client.from_("job_steps").insert(step_data).execute()
        
        if response.data:
            logger.debug(f"Created job step {step_name} for job {job_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error creating job step: {e}")
        return False

async def update_job_step(
    job_id: str,
    step_name: str,
    status: str,
    progress_percentage: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Update job step status and progress"""
    try:
        db = get_database_service()
        
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if progress_percentage is not None:
            update_data["progress_percentage"] = progress_percentage
            
        if metadata:
            update_data["metadata"] = metadata
            
        if status == "completed":
            update_data["completed_at"] = datetime.utcnow().isoformat()
        
        response = await db.client.from_("job_steps").update(update_data).eq("job_id", job_id).eq("step_name", step_name).execute()
        
        if response.data:
            logger.debug(f"Updated job step {step_name} for job {job_id}")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error updating job step: {e}")
        return False

async def get_job_steps(job_id: str) -> List[Dict[str, Any]]:
    """Get all steps for a job"""
    try:
        db = get_database_service()
        
        response = await db.client.from_("job_steps").select("*").eq("job_id", job_id).order("step_order").execute()
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting job steps for {job_id}: {e}")
        return []

async def get_current_job_step(job_id: str) -> Optional[Dict[str, Any]]:
    """Get the current active step for a job"""
    try:
        db = get_database_service()
        
        response = await db.client.from_("job_steps").select("*").eq("job_id", job_id).in_("status", ["processing", "pending"]).order("step_order").limit(1).execute()
        
        return response.data[0] if response.data else None
        
    except Exception as e:
        logger.error(f"Error getting current job step for {job_id}: {e}")
        return None

# ============================================================================
# JOB QUERIES AND FILTERING
# ============================================================================

async def get_jobs_by_user(
    user_id: str,
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Get jobs for a user with optional filtering"""
    try:
        db = get_database_service()
        
        query = db.client.from_("jobs").select("*").eq("user_id", user_id)
        
        if job_type:
            query = query.eq("job_type", job_type)
            
        if status:
            query = query.eq("status", status)
            
        response = await query.order("created_at", desc=True).limit(limit).execute()
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting jobs for user {user_id}: {e}")
        return []

async def get_active_jobs(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all active (pending/processing) jobs"""
    try:
        db = get_database_service()
        
        query = db.client.from_("jobs").select("*").in_("status", ["pending", "processing"])
        
        if user_id:
            query = query.eq("user_id", user_id)
            
        response = await query.order("created_at", desc=True).execute()
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting active jobs: {e}")
        return []

async def get_failed_jobs(user_id: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
    """Get failed jobs from the last N hours"""
    try:
        db = get_database_service()
        
        # Calculate time threshold
        threshold_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        
        query = db.client.from_("jobs").select("*").eq("status", "failed").gte("created_at", threshold_time)
        
        if user_id:
            query = query.eq("user_id", user_id)
            
        response = await query.order("created_at", desc=True).execute()
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting failed jobs: {e}")
        return []

async def get_jobs_by_status(status: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get jobs filtered by status"""
    try:
        db = get_database_service()
        
        query = db.client.from_("jobs").select("*").eq("status", status)
        
        if user_id:
            query = query.eq("user_id", user_id)
            
        response = await query.order("created_at", desc=True).execute()
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting jobs by status {status}: {e}")
        return []

async def get_jobs_by_type(job_type: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get jobs filtered by type"""
    try:
        db = get_database_service()
        
        query = db.client.from_("jobs").select("*").eq("job_type", job_type)
        
        if user_id:
            query = query.eq("user_id", user_id)
            
        response = await query.order("created_at", desc=True).execute()
        
        return response.data or []
        
    except Exception as e:
        logger.error(f"Error getting jobs by type {job_type}: {e}")
        return []

# ============================================================================
# JOB STATUS AND MONITORING
# ============================================================================

async def get_job_status_detailed(job_id: str) -> Dict[str, Any]:
    """Get detailed job status including steps"""
    try:
        db = get_database_service()
        
        # Get job data
        job_data = await get_job_by_id(job_id)
        if not job_data:
            return {"error": "Job not found"}
        
        # Get job steps
        steps = await get_job_steps(job_id)
        
        # Calculate overall progress
        total_steps = len(steps)
        completed_steps = len([step for step in steps if step.get("status") == "completed"])
        
        progress_percentage = 0
        if total_steps > 0:
            progress_percentage = int((completed_steps / total_steps) * 100)
        
        # Get current step
        current_step = await get_current_job_step(job_id)
        
        return {
            "job_id": job_id,
            "status": job_data.get("status"),
            "job_type": job_data.get("job_type"),
            "created_at": job_data.get("created_at"),
            "updated_at": job_data.get("updated_at"),
            "completed_at": job_data.get("completed_at"),
            "progress_percentage": progress_percentage,
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "current_step": current_step.get("step_name") if current_step else None,
            "current_step_name": current_step.get("step_name") if current_step else "",
            "metadata": job_data.get("metadata", {}),
            "result": job_data.get("result"),
            "error_message": job_data.get("error_message"),
            "celery_task_id": job_data.get("celery_task_id"),
            "steps": steps
        }
        
    except Exception as e:
        logger.error(f"Error getting detailed job status for {job_id}: {e}")
        return {"error": str(e)}

async def get_job_statistics(user_id: str, days: int = 30) -> Dict[str, Any]:
    """Get job statistics for a user"""
    try:
        db = get_database_service()
        
        # Calculate time threshold
        threshold_time = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Get all jobs for the user in the time period
        response = await db.client.from_("jobs").select("status, job_type, created_at, completed_at").eq("user_id", user_id).gte("created_at", threshold_time).execute()
        
        jobs_data = response.data or []
        
        # Calculate statistics
        total_jobs = len(jobs_data)
        
        # Status counts
        status_counts = {}
        for job in jobs_data:
            status = job.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Job type counts
        type_counts = {}
        for job in jobs_data:
            job_type = job.get("job_type", "unknown")
            type_counts[job_type] = type_counts.get(job_type, 0) + 1
        
        # Calculate average completion time for completed jobs
        completed_jobs = [job for job in jobs_data if job.get("status") == "completed" and job.get("completed_at")]
        avg_completion_time = 0
        
        if completed_jobs:
            total_time = 0
            for job in completed_jobs:
                created_at = datetime.fromisoformat(job["created_at"].replace("Z", "+00:00"))
                completed_at = datetime.fromisoformat(job["completed_at"].replace("Z", "+00:00"))
                duration = (completed_at - created_at).total_seconds()
                total_time += duration
            
            avg_completion_time = total_time / len(completed_jobs)
        
        return {
            "period_days": days,
            "total_jobs": total_jobs,
            "status_counts": status_counts,
            "type_counts": type_counts,
            "completed_jobs": len(completed_jobs),
            "success_rate": round((len(completed_jobs) / total_jobs) * 100, 2) if total_jobs > 0 else 0,
            "average_completion_time_seconds": round(avg_completion_time, 2),
            "jobs_per_day": round(total_jobs / days, 2) if days > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting job statistics for user {user_id}: {e}")
        return {
            "period_days": days,
            "total_jobs": 0,
            "status_counts": {},
            "type_counts": {},
            "completed_jobs": 0,
            "success_rate": 0,
            "average_completion_time_seconds": 0,
            "jobs_per_day": 0
        }

# ============================================================================
# JOB CLEANUP AND MAINTENANCE
# ============================================================================

async def cleanup_old_jobs(days: int = 30, user_id: Optional[str] = None) -> int:
    """Clean up old completed/failed jobs"""
    try:
        db = get_database_service()
        
        # Calculate cutoff date
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        # Query for old jobs
        query = db.client.from_("jobs").select("id").in_("status", ["completed", "failed"]).lt("created_at", cutoff_date)
        
        if user_id:
            query = query.eq("user_id", user_id)
        
        response = await query.execute()
        
        old_jobs = response.data or []
        
        if not old_jobs:
            return 0
        
        job_ids = [job["id"] for job in old_jobs]
        
        # Delete job steps first (foreign key constraint)
        await db.client.from_("job_steps").delete().in_("job_id", job_ids).execute()
        
        # Delete jobs
        delete_response = await db.client.from_("jobs").delete().in_("id", job_ids).execute()
        
        deleted_count = len(delete_response.data) if delete_response.data else 0
        logger.info(f"Cleaned up {deleted_count} old jobs (older than {days} days)")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error cleaning up old jobs: {e}")
        return 0

async def cancel_stale_jobs(hours: int = 24) -> int:
    """Cancel jobs that have been processing for too long"""
    try:
        db = get_database_service()
        
        # Calculate stale threshold
        stale_threshold = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        
        # Find stale processing jobs
        response = await db.client.from_("jobs").select("id").eq("status", "processing").lt("updated_at", stale_threshold).execute()
        
        stale_jobs = response.data or []
        
        if not stale_jobs:
            return 0
        
        job_ids = [job["id"] for job in stale_jobs]
        
        # Update status to failed
        update_response = await db.client.from_("jobs").update({
            "status": "failed",
            "error_message": f"Job cancelled due to timeout (stale for {hours} hours)",
            "updated_at": datetime.utcnow().isoformat()
        }).in_("id", job_ids).execute()
        
        cancelled_count = len(update_response.data) if update_response.data else 0
        logger.warning(f"Cancelled {cancelled_count} stale jobs (stale for {hours} hours)")
        
        return cancelled_count
        
    except Exception as e:
        logger.error(f"Error cancelling stale jobs: {e}")
        return 0

async def retry_failed_job(job_id: str) -> bool:
    """Retry a failed job by resetting its status"""
    try:
        db = get_database_service()
        
        # Get job data
        job_data = await get_job_by_id(job_id)
        if not job_data or job_data.get("status") != "failed":
            return False
        
        # Reset job status
        update_data = {
            "status": "pending",
            "current_step": 0,
            "error_message": None,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        job_response = await db.client.from_("jobs").update(update_data).eq("id", job_id).execute()
        
        # Reset job steps
        await db.client.from_("job_steps").update({
            "status": "pending",
            "progress_percentage": 0,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("job_id", job_id).execute()
        
        if job_response.data:
            logger.info(f"Reset failed job {job_id} for retry")
            return True
        return False
        
    except Exception as e:
        logger.error(f"Error retrying failed job {job_id}: {e}")
        return False
