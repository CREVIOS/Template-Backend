import asyncio
import json
import threading
import concurrent.futures
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from uuid import uuid4
from loguru import logger
import os

from core.database import get_database_service
from core.api_config import APIConfiguration
from core.template_generator import TemplateGenerator
from celery_tasks import start_file_processing_pipeline

logger.add("logs/background_tasks.log", rotation="10 MB", level="DEBUG")

# Template generation now always uses Celery for reliability

class JobManager:
    """Manages jobs using the unified jobs/job_steps system"""
    
    def __init__(self):
        self.db = get_database_service()
        self.api_config = APIConfiguration()
        self.template_generator = TemplateGenerator(self.api_config)
    
    async def create_job(self, user_id: str, job_type: str, metadata: dict = None, total_steps: int = 1) -> str:
        """Create a new job with steps"""
        job_data = {
            "user_id": user_id,
            "job_type": job_type,
            "status": "pending",
            "total_steps": total_steps,
            "current_step": 0,
            "metadata": metadata or {}
        }
        
        job_result = await self.db.client.from_("jobs").insert(job_data).execute()        
        if not job_result.data:
            raise Exception("Failed to create job")
        
        job_id = job_result.data[0]["id"]
        logger.info(f"Created {job_type} job {job_id}")
        return job_id
    
    async def create_job_step(self, job_id: str, step_name: str, step_order: int):
        """Create a job step"""
        step_data = {
            "job_id": job_id,
            "step_name": step_name,
            "step_order": step_order,
            "status": "pending"
        }
        
        await self.db.client.from_("job_steps").insert(step_data).execute()
        logger.info(f"Created job step {step_name} for job {job_id}")
    
    async def update_job_step(self, job_id: str, step_name: str, status: str, progress: int = 0, error_message: Optional[str] = None, metadata: Optional[dict] = None):
        """Update a job step status"""
        try:
            update_data = {
                "status": status,
                "progress": progress,
                "error_message": error_message,
                "metadata": metadata or {}
            }
            
            if status == "processing":
                update_data["started_at"] = datetime.utcnow().isoformat()
            elif status in ["completed", "failed"]:
                update_data["completed_at"] = datetime.utcnow().isoformat()
            
            # Update the specific job step
            await self.db.client.from_("job_steps").update(update_data).eq("job_id", job_id).eq("step_name", step_name).execute()
            
            # Update the main job current step info
            job_update = {
                "current_step_name": step_name,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Calculate overall progress
            steps_response = await self.db.client.from_("job_steps").select("*").eq("job_id", job_id).order("step_order").execute()
            if steps_response.data:
                completed_steps = sum(1 for step in steps_response.data if step["status"] == "completed")
                total_steps = len(steps_response.data)
                overall_progress = int((completed_steps / total_steps) * 100)
                job_update["progress"] = overall_progress
                job_update["current_step"] = completed_steps + 1 if status == "processing" else completed_steps
            
            if status == "processing":
                job_update["status"] = "processing"
                if not job_update.get("started_at"):
                    job_update["started_at"] = datetime.utcnow().isoformat()
            elif status == "completed":
                # Check if this is the last step
                if steps_response.data:
                    last_step = steps_response.data[-1]
                    if last_step["step_name"] == step_name:
                        job_update["status"] = "completed"
                        job_update["completed_at"] = datetime.utcnow().isoformat()
                        job_update["progress"] = 100
            elif status == "failed":
                job_update["status"] = "failed"
                job_update["error_message"] = error_message
                job_update["completed_at"] = datetime.utcnow().isoformat()
            
            await self.db.client.from_("jobs").update(job_update).eq("id", job_id).execute()
            
            logger.info(f"Updated job {job_id} step '{step_name}' to status '{status}'")
            
        except Exception as e:
            logger.error(f"Failed to update job step for {job_id}.{step_name}: {e}")
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive job status including steps"""
        try:
            # Get main job info
            job_response = await self.db.client.from_("jobs").select("*").eq("id", job_id).execute()
            
            if not job_response.data:
                return {"error": "Job not found"}
            
            job = job_response.data[0]
            
            # Get job steps
            steps_response = await self.db.client.from_("job_steps").select("*").eq("job_id", job_id).order("step_order").execute()
            
            steps = steps_response.data if steps_response.data else []
            
            return {
                "job_id": job_id,
                "job_type": job.get("job_type"),
                "status": job.get("status"),
                "progress": job.get("progress", 0),
                "current_step": job.get("current_step", 0),
                "current_step_name": job.get("current_step_name"),
                "total_steps": job.get("total_steps", 1),
                "error_message": job.get("error_message"),
                "metadata": job.get("metadata", {}),
                "celery_task_id": job.get("celery_task_id"),
                "created_at": job.get("created_at"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "steps": steps
            }
            
        except Exception as e:
            logger.error(f"Error getting job status for {job_id}: {e}")
            return {"error": f"Error retrieving job status: {str(e)}"}

class TemplateGenerationJobManager(JobManager):
    """Manages template generation jobs"""
    
    async def create_template_generation_job(
        self, 
        user_id: str, 
        folder_id: str, 
        priority_template_id: str, 
        template_name: str
    ) -> str:
        """Create a new template generation job"""
        
        # Check if there's already an active job for this specific folder
        existing_job = await self.db.client.from_("jobs").select("id, metadata").eq("user_id", user_id).eq("job_type", "template_generation").in_("status", ["pending", "processing"]).execute()
        
        if existing_job.data:
            # Check if any existing job is for the same folder
            for job in existing_job.data:
                job_metadata = job.get("metadata", {})
                if job_metadata.get("folder_id") == folder_id:
                    logger.info(f"Found existing template generation job {job['id']} for folder {folder_id}")
                    return job["id"]  # Return existing job instead of creating new one
        
        # Check for too many concurrent jobs (limit to 2 per user)
        if existing_job.data and len(existing_job.data) >= 2:
            raise Exception("Too many active template generation jobs. Please wait for existing jobs to complete.")
        
        # Get files in folder
        files_response = await self.db.client.from_("files").select("*").eq("folder_id", folder_id).eq("user_id", user_id).execute()
        
        if not files_response.data:
            raise Exception("No files found in folder")
        
        total_files = len(files_response.data)
        
        # Create job record
        metadata = {
            "folder_id": folder_id,
            "priority_template_id": priority_template_id,
            "template_name": template_name,
            "file_ids": [f["id"] for f in files_response.data],
            "original_filenames": [f["original_filename"] for f in files_response.data],
            "total_files": total_files
        }
        
        job_id = await self.create_job(
            user_id=user_id,
            job_type="template_generation",
            metadata=metadata,
            total_steps=3
        )
        
        # Create job steps
        steps = [
            {"step_name": "process_files", "step_order": 1},
            {"step_name": "generate_template", "step_order": 2},
            {"step_name": "save_template", "step_order": 3}
        ]
        
        for step in steps:
            await self.create_job_step(job_id, step["step_name"], step["step_order"])
        
        # Start background processing using Celery
        from celery_tasks import template_generation_task
        
        task = template_generation_task.delay(job_id)
        
        await self.db.client.from_("jobs").update({
            "celery_task_id": task.id,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", job_id).execute()
        
        logger.info(f"Started Celery template generation task {task.id} for job {job_id}")
        
        return job_id
    
    async def _ensure_files_processed(self, file_ids: List[str], job_id: str):
        """Ensure all files are in the processing pipeline"""
        for file_id in file_ids:
            # Check file status
            file_response = await self.db.client.from_("files").select("status, user_id").eq("id", file_id).single().execute()
            
            if not file_response.data:
                continue
            
            status = file_response.data["status"]
            user_id = file_response.data["user_id"]
            
            # If file needs processing, start the pipeline
            if status in ["uploaded", "queued"]:
                logger.info(f"Starting processing pipeline for file {file_id}")
                start_file_processing_pipeline(file_id, user_id)
    
    async def _wait_for_files_processed(self, file_ids: List[str], job_id: str, max_wait_minutes: int = 10):
        """Wait for all files to be processed"""
        start_time = datetime.utcnow()
        timeout = start_time + timedelta(minutes=max_wait_minutes)
        
        while datetime.utcnow() < timeout:
            # Check status of all files
            files_response = await self.db.client.from_("files").select("id, status, original_filename").in_("id", file_ids).execute()
            
            if not files_response.data:
                raise Exception("Files not found")
            
            processed_count = 0
            error_count = 0
            
            for file_data in files_response.data:
                status = file_data["status"]
                if status == "processed":
                    processed_count += 1
                elif status == "error":
                    error_count += 1
                    logger.warning(f"File {file_data['id']} ({file_data['original_filename']}) has error status")
            
            # Update step progress
            progress = int((processed_count / len(file_ids)) * 80) + 10  # 10-90% range
            await self.update_job_step(job_id, "process_files", "processing", progress, metadata={
                "processed_files": processed_count,
                "error_files": error_count,
                "total_files": len(file_ids)
            })
            
            # Check if we have enough processed files
            if processed_count > 0 and (processed_count + error_count) >= len(file_ids):
                logger.info(f"Files processing complete: {processed_count} processed, {error_count} errors")
                break
            
            # Wait before checking again
            await asyncio.sleep(10)
        
        if processed_count == 0:
            raise Exception("No files were successfully processed")
        
        return processed_count
    
    def _generate_template_content_sync(
        self, 
        folder_id: str, 
        priority_template_id: str, 
        file_ids: List[str],
        job_id: str
    ) -> str:
        """Synchronous template generation for thread execution"""
        try:
            logger.info(f"Starting synchronous template generation for job {job_id}")
            
            # Get processed files with their content
            processed_files = []
            
            for file_id in file_ids:
                # Get file with markdown content and metadata
                file_response = self.db.client.from_("files").select(
                    "*, markdown_content(content)"
                ).eq("id", file_id).eq("status", "processed").execute()
                
                if file_response.data and file_response.data[0].get("markdown_content"):
                    file_data = file_response.data[0]
                    markdown_content = file_data["markdown_content"][0]["content"]
                    
                    # Truncate very large content to prevent timeouts
                    max_content_length = 25000  # Reduced to 25k characters per file
                    if len(markdown_content) > max_content_length:
                        logger.warning(f"File {file_id} content too large ({len(markdown_content)} chars), truncating")
                        markdown_content = markdown_content[:max_content_length] + "\n\n[Content truncated due to length]"
                    
                    processed_files.append({
                        'file_id': file_data['id'],
                        'filename': file_data['original_filename'],
                        'extracted_text': markdown_content,
                        'metadata': file_data.get('extracted_metadata', {}),
                    })
            
            if not processed_files:
                raise Exception("No processed files with content found")
            
            # Update progress
            self.update_job_step(job_id, "generate_template", "processing", 40)
            
            # Reorder files to put priority file first
            if priority_template_id:
                priority_file = next((f for f in processed_files if f['file_id'] == priority_template_id), None)
                if priority_file:
                    processed_files.remove(priority_file)
                    processed_files.insert(0, priority_file)
            
            # Limit number of files to prevent timeout
            max_files = 3  # Reduced to 3 files to prevent timeout
            if len(processed_files) > max_files:
                logger.warning(f"Too many files ({len(processed_files)}), limiting to {max_files}")
                processed_files = processed_files[:max_files]
            
            # Generate template using TemplateGenerator
            if len(processed_files) == 1:
                # Single file
                template_content = self.template_generator.generate_initial_template(
                    processed_files[0]['extracted_text']
                )
            else:
                # Multiple files - but limit to avoid timeouts
                first_contract = processed_files[0]
                template_content = self.template_generator.generate_initial_template(
                    first_contract['extracted_text']
                )
                
                # Update progress
                self.update_job_step(job_id, "generate_template", "processing", 60)
                
                # Update template with additional contracts (limit to 2 additional)
                for contract in processed_files[1:2]:  # Only process 1 additional file
                    template_content = self.template_generator.update_template(
                        template_content, contract['extracted_text']
                    )
            
            # Update progress
            self.update_job_step(job_id, "generate_template", "processing", 80)
            
            # Skip drafting notes for now to prevent timeout
            final_template = self.template_generator.add_drafting_notes(
                template_content, processed_files
            )
            
            logger.info(f"Template generation completed for job {job_id}")
            return final_template
            
        except Exception as e:
            logger.error(f"Synchronous template generation failed: {str(e)}")
            raise Exception(f"AI template generation failed: {str(e)}")
    
    async def _generate_template_content(
        self, 
        folder_id: str, 
        priority_template_id: str, 
        file_ids: List[str],
        job_id: str
    ) -> str:
        """DEPRECATED: Use _generate_template_content_sync instead"""
        return self._generate_template_content_sync(folder_id, priority_template_id, file_ids, job_id)
    

    async def _save_template_to_database(
        self, 
        user_id: str,
        folder_id: str, 
        priority_template_id: str, 
        template_name: str, 
        template_content: str,
        file_ids: List[str]
    ) -> str:
        """Save template to database"""
        
        # Get source filenames
        files_response = await self.db.client.from_("files").select("original_filename").in_("id", file_ids).execute()
        source_files = [f["original_filename"] for f in files_response.data] if files_response.data else []
        
        # Save template to database
        template_data = {
            "folder_id": folder_id,
            "name": template_name,
            "content": template_content,
            "template_type": "ai_generated",
            "file_extension": ".docx",
            "formatting_data": {
                "source_files": source_files,
                "generation_date": datetime.utcnow().isoformat(),
                "ai_generated": True,
                "priority_file": priority_template_id,
                "generation_method": "background_ai"
            },
            "word_compatible": True,
            "is_active": True,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        template_result = await self.db.client.from_("templates").insert(template_data).execute()
        
        if not template_result.data:
            raise Exception("Failed to save template to database")
        
        template_id = template_result.data[0]["id"]
        
        # Track template usage
        try:
            usage_data = {
                "template_id": template_id,
                "user_id": user_id,
                "action_type": "generated",
                "metadata": {
                    "source_files": len(file_ids),
                    "generation_method": "background_ai",
                    "folder_id": folder_id
                }
            }
            await self.db.client.from_("template_usage_stats").insert(usage_data).execute()
        except Exception as e:
            logger.warning(f"Failed to track template usage: {e}")
        
        logger.info(f"Template {template_id} created successfully from {len(file_ids)} files")
        return template_id

    async def cleanup_orphaned_jobs(self, max_age_hours: int = 24):
        """Clean up jobs that are stuck or too old"""
        try:
            from datetime import datetime, timedelta
            
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            cutoff_iso = cutoff_time.isoformat()
            
            # Find jobs that are:
            # 1. Stuck without Celery task ID (old system)
            # 2. OR older than max_age_hours and still pending/processing
            stuck_jobs_query = self.db.client.from_("jobs").select("*").eq("job_type", "template_generation")
            
            # Get stuck jobs (no celery_task_id)
            stuck_without_celery = stuck_jobs_query.in_("status", ["pending", "processing"]).is_("celery_task_id", None).execute()
            
            # Get old jobs
            old_jobs = stuck_jobs_query.in_("status", ["pending", "processing"]).lt("created_at", cutoff_iso).execute()
            
            # Combine and deduplicate
            all_stuck = {}
            if stuck_without_celery.data:
                for job in stuck_without_celery.data:
                    all_stuck[job["id"]] = job
            if old_jobs.data:
                for job in old_jobs.data:
                    all_stuck[job["id"]] = job
            
            cleaned_count = 0
            for job_id, job in all_stuck.items():
                logger.info(f"Cleaning up stuck job {job_id} (created: {job['created_at']})")
                
                # Mark job as failed
                await self.db.client.from_("jobs").update({
                    "status": "failed",
                    "error_message": f"Job auto-cleaned: stuck for {max_age_hours}+ hours or missing Celery task",
                    "completed_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", job_id).execute()
                
                # Clean up job steps
                await self.db.client.from_("job_steps").update({
                    "status": "failed",
                    "error_message": "Parent job auto-cleaned",
                    "completed_at": datetime.utcnow().isoformat()
                }).eq("job_id", job_id).in_("status", ["pending", "processing"]).execute()
                
                cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Auto-cleaned {cleaned_count} stuck template generation jobs")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during job cleanup: {e}")
            return 0

# Global job manager instance
_template_job_manager = None

def get_template_job_manager() -> TemplateGenerationJobManager:
    """Get global template job manager instance"""
    global _template_job_manager
    if _template_job_manager is None:
        _template_job_manager = TemplateGenerationJobManager()
    return _template_job_manager

# Public API functions
async def start_template_generation_background(
    user_id: str,
    folder_id: str,
    priority_template_id: str,
    template_name: str
) -> str:
    """Start template generation in background and return job ID"""
    manager = get_template_job_manager()
    return await manager.create_template_generation_job(
        user_id, folder_id, priority_template_id, template_name
    )

async def get_template_generation_status(job_id: str) -> Dict[str, Any]:
    """Get template generation job status"""
    manager = get_template_job_manager()
    return await manager.get_job_status(job_id)

async def cleanup_template_jobs(max_age_hours: int = 24) -> int:
    """Public function to clean up stuck template generation jobs"""
    manager = get_template_job_manager()
    return await manager.cleanup_orphaned_jobs(max_age_hours)