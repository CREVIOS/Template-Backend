import os
import sys
from celery import Celery, chain
from loguru import logger
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import requests
import hashlib
from functools import lru_cache
import random
from celery.exceptions import SoftTimeLimitExceeded
from celery.exceptions import Retry
from requests.exceptions import RequestException, ConnectionError, Timeout, HTTPError

# Use sync Supabase client for Celery tasks
from supabase import create_client, Client

# Configuration – secrets must come from environment variables (.env or container env)
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

# Basic validation to fail fast in mis-configurations
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables must be set for Celery tasks.")

# Path configuration
sys.path.insert(0, os.path.dirname(__file__))
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

parent_root = project_root.parent
if str(parent_root) not in sys.path:
    sys.path.insert(0, str(parent_root))

os.makedirs("logs", exist_ok=True)

# Celery configuration – allow override via environment variables
BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", BROKER_URL)

celery_app = Celery(
    'celery_tasks',
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=['celery_tasks']
)

# Industry-standard Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Timezone
    timezone='UTC',
    enable_utc=True,
    
    # Task execution
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    
    # Global timeouts - can be overridden per task
    task_soft_time_limit=1800,  # 30 minutes soft limit
    task_time_limit=2400,       # 40 minutes hard limit
    
    # Retry configuration
    task_default_retry_delay=60,
    task_max_retries=5,
    
    # Connection management
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    broker_connection_retry_delay=5,
    
    # Result backend
    result_expires=86400,  # 24 hours
    result_persistent=True,
    
    # Worker configuration
    worker_disable_rate_limits=False,
    worker_enable_remote_control=True,
    worker_send_task_events=True,
    
    # Redis-specific settings for visibility timeout
    broker_transport_options={
        'visibility_timeout': 7200,  # 2 hours
        'retry_policy': {
            'timeout': 5.0
        }
    },
    
    # Result backend transport options
    result_backend_transport_options={
        'visibility_timeout': 7200,
        'retry_policy': {
            'timeout': 5.0
        }
    }
)

# Enhanced logging with structured format
logger.add(
    "logs/celery_tasks.log", 
    rotation="100 MB", 
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    enqueue=True,  # Thread-safe logging
    serialize=True  # JSON format for structured logging
)

# ============================================================================
# BASE TASK CLASSES WITH RETRY STRATEGIES
# ============================================================================

class BaseTaskWithRetry(celery_app.Task):
    """Base task class with industry-standard retry configuration"""
    
    # Default retry configuration for transient failures
    autoretry_for = (
        ConnectionError,
        Timeout,
        HTTPError,
        RequestException,
        OSError,  # Network-related OS errors
        Exception,  # Catch-all for unexpected errors
    )
    
    # Exponential backoff with jitter
    retry_backoff = True
    retry_backoff_max = 600  # Maximum 10 minutes between retries
    retry_jitter = True  # Add randomness to prevent thundering herd
    
    # Task-specific limits (can be overridden)
    max_retries = 5
    default_retry_delay = 30
    
    # Timeout configuration
    soft_time_limit = 1800  # 30 minutes
    time_limit = 2400       # 40 minutes
    
    # Acknowledgment strategy
    acks_late = True
    reject_on_worker_lost = True
    
    def retry(self, args=None, kwargs=None, exc=None, throw=True, eta=None, countdown=None, max_retries=None, **options):
        """Enhanced retry with proper error categorization"""
        
        # Don't retry certain types of errors
        if exc and isinstance(exc, (ValueError, TypeError, KeyError, AttributeError)):
            logger.error(f"Non-retryable error in task {self.name}: {exc}")
            raise exc
        
        # Add jitter to countdown if not specified
        if countdown is None and eta is None:
            countdown = self.default_retry_delay + random.uniform(0, 10)
        
        logger.warning(f"Retrying task {self.name} (attempt {self.request.retries + 1}/{self.max_retries}): {exc}")
        
        return super().retry(
            args=args, 
            kwargs=kwargs, 
            exc=exc, 
            throw=throw, 
            eta=eta, 
            countdown=countdown, 
            max_retries=max_retries,
            **options
        )
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Enhanced failure handling with detailed logging"""
        logger.error(
            f"Task {self.name} failed permanently",
            extra={
                "task_id": task_id,
                "exception": str(exc),
                "args": args,
                "kwargs": kwargs,
                "traceback": str(einfo),
                "retries": self.request.retries
            }
        )
        
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Log retry attempts"""
        logger.warning(
            f"Task {self.name} retry",
            extra={
                "task_id": task_id,
                "exception": str(exc),
                "retry_count": self.request.retries,
                "max_retries": self.max_retries
            }
        )

class OCRTaskRetry(BaseTaskWithRetry):
    """Specialized retry strategy for OCR tasks"""
    
    # More aggressive retry for OCR due to external API
    max_retries = 7
    retry_backoff_max = 300  # 5 minutes max
    soft_time_limit = 600    # 10 minutes
    time_limit = 900         # 15 minutes

class MetadataTaskRetry(BaseTaskWithRetry):
    """Retry strategy for metadata extraction"""
    
    max_retries = 3
    soft_time_limit = 300    # 5 minutes
    time_limit = 600         # 10 minutes

class ClauseTaskRetry(BaseTaskWithRetry):
    """Retry strategy for clause extraction"""
    
    max_retries = 3
    soft_time_limit = 600    # 10 minutes
    time_limit = 900         # 15 minutes

class TemplateTaskRetry(BaseTaskWithRetry):
    """Retry strategy for template operations"""
    
    max_retries = 3
    soft_time_limit = 1200   # 20 minutes
    time_limit = 1800        # 30 minutes

# ============================================================================
# DATABASE CLIENT MANAGEMENT
# ============================================================================

@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Get sync Supabase client for Celery tasks"""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
    
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

class DatabaseManager:
    """Centralized database operations with enhanced error handling and retries"""
    
    def __init__(self):
        self.client = get_supabase_client()
        self.max_db_retries = 3
        self.db_retry_delay = 1
    
    def _execute_with_retry(self, operation, *args, **kwargs):
        """Execute database operation with retry logic"""
        for attempt in range(self.max_db_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_db_retries - 1:
                    logger.error(f"Database operation failed after {self.max_db_retries} attempts: {e}")
                    raise
                
                logger.warning(f"Database operation attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(self.db_retry_delay * (2 ** attempt))  # Exponential backoff
    
    def update_file_status(self, file_id: str, status: str, error_message: Optional[str] = None) -> bool:
        """Update file status with retry logic"""
        if not file_id or not status:
            logger.error("file_id and status are required")
            return False
        
        def _update():
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if error_message:
                update_data["error_message"] = error_message if status == "error" else None
            
            if status in ["markdown_ready", "metadata_ready", "processed"]:
                update_data["processed_at"] = datetime.utcnow().isoformat()
            
            result = self.client.table("files").update(update_data).eq("id", file_id).execute()
            return bool(result.data)
        
        try:
            success = self._execute_with_retry(_update)
            if success:
                logger.info(f"Updated file {file_id} status to '{status}'")
            return success
        except Exception as e:
            logger.error(f"Failed to update file status for {file_id}: {e}")
            return False
    
    def update_job_step(
        self, 
        job_id: str, 
        step_name: str, 
        status: str, 
        progress: int = 0, 
        error_message: Optional[str] = None, 
        metadata: Optional[dict] = None
    ) -> bool:
        """Update job step with enhanced error handling"""
        if not job_id or not step_name or not status:
            logger.error("job_id, step_name, and status are required")
            return False
        
        def _update_step():
            update_data = {
                "status": status,
                "progress": progress,
                "error_message": error_message,
                "metadata": metadata or {},
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if status == "processing":
                update_data["started_at"] = datetime.utcnow().isoformat()
            elif status in ["completed", "failed"]:
                update_data["completed_at"] = datetime.utcnow().isoformat()
            
            # Update the specific job step
            self.client.table("job_steps").update(update_data).eq("job_id", job_id).eq("step_name", step_name).execute()
            
            # Update the main job
            self._update_job_progress(job_id, step_name, status, error_message)
            return True
        
        try:
            self._execute_with_retry(_update_step)
            logger.info(f"Updated job {job_id} step '{step_name}' to status '{status}' with progress {progress}%")
            return True
        except Exception as e:
            logger.error(f"Failed to update job step for {job_id}.{step_name}: {e}")
            return False
    
    def _update_job_progress(self, job_id: str, step_name: str, status: str, error_message: Optional[str] = None):
        """Update main job progress atomically using PostgreSQL function"""
        try:
            # Call the atomic PostgreSQL function via Supabase RPC
            result = self.client.rpc(
                'update_job_progress',
                {
                    'p_job_id': job_id,
                    'p_step_name': step_name,
                    'p_status': status,
                    'p_error_message': error_message
                }
            ).execute()
            
            logger.debug(f"Atomically updated job {job_id} progress: step='{step_name}', status='{status}'")
            
        except Exception as e:
            logger.error(f"Failed to update job progress for {job_id}: {e}")
            # Fallback to manual update if RPC fails (for backwards compatibility)
            try:
                self._update_job_progress_fallback(job_id, step_name, status, error_message)
            except Exception as fallback_error:
                logger.error(f"Fallback job progress update also failed for {job_id}: {fallback_error}")
    
    def _update_job_progress_fallback(self, job_id: str, step_name: str, status: str, error_message: Optional[str] = None):
        """Fallback manual job progress update (original logic)"""
        job_update = {
            "current_step_name": step_name,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if status == "processing":
            job_update["started_at"] = datetime.utcnow().isoformat()
        
        # Get all steps to calculate progress
        steps_response = self.client.table("job_steps").select("*").eq("job_id", job_id).order("step_order").execute()
        if steps_response.data:
            completed_steps = sum(1 for step in steps_response.data if step["status"] == "completed")
            total_steps = len(steps_response.data)
            job_update["progress"] = int((completed_steps / total_steps) * 100)
            job_update["current_step"] = completed_steps + (1 if status == "processing" else 0)
            
            # Check if all steps completed
            last_step = steps_response.data[-1]
            if status == "completed" and last_step["step_name"] == step_name:
                job_update["status"] = "completed"
                job_update["completed_at"] = datetime.utcnow().isoformat()
                job_update["progress"] = 100
        
        if status == "failed":
            job_update["status"] = "failed"
            job_update["error_message"] = error_message
            job_update["completed_at"] = datetime.utcnow().isoformat()
        
        self.client.table("jobs").update(job_update).eq("id", job_id).execute()
    
    def get_file_record(self, file_id: str) -> Optional[Dict]:
        """Get file record with retry logic"""
        if not file_id:
            logger.error("file_id is required")
            return None
        
        def _get_file():
            result = self.client.table("files").select("*").eq("id", file_id).execute()
            return result.data[0] if result.data else None
        
        try:
            return self._execute_with_retry(_get_file)
        except Exception as e:
            logger.error(f"Failed to get file record {file_id}: {e}")
            return None
    
    def save_markdown_content(self, file_id: str, user_id: str, content: str) -> bool:
        """Save markdown content with retry logic"""
        if not all([file_id, user_id, content]):
            logger.error("file_id, user_id, and content are required")
            return False
        
        def _save_content():
            markdown_data = {
                "file_id": file_id,
                "user_id": user_id,
                "content": content,
                "word_count": len(content.split()),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.client.table("markdown_content").insert(markdown_data).execute()
            return bool(result.data)
        
        try:
            return self._execute_with_retry(_save_content)
        except Exception as e:
            logger.error(f"Failed to save markdown content for {file_id}: {e}")
            return False
    
    def get_markdown_content(self, file_id: str) -> Optional[str]:
        """Get markdown content for file with retry logic"""
        if not file_id:
            return None
        
        def _get_content():
            result = self.client.table("markdown_content").select("content").eq("file_id", file_id).execute()
            return result.data[0]["content"] if result.data else None
        
        try:
            return self._execute_with_retry(_get_content)
        except Exception as e:
            logger.error(f"Failed to get markdown content for {file_id}: {e}")
            return None
    
    def save_clause_to_library(self, clause_data: Dict) -> bool:
        """Save clause to clause_library with retry logic"""
        def _save_clause():
            result = self.client.table("clause_library").insert(clause_data).execute()
            return bool(result.data)
        
        try:
            return self._execute_with_retry(_save_clause)
        except Exception as e:
            logger.error(f"Failed to save clause: {e}")
            return False
    
    def create_job(self, job_data: Dict) -> Optional[str]:
        """Create job and return job_id with retry logic"""
        def _create_job():
            result = self.client.table("jobs").insert(job_data).execute()
            return result.data[0]["id"] if result.data else None
        
        try:
            return self._execute_with_retry(_create_job)
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            return None
    
    def create_job_step(self, step_data: Dict) -> bool:
        """Create job step with retry logic"""
        def _create_step():
            result = self.client.table("job_steps").insert(step_data).execute()
            return bool(result.data)
        
        try:
            return self._execute_with_retry(_create_step)
        except Exception as e:
            logger.error(f"Failed to create job step: {e}")
            return False
    
    def update_job_with_task_id(self, job_id: str, task_id: str) -> bool:
        """Update job with Celery task ID"""
        def _update_job():
            result = self.client.table("jobs").update({
                "celery_task_id": task_id,
                "status": "processing",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", job_id).execute()
            return bool(result.data)
        
        try:
            return self._execute_with_retry(_update_job)
        except Exception as e:
            logger.error(f"Failed to update job {job_id} with task_id: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive job status including steps"""
        if not job_id:
            return {"error": "job_id is required"}
        
        def _get_status():
            # Get main job info
            job_response = self.client.table("jobs").select("*").eq("id", job_id).execute()
            
            if not job_response.data:
                return {"error": "Job not found"}
            
            job = job_response.data[0]
            
            # Get job steps
            steps_response = self.client.table("job_steps").select("*").eq("job_id", job_id).order("step_order").execute()
            steps = steps_response.data if steps_response.data else []
            
            return {
                "job_id": job_id,
                "job_type": job.get("job_type"),
                "user_id": job.get("user_id"),
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
        
        try:
            return self._execute_with_retry(_get_status)
        except Exception as e:
            logger.error(f"Error getting job status for {job_id}: {e}")
            return {"error": f"Error retrieving job status: {str(e)}"}
    
    def create_template_placeholder(self, template_data: Dict) -> Optional[str]:
        """Create template with null content initially"""
        def _create_template():
            # Ensure content is null for placeholder
            template_data["content"] = None
            template_data["status"] = "generating"
            template_data["created_at"] = datetime.utcnow().isoformat()
            template_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = self.client.table("templates").insert(template_data).execute()
            return result.data[0]["id"] if result.data else None
        
        try:
            template_id = self._execute_with_retry(_create_template)
            if template_id:
                logger.info(f"Created template placeholder with ID: {template_id}")
            return template_id
        except Exception as e:
            logger.error(f"Failed to create template placeholder: {e}")
            return None
    
    def update_template_content(self, template_id: str, content: str, content_json: Optional[Dict] = None) -> bool:
        """Update template with generated content"""
        def _update_template():
            update_data = {
                "content": content,
                "status": "completed",
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if content_json:
                update_data["content_json"] = content_json
            
            result = self.client.table("templates").update(update_data).eq("id", template_id).execute()
            return bool(result.data)
        
        try:
            success = self._execute_with_retry(_update_template)
            if success:
                logger.info(f"Updated template {template_id} with generated content")
            return success
        except Exception as e:
            logger.error(f"Failed to update template content for {template_id}: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager()

# ============================================================================
# DOCUMENT PROCESSING UTILITIES
# ============================================================================

def get_document_processor():
    """Get document processor instance with error handling"""
    try:
        from core.api_config import APIConfiguration
        from core.document_processor import DocumentProcessor
        
        api_config = APIConfiguration()
        return DocumentProcessor(api_config)
    except Exception as e:
        logger.error(f"Failed to initialize document processor: {e}")
        raise

def _extract_content_from_ocr_response(ocr_data: Dict) -> str:
    """Extract text content from OCR response with enhanced error handling"""
    if not ocr_data:
        raise ValueError("Empty OCR response received")
    
    markdown_content = ""
    
    if not isinstance(ocr_data, dict):
        content = str(ocr_data).strip()
        if not content:
            raise ValueError("No content extracted from OCR response")
        return content
    
    # Check various response formats with logging
    if 'pages' in ocr_data and isinstance(ocr_data['pages'], list):
        logger.info(f"🔍 Found pages array with {len(ocr_data['pages'])} pages")
        for i, page in enumerate(ocr_data['pages']):
            if isinstance(page, dict):
                page_content = (
                    page.get('content') or 
                    page.get('text') or 
                    page.get('markdown') or 
                    page.get('extracted_text') or
                    page.get('ocr_text') or ''
                )
                if page_content:
                    markdown_content += str(page_content) + "\n\n"
                    logger.debug(f"Extracted {len(str(page_content))} chars from page {i+1}")
    
    # Check for direct content fields
    elif any(key in ocr_data for key in ['content', 'text', 'markdown', 'extracted_text', 'ocr_text']):
        markdown_content = (
            ocr_data.get('content') or 
            ocr_data.get('text') or 
            ocr_data.get('markdown') or 
            ocr_data.get('extracted_text') or
            ocr_data.get('ocr_text') or ''
        )
        logger.debug(f"Extracted content directly: {len(str(markdown_content))} chars")
    
    # Check for result/data wrapper
    elif 'result' in ocr_data:
        result = ocr_data['result']
        if isinstance(result, dict):
            markdown_content = (
                result.get('content') or 
                result.get('text') or 
                result.get('markdown') or ''
            )
        elif isinstance(result, str):
            markdown_content = result
        logger.debug(f"Extracted from result wrapper: {len(str(markdown_content))} chars")
    
    elif 'data' in ocr_data:
        data = ocr_data['data']
        if isinstance(data, dict):
            markdown_content = (
                data.get('content') or 
                data.get('text') or 
                data.get('markdown') or ''
            )
        elif isinstance(data, str):
            markdown_content = data
        logger.debug(f"Extracted from data wrapper: {len(str(markdown_content))} chars")
    
    final_content = str(markdown_content).strip()
    if not final_content:
        raise ValueError("No content could be extracted from OCR response")
    
    logger.info(f"✅ Successfully extracted {len(final_content)} characters from OCR response")
    return final_content

def _process_with_mistral_ocr(download_url: str, file_record: Dict, job_id: Optional[str] = None) -> str:
    """Process document with Mistral OCR API with enhanced error handling and circuit breaker"""
    
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is not configured")
    
    try:
        logger.info(f"🤖 Starting Mistral OCR processing for file: {file_record.get('original_filename', 'unknown')}")
        
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "DocProcessor/1.0"
        }
        
        payload = {
            "model": "mistral-ocr-latest",
            "id": f"ocr-{int(time.time())}-{file_record.get('id', 'unknown')}",
            "document": {
                "document_url": download_url,
                "document_name": file_record.get("original_filename", "document"),
                "type": "document_url"
            },
            "include_image_base64": False,
            "image_limit": 0,
            "image_min_size": 0
        }
        
        # Update progress
        if job_id:
            db_manager.update_job_step(job_id, "extract_markdown", "processing", 50)
        
        # Make request with proper timeout and retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Making OCR API request (attempt {attempt + 1}/{max_retries})")
                
                ocr_response = requests.post(
                    "https://api.mistral.ai/v1/ocr",
                    headers=headers,
                    json=payload,
                    timeout=(30, 300)  # (connection timeout, read timeout)
                )
                
                logger.info(f"📊 Mistral OCR API response status: {ocr_response.status_code}")
                
                if ocr_response.status_code == 200:
                    break
                elif ocr_response.status_code in [429, 503, 502, 504]:
                    # Rate limiting or server errors - retry with backoff
                    if attempt < max_retries - 1:
                        backoff_time = (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"OCR API rate limited/server error (status {ocr_response.status_code}), retrying in {backoff_time:.1f}s")
                        time.sleep(backoff_time)
                        continue
                
                # Other HTTP errors
                error_message = f"Mistral OCR API error (status {ocr_response.status_code}): {ocr_response.text}"
                logger.error(error_message)
                raise HTTPError(error_message)
                
            except (ConnectionError, Timeout) as e:
                if attempt < max_retries - 1:
                    backoff_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"OCR API connection error: {e}, retrying in {backoff_time:.1f}s")
                    time.sleep(backoff_time)
                    continue
                raise ConnectionError(f"OCR API connection failed after {max_retries} attempts: {e}")
        
        # Update progress
        if job_id:
            db_manager.update_job_step(job_id, "extract_markdown", "processing", 70)
        
        # Parse and extract content
        try:
            ocr_data = ocr_response.json()
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from OCR API: {e}")
        
        markdown_content = _extract_content_from_ocr_response(ocr_data)
        
        if len(markdown_content) < 10:  # Sanity check
            raise ValueError(f"OCR content too short ({len(markdown_content)} chars), likely extraction failed")
        
        logger.info(f"✅ OCR completed successfully, generated {len(markdown_content)} characters")
        return markdown_content
        
    except (ValueError, ConnectionError, HTTPError) as e:
        error_message = f"OCR processing failed: {str(e)}"
        logger.error(error_message)
        raise
    except Exception as e:
        error_message = f"Unexpected error in OCR processing: {str(e)}"
        logger.error(error_message, exc_info=True)
        raise RuntimeError(error_message) from e

# ============================================================================
# ENHANCED CELERY TASKS
# ============================================================================

@celery_app.task
def monitor_job_concurrency():
    """Monitor concurrent job execution and detect potential race conditions"""
    try:
        # Get jobs that are currently processing
        processing_jobs = get_supabase_client().table("jobs").select("*").eq("status", "processing").execute()
        
        # Group by job type to see concurrency patterns
        job_types = {}
        concurrent_steps = {}
        
        for job in processing_jobs.data if processing_jobs.data else []:
            job_type = job.get("job_type", "unknown")
            current_step = job.get("current_step_name", "unknown")
            
            if job_type not in job_types:
                job_types[job_type] = 0
            job_types[job_type] += 1
            
            if current_step not in concurrent_steps:
                concurrent_steps[current_step] = 0
            concurrent_steps[current_step] += 1
        
        # Check for high concurrency that might cause issues
        warnings = []
        if job_types.get("template_generation", 0) > 5:
            warnings.append("High template generation concurrency detected")
        if concurrent_steps.get("extract_markdown", 0) > 10:
            warnings.append("High OCR concurrency detected")
        
        monitoring_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "processing_jobs_total": len(processing_jobs.data) if processing_jobs.data else 0,
            "job_types": job_types,
            "concurrent_steps": concurrent_steps,
            "warnings": warnings
        }
        
        logger.info(f"Job concurrency monitoring: {monitoring_result}")
        return monitoring_result
        
    except Exception as e:
        logger.error(f"Job concurrency monitoring failed: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

@celery_app.task(bind=True, base=OCRTaskRetry)
def extract_markdown_task(self, file_id: str, job_id: Optional[str] = None):
    """Extract markdown content from document using Mistral OCR API with enhanced retry logic"""
    if not file_id:
        raise ValueError("file_id is required")
    
    try:
        logger.info(f"🔄 Starting markdown extraction for file: {file_id}")
        
        if job_id:
            db_manager.update_job_step(job_id, "extract_markdown", "processing", 10)
        
        # Get file record
        file_record = db_manager.get_file_record(file_id)
        if not file_record:
            raise ValueError(f"File {file_id} not found")
        
        # Check if already processed (idempotency)
        if file_record.get("status") in ["markdown_ready", "processed"]:
            logger.info(f"File {file_id} already processed, skipping")
            if job_id:
                db_manager.update_job_step(job_id, "extract_markdown", "completed", 100)
            return {"status": "already_processed", "file_id": file_id}
        
        # Update status
        db_manager.update_file_status(file_id, "converting", "Converting document to markdown")
        
        # Get signed URL with retry
        try:
            signed_url_response = get_supabase_client().storage.from_("documents").create_signed_url(
                file_record["storage_path"], 
                expires_in=7200  # 2 hours
            )
            
            if not signed_url_response or 'signedURL' not in signed_url_response:
                raise ConnectionError("Failed to generate signed URL")
                
            download_url = signed_url_response['signedURL']
            logger.info(f"📄 Generated signed URL for file: {file_record['storage_path']}")
            
        except Exception as e:
            error_msg = f"Failed to generate signed URL: {str(e)}"
            db_manager.update_file_status(file_id, "error", error_msg)
            if job_id:
                db_manager.update_job_step(job_id, "extract_markdown", "failed", 0, error_msg)
            raise ConnectionError(error_msg) from e
        
        if job_id:
            db_manager.update_job_step(job_id, "extract_markdown", "processing", 30)
        
        # Process with Mistral OCR (has its own retry logic)
        try:
            markdown_content = _process_with_mistral_ocr(download_url, file_record, job_id)
        except SoftTimeLimitExceeded:
            logger.warning(f"OCR task soft time limit exceeded for file {file_id}")
            # Move to a different queue with longer timeout
            raise self.retry(countdown=60, queue='long_running_tasks')
        
        # Save markdown content
        if not db_manager.save_markdown_content(
            file_id, 
            file_record["user_id"], 
            markdown_content
        ):
            raise RuntimeError("Failed to save markdown content to database")
        
        # Update file status
        db_manager.update_file_status(file_id, "markdown_ready")
        
        if job_id:
            db_manager.update_job_step(job_id, "extract_markdown", "completed", 100, metadata={
                "word_count": len(markdown_content.split()),
                "content_length": len(markdown_content),
                "processing_time": time.time()
            })
        
        logger.info(f"✅ Markdown extraction completed successfully for file: {file_id}")
        return {
            "status": "success", 
            "word_count": len(markdown_content.split()), 
            "content_length": len(markdown_content),
            "file_id": file_id
        }
        
    except (ValueError, ConnectionError, HTTPError) as e:
        # Non-retryable errors
        logger.error(f"❌ Non-retryable error in markdown extraction for {file_id}: {str(e)}")
        db_manager.update_file_status(file_id, "error", str(e))
        if job_id:
            db_manager.update_job_step(job_id, "extract_markdown", "failed", 0, str(e))
        raise
        
    except Exception as e:
        logger.error(f"❌ Markdown extraction task failed for file {file_id}: {str(e)}")
        db_manager.update_file_status(file_id, "error", str(e))
        if job_id:
            db_manager.update_job_step(job_id, "extract_markdown", "failed", 0, str(e))
        
        # Let the base class handle retries
        raise

@celery_app.task(bind=True, base=MetadataTaskRetry)
def extract_metadata_task(self, file_id: str, job_id: Optional[str] = None):
    """Extract metadata from markdown content using AI with enhanced retry logic"""
    if not file_id:
        raise ValueError("file_id is required")
    
    try:
        logger.info(f"Starting metadata extraction for file: {file_id}")
        
        if job_id:
            db_manager.update_job_step(job_id, "extract_metadata", "processing", 10)
        
        db_manager.update_file_status(file_id, "extracting_metadata", "Extracting document metadata")
        
        # Get file record
        file_record = db_manager.get_file_record(file_id)
        if not file_record:
            raise ValueError(f"File {file_id} not found")
        
        # Check if already processed (idempotency)
        if file_record.get("status") in ["metadata_ready", "processed"]:
            logger.info(f"File {file_id} metadata already processed, skipping")
            if job_id:
                db_manager.update_job_step(job_id, "extract_metadata", "completed", 100)
            return {"status": "already_processed", "file_id": file_id}
        
        # Get markdown content
        markdown_content = db_manager.get_markdown_content(file_id)
        if not markdown_content:
            raise ValueError("No markdown content found for metadata extraction")
        
        if job_id:
            db_manager.update_job_step(job_id, "extract_metadata", "processing", 50)
        
        # Extract metadata using AI with timeout handling
        try:
            processor = get_document_processor()
            metadata = processor.extract_metadata_from_text(
                markdown_content, 
                file_record.get("original_filename", "unknown")
            )
        except SoftTimeLimitExceeded:
            logger.warning(f"Metadata extraction soft time limit exceeded for file {file_id}")
            raise self.retry(countdown=30)
        
        # Save metadata
        result = get_supabase_client().table("files").update({
            "extracted_metadata": metadata,
            "status": "metadata_ready",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", file_id).execute()
        
        if not result.data:
            raise RuntimeError("Failed to save metadata to database")
        
        if job_id:
            db_manager.update_job_step(job_id, "extract_metadata", "completed", 100, metadata={
                "extracted_metadata": metadata,
                "processing_time": time.time()
            })
        
        logger.info(f"✅ Metadata extraction completed for file: {file_id}")
        return {"status": "success", "metadata": metadata, "file_id": file_id}
        
    except ValueError as e:
        logger.error(f"❌ Validation error in metadata extraction for {file_id}: {str(e)}")
        db_manager.update_file_status(file_id, "error", f"Metadata extraction failed: {str(e)}")
        if job_id:
            db_manager.update_job_step(job_id, "extract_metadata", "failed", 50, str(e))
        raise
        
    except Exception as e:
        logger.error(f"❌ Metadata extraction failed for file {file_id}: {str(e)}")
        db_manager.update_file_status(file_id, "error", f"Metadata extraction failed: {str(e)}")
        if job_id:
            db_manager.update_job_step(job_id, "extract_metadata", "failed", 50, str(e))
        
        # Let base class handle retries
        raise

@celery_app.task(bind=True, base=ClauseTaskRetry)
def extract_clauses_task(self, file_id: str, job_id: Optional[str] = None):
    """Extract legal clauses from markdown content with enhanced retry logic"""
    if not file_id:
        raise ValueError("file_id is required")
    
    try:
        logger.info(f"Starting clause extraction for file: {file_id}")
        
        if job_id:
            db_manager.update_job_step(job_id, "extract_clauses", "processing", 10)
        
        db_manager.update_file_status(file_id, "extracting_clauses", "Extracting legal clauses")
        
        # Get file record
        file_record = db_manager.get_file_record(file_id)
        if not file_record:
            raise ValueError(f"File {file_id} not found")
        
        # Check if already processed (idempotency)
        if file_record.get("status") == "processed":
            logger.info(f"File {file_id} clauses already processed, skipping")
            if job_id:
                db_manager.update_job_step(job_id, "extract_clauses", "completed", 100)
            return {"status": "already_processed", "file_id": file_id}
        
        metadata = file_record.get("extracted_metadata", {})
        markdown_content = db_manager.get_markdown_content(file_id)
        
        if not markdown_content:
            raise ValueError("No markdown content found for clause extraction")
        
        if job_id:
            db_manager.update_job_step(job_id, "extract_clauses", "processing", 30)
        
        # Extract clauses using AI with timeout handling
        clauses = []
        try:
            processor = get_document_processor()
            clauses = processor.extract_clauses_from_text(markdown_content, metadata)
        except SoftTimeLimitExceeded:
            logger.warning(f"Clause extraction soft time limit exceeded for file {file_id}")
            raise self.retry(countdown=60)
        except Exception as ai_error:
            logger.warning(f"AI clause extraction failed for {file_id}: {ai_error}")
            # Create a basic clause entry to indicate processing was attempted
            clauses = [{
                "clause_text": f"Document processed but clause extraction failed: {str(ai_error)}",
                "clause_type": "processing_note",
                "position_context": "N/A",
                "clause_purpose": "Error documentation",
                "source_contract": file_record.get("original_filename", "unknown"),
                "contract_filename": file_record.get("original_filename", "unknown")
            }]
        
        if job_id:
            db_manager.update_job_step(job_id, "extract_clauses", "processing", 70)
        
        # Save clauses to clause_library
        clauses_saved = 0
        if clauses:
            for clause in clauses:
                clause_text = clause.get("clause_text", "")
                if not clause_text:
                    continue
                
                # Create a hash for deduplication
                clause_hash = hashlib.sha256(clause_text.encode('utf-8')).hexdigest()
                
                clause_data = {
                    "user_id": file_record["user_id"],
                    "file_id": file_id,
                    "folder_id": file_record["folder_id"],
                    "clause_type": clause.get("clause_type", "general"),
                    "clause_text": clause_text,
                    "clause_metadata": {
                        "position_context": clause.get("position_context"),
                        "clause_purpose": clause.get("clause_purpose"),
                        "relevance_assessment": clause.get("relevance_assessment", {}),
                        "source_contract": clause.get("source_contract"),
                        "contract_filename": clause.get("contract_filename"),
                        "contract_parties": clause.get("contract_parties", []),
                        "contract_date": clause.get("contract_date"),
                        "extraction_method": "ai_gemini",
                        "clause_hash": clause_hash,
                        "extraction_timestamp": datetime.utcnow().isoformat()
                    },
                    "created_at": datetime.utcnow().isoformat()
                }
                
                if db_manager.save_clause_to_library(clause_data):
                    clauses_saved += 1
        
        # Update file status to processed
        db_manager.update_file_status(file_id, "processed")
        
        if job_id:
            db_manager.update_job_step(job_id, "extract_clauses", "completed", 100, metadata={
                "clauses_extracted": len(clauses),
                "clauses_saved": clauses_saved,
                "processing_time": time.time()
            })
        
        logger.info(f"✅ Clause extraction completed for file: {file_id}, saved {clauses_saved} clauses")
        return {"status": "success", "clauses_extracted": len(clauses), "clauses_saved": clauses_saved, "file_id": file_id}
        
    except ValueError as e:
        logger.error(f"❌ Validation error in clause extraction for {file_id}: {str(e)}")
        db_manager.update_file_status(file_id, "error", f"Clause extraction failed: {str(e)}")
        if job_id:
            db_manager.update_job_step(job_id, "extract_clauses", "failed", 70, str(e))
        raise
        
    except Exception as e:
        logger.error(f"❌ Clause extraction failed for file {file_id}: {str(e)}")
        db_manager.update_file_status(file_id, "error", f"Clause extraction failed: {str(e)}")
        if job_id:
            db_manager.update_job_step(job_id, "extract_clauses", "failed", 70, str(e))
        
        # Let base class handle retries
        raise

# ============================================================================
# ENHANCED TEMPLATE GENERATION TASK
# ============================================================================

@celery_app.task(bind=True, base=TemplateTaskRetry)
def template_generation_task(self, job_id: str):
    """Enhanced template generation task with placeholder creation"""
    if not job_id:
        raise ValueError("job_id is required")
    
    template_id = None
    
    try:
        logger.info(f"Starting template generation Celery task for job: {job_id}")
        
        # Get job details
        job_status = db_manager.get_job_status(job_id)
        if "error" in job_status:
            raise ValueError(f"Job {job_id} not found")
        
        metadata = job_status["metadata"]
        user_id = job_status.get("user_id")
        folder_id = metadata["folder_id"]
        priority_template_id = metadata["priority_template_id"]
        template_name = metadata["template_name"]
        file_ids = metadata["file_ids"]
        
        # Update job with Celery task ID
        db_manager.update_job_with_task_id(job_id, self.request.id)
        
        # Step 0: Create template placeholder immediately
        db_manager.update_job_step(job_id, "create_placeholder", "processing", 5)
        
        template_placeholder_data = {
            "folder_id": folder_id,
            "name": template_name,
            "content": None,  # Null content initially
            "template_type": "ai_generated",
            "file_extension": ".docx",
            "formatting_data": {
                "generation_status": "in_progress",
                "job_id": job_id,
                "source_files": [],
                "generation_date": datetime.utcnow().isoformat(),
                "ai_generated": True,
                "priority_file": priority_template_id
            },
            "word_compatible": True,
            "is_active": False,  # Not active until content is generated
            "status": "generating"
        }
        
        template_id = db_manager.create_template_placeholder(template_placeholder_data)
        if not template_id:
            raise RuntimeError("Failed to create template placeholder")
        
        db_manager.update_job_step(job_id, "create_placeholder", "completed", 100, metadata={
            "template_id": template_id
        })
        
        # Step 1: Ensure all files are processed
        db_manager.update_job_step(job_id, "process_files", "processing", 10)
        _ensure_files_processed(file_ids, job_id)
        
        # Step 2: Wait for all files to be processed
        processed_count = _wait_for_files_processed(file_ids, job_id)
        db_manager.update_job_step(job_id, "process_files", "completed", 100, metadata={
            "processed_files": processed_count,
            "total_files": len(file_ids)
        })
        
        # Step 3: Generate template content
        db_manager.update_job_step(job_id, "generate_template", "processing", 20)
        
        try:
            template_content = _generate_template_content(
                folder_id, priority_template_id, file_ids, job_id
            )
        except SoftTimeLimitExceeded:
            logger.warning(f"Template generation soft time limit exceeded for job {job_id}")
            # Update placeholder status and retry
            if template_id:
                get_supabase_client().table("templates").update({
                    "formatting_data": {
                        **template_placeholder_data["formatting_data"],
                        "generation_status": "retrying",
                        "last_retry": datetime.utcnow().isoformat()
                    }
                }).eq("id", template_id).execute()
            raise self.retry(countdown=120, queue='long_running_tasks')
        
        db_manager.update_job_step(job_id, "generate_template", "completed", 100, metadata={
            "template_length": len(template_content)
        })
        
        # Step 4: Update template with generated content
        db_manager.update_job_step(job_id, "save_template", "processing", 50)
        
        # Extract clauses and create content_json
        content_json = template_content  # Default to the content itself
        
        try:
            from core.template_generator import TemplateGenerator
            from core.api_config import APIConfiguration
            
            api_config = APIConfiguration()
            if api_config.is_configured():
                template_generator = TemplateGenerator(api_config)
                content_json = template_generator.add_drafting_notes(template_content)
                logger.info(f"✅ Successfully added drafting notes to template")
        except Exception as e:
            logger.warning(f"Failed to add drafting notes: {e}, using content as-is")
        
        # Update template with content
        success = db_manager.update_template_content(template_id, template_content, content_json)
        if not success:
            raise RuntimeError("Failed to update template with generated content")
        
        # Activate the template
        get_supabase_client().table("templates").update({
            "is_active": True,
            "status": "completed"
        }).eq("id", template_id).execute()
        
        db_manager.update_job_step(job_id, "save_template", "completed", 100, metadata={
            "template_id": template_id,
            "content_length": len(template_content)
        })
        
        # Update job result
        get_supabase_client().table("jobs").update({
            "result": {
                "template_id": template_id,
                "template_name": template_name,
                "processed_files": processed_count,
                "action": "template_generated"
            },
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", job_id).execute()
        
        # Track template usage
        try:
            usage_data = {
                "template_id": template_id,
                "user_id": user_id,
                "action_type": "generated",
                "metadata": {
                    "source_files": len(file_ids),
                    "folder_id": folder_id,
                    "generation_method": "celery_enhanced"
                }
            }
            get_supabase_client().table("template_usage_stats").insert(usage_data).execute()
        except Exception as e:
            logger.warning(f"Failed to track template usage: {e}")
        
        # Invalidate cache for the user
        try:
            from core.redis_cache import invalidate_user_cache_sync
            cache_invalidated = invalidate_user_cache_sync(user_id)
            if cache_invalidated:
                logger.info(f"✅ Cache invalidated for user {user_id} after template generation")
            else:
                logger.warning(f"⚠️  Cache invalidation failed for user {user_id}")
        except Exception as cache_error:
            logger.error(f"❌ Error invalidating cache for user {user_id}: {cache_error}")
        
        logger.info(f"Template generation task {job_id} completed successfully with template ID: {template_id}")
        
        return {
            "job_id": job_id,
            "template_id": template_id,
            "status": "completed",
            "processed_files": processed_count
        }
        
    except Exception as e:
        logger.error(f"Template generation task {job_id} failed: {str(e)}")
        
        # Update template placeholder with error status
        if template_id:
            try:
                get_supabase_client().table("templates").update({
                    "status": "failed",
                    "formatting_data": {
                        **template_placeholder_data.get("formatting_data", {}),
                        "generation_status": "failed",
                        "error_message": str(e),
                        "failed_at": datetime.utcnow().isoformat()
                    }
                }).eq("id", template_id).execute()
            except:
                pass
        
        # Update job status
        try:
            current_step_name = job_status.get("current_step_name", "unknown")
            db_manager.update_job_step(job_id, current_step_name, "failed", 0, str(e))
        except:
            pass
        
        # Let base class handle retries
        raise

@celery_app.task(bind=True, base=TemplateTaskRetry)
def template_update_task(self, job_id: str):
    """Enhanced template update task with better error handling"""
    if not job_id:
        raise ValueError("job_id is required")
    
    try:
        logger.info(f"Starting template update Celery task for job: {job_id}")
        
        # Get job details
        job_status = db_manager.get_job_status(job_id)
        if "error" in job_status:
            raise ValueError(f"Job {job_id} not found")
        
        metadata = job_status["metadata"]
        user_id = job_status.get("user_id")
        template_id = metadata["template_id"]
        template_name = metadata["template_name"]
        file_ids = metadata["file_ids"]
        
        # Update job with Celery task ID
        db_manager.update_job_with_task_id(job_id, self.request.id)
        
        # Step 1: Ensure files are processed
        db_manager.update_job_step(job_id, "process_files", "processing", 10)
        _ensure_files_processed(file_ids, job_id)
        
        processed_count = _wait_for_files_processed(file_ids, job_id, max_wait_minutes=15)
        db_manager.update_job_step(job_id, "process_files", "completed", 100, metadata={
            "processed_files": processed_count,
            "total_files": len(file_ids)
        })
        
        # Step 2: Update template with new files
        db_manager.update_job_step(job_id, "update_template", "processing", 20)
        
        try:
            updated_template_content = _update_template_with_files(
                template_id, file_ids, job_id
            )
        except SoftTimeLimitExceeded:
            logger.warning(f"Template update soft time limit exceeded for job {job_id}")
            raise self.retry(countdown=120, queue='long_running_tasks')
        
        db_manager.update_job_step(job_id, "update_template", "completed", 100, metadata={
            "updated_template_length": len(updated_template_content)
        })
        
        # Step 3: Extract clauses and save updated template
        db_manager.update_job_step(job_id, "extract_clauses", "processing", 50)
        final_template_id = _save_updated_template_with_clauses(
            template_id, updated_template_content, file_ids, job_id
        )
        db_manager.update_job_step(job_id, "extract_clauses", "completed", 100, metadata={
            "final_template_id": final_template_id
        })
        
        # Update job result
        get_supabase_client().table("jobs").update({
            "result": {
                "template_id": final_template_id,
                "template_name": template_name,
                "updated_files": processed_count,
                "action": "template_updated"
            },
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", job_id).execute()
        
        # Invalidate cache for the user
        try:
            from core.redis_cache import invalidate_user_cache_sync
            cache_invalidated = invalidate_user_cache_sync(user_id)
            if cache_invalidated:
                logger.info(f"✅ Cache invalidated for user {user_id} after template update")
        except Exception as cache_error:
            logger.error(f"❌ Error invalidating cache for user {user_id}: {cache_error}")
        
        logger.info(f"Template update task {job_id} completed successfully")
        
        return {
            "job_id": job_id,
            "template_id": final_template_id,
            "status": "completed",
            "updated_files": processed_count
        }
        
    except Exception as e:
        logger.error(f"Template update task {job_id} failed: {str(e)}")
        
        # Update job status
        try:
            current_step_name = job_status.get("current_step_name", "unknown")
            db_manager.update_job_step(job_id, current_step_name, "failed", 0, str(e))
        except:
            pass
        
        # Let base class handle retries
        raise

# ============================================================================
# HELPER FUNCTIONS (UNCHANGED LOGIC)
# ============================================================================

def _ensure_files_processed(file_ids: List[str], job_id: str):
    """Ensure all files are processed or start processing pipeline"""
    for file_id in file_ids:
        file_record = db_manager.get_file_record(file_id)
        if not file_record:
            continue
        
        status = file_record["status"]
        user_id = file_record["user_id"]
        
        # If file needs processing, start the pipeline
        if status in ["uploaded", "queued"]:
            logger.info(f"Starting processing pipeline for file {file_id}")
            start_file_processing_pipeline(file_id, user_id)

def _wait_for_files_processed(file_ids: List[str], job_id: str, max_wait_minutes: int = 15) -> int:
    """Wait for files to be processed with timeout"""
    # Guard against empty file_ids to prevent division by zero
    if not file_ids:
        logger.warning(f"Job {job_id}: No files to process, returning 0")
        db_manager.update_job_step(job_id, "process_files", "completed", 100)
        return 0
    
    start_time = datetime.utcnow()
    timeout = start_time + timedelta(minutes=max_wait_minutes)
    
    while datetime.utcnow() < timeout:
        # Check status of all files
        files_response = get_supabase_client().table("files").select("id, status, original_filename").in_("id", file_ids).execute()
        
        if not files_response.data:
            raise RuntimeError("Files not found")
        
        processed_count = 0
        error_count = 0
        
        for file_data in files_response.data:
            status = file_data["status"]
            if status == "processed":
                processed_count += 1
            elif status == "error":
                error_count += 1
                logger.warning(f"File {file_data['id']} ({file_data['original_filename']}) has error status")
        
        # Update progress
        progress = int((processed_count / len(file_ids)) * 80) + 10
        db_manager.update_job_step(job_id, "process_files", "processing", progress, metadata={
            "processed_files": processed_count,
            "error_files": error_count,
            "total_files": len(file_ids)
        })
        
        # Check if processing complete
        if processed_count > 0 and (processed_count + error_count) >= len(file_ids):
            logger.info(f"Files processing complete: {processed_count} processed, {error_count} errors")
            break
        
        time.sleep(10)
    
    if processed_count == 0:
        raise RuntimeError("No files were successfully processed")
    
    return processed_count

def _generate_template_content(
    folder_id: str, 
    priority_template_id: str, 
    file_ids: List[str],
    job_id: str
) -> str:
    """Generate template content using AI (unchanged logic)"""
    try:
        logger.info(f"Starting template generation for job {job_id}")
        
        from core.api_config import APIConfiguration
        from core.template_generator import TemplateGenerator
        
        api_config = APIConfiguration()
        template_generator = TemplateGenerator(api_config)
        
        # Get processed files with content
        processed_files = []
        
        for file_id in file_ids:
            # Get file with markdown content
            file_response = get_supabase_client().table("files").select(
                "*, markdown_content(content)"
            ).eq("id", file_id).eq("status", "processed").execute()
            
            if file_response.data and file_response.data[0].get("markdown_content"):
                file_data = file_response.data[0]
                markdown_content = file_data["markdown_content"][0]["content"]
                
                processed_files.append({
                    'file_id': file_data['id'],
                    'filename': file_data['original_filename'],
                    'extracted_text': markdown_content,
                    'metadata': file_data.get('extracted_metadata', {}),
                })
        
        if not processed_files:
            raise RuntimeError("No processed files with content found")
        
        # Update progress
        db_manager.update_job_step(job_id, "generate_template", "processing", 40)
        
        # Reorder files to put priority file first
        if priority_template_id:
            priority_file = next((f for f in processed_files if f['file_id'] == priority_template_id), None)
            if priority_file:
                processed_files.remove(priority_file)
                processed_files.insert(0, priority_file)
        
        # Limit number of files
        if len(processed_files) > 5:
            logger.warning(f"Too many files ({len(processed_files)}), limiting to 5")
            processed_files = processed_files[:5]
        
        # Generate template
        if len(processed_files) == 1:
            template_content = template_generator.generate_initial_template(
                processed_files[0]['extracted_text']
            )
        else:
            # Generate from first file
            first_contract = processed_files[0]
            template_content = template_generator.generate_initial_template(
                first_contract['extracted_text']
            )
            
            # Update with additional files
            db_manager.update_job_step(job_id, "generate_template", "processing", 60)
            
            for i, contract in enumerate(processed_files[1:], 1):
                template_content = template_generator.update_template(
                    template_content, contract['extracted_text']
                )
                progress = 60 + (i * 15 // len(processed_files[1:]))
                db_manager.update_job_step(job_id, "generate_template", "processing", progress)
        
        # Add drafting notes
        db_manager.update_job_step(job_id, "generate_template", "processing", 80)
        final_template = template_generator.add_drafting_notes(
            template_content
        )
        
        logger.info(f"Template generation completed for job {job_id}")
        return final_template
        
    except Exception as e:
        logger.error(f"Template generation failed: {str(e)}")
        raise RuntimeError(f"AI template generation failed: {str(e)}") from e

def _update_template_with_files(template_id: str, file_ids: List[str], job_id: str) -> str:
    """Update template content with new files using AI (unchanged logic)"""
    try:
        logger.info(f"Updating template {template_id} with {len(file_ids)} files")
        
        # Get existing template
        template_response = get_supabase_client().table("templates").select(
            "id, name, content, content_json"
        ).eq("id", template_id).single().execute()
        
        if not template_response.data:
            raise RuntimeError(f"Template {template_id} not found")
        
        template_data = template_response.data
        existing_content = template_data.get("content", "")
        
        if not existing_content:
            raise RuntimeError("Template has no content to update")
        
        # Get processed files with markdown content
        processed_files = []
        for file_id in file_ids:
            file_response = get_supabase_client().table("files").select(
                "*, markdown_content(content)"
            ).eq("id", file_id).eq("status", "processed").execute()
            
            if file_response.data and file_response.data[0].get("markdown_content"):
                file_data = file_response.data[0]
                markdown_content = file_data["markdown_content"][0]["content"]
                
                processed_files.append({
                    'file_id': file_data['id'],
                    'filename': file_data['original_filename'],
                    'extracted_text': markdown_content,
                    'metadata': file_data.get('extracted_metadata', {}),
                })
        
        if not processed_files:
            raise RuntimeError("No processed files with content found for update")
        
        # Use TemplateGenerator to update template
        from core.api_config import APIConfiguration
        from core.template_generator import TemplateGenerator
        
        api_config = APIConfiguration()
        template_generator = TemplateGenerator(api_config)
        
        # Update template with each file
        updated_content = existing_content
        
        for i, file_data in enumerate(processed_files):
            logger.info(f"Updating template with file {i+1}/{len(processed_files)}: {file_data['filename']}")
            
            # Update progress
            progress = 20 + (i * 60 // len(processed_files))
            db_manager.update_job_step(job_id, "update_template", "processing", progress)
            
            updated_content = template_generator.update_template_new_files(
                updated_content, file_data['extracted_text']
            )
        
        logger.info(f"Template update completed. New content length: {len(updated_content)}")
        return updated_content
        
    except Exception as e:
        logger.error(f"Template update failed: {str(e)}")
        raise RuntimeError(f"AI template update failed: {str(e)}") from e

def _save_updated_template_with_clauses(
    template_id: str, 
    updated_content: str, 
    file_ids: List[str], 
    job_id: str
) -> str:
    """Save updated template and extract clauses for content_json (unchanged logic)"""
    try:
        client = get_supabase_client()
        
        # Get source filenames
        files_response = client.table("files").select("original_filename").in_("id", file_ids).execute()
        source_files = [f["original_filename"] for f in files_response.data] if files_response.data else []
        
        # Extract clauses from the updated template content
        content_json = None
        
        try:
            from core.template_generator import TemplateGenerator
            from core.api_config import APIConfiguration
            
            api_config = APIConfiguration()
            
            if api_config.is_configured():
                template_generator = TemplateGenerator(api_config)
                
                logger.info(f"Extracting clauses from updated template (length: {len(updated_content)} chars)")
                
                # Update progress
                db_manager.update_job_step(job_id, "extract_clauses", "processing", 70)
                
                # Extract clauses used in the updated template
                content_json = template_generator.add_drafting_notes(updated_content)
                
                logger.info(f"✅ Successfully added drafting notes to updated template")
            else:
                logger.warning("API not configured for clause extraction, skipping content_json update")
                
        except Exception as e:
            logger.error(f"❌ Failed to extract clauses for updated template: {str(e)}", exc_info=True)
            # Continue without updating content_json
        
        # Prepare update data
        update_data = {
            "content": updated_content,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Add content_json if clauses were extracted
        if content_json:
            update_data["content_json"] = content_json
        
        # Update formatting_data to track the update
        formatting_data_response = client.table("templates").select("formatting_data").eq("id", template_id).single().execute()
        existing_formatting = formatting_data_response.data.get("formatting_data", {}) if formatting_data_response.data else {}
        
        # Update formatting data
        updated_formatting = {
            **existing_formatting,
            "last_updated": datetime.utcnow().isoformat(),
            "update_method": "template_update_celery_enhanced",
            "updated_with_files": source_files,
            "update_history": existing_formatting.get("update_history", []) + [{
                "updated_at": datetime.utcnow().isoformat(),
                "files_added": source_files,
                "drafting_notes_added": len(content_json) if content_json else 0
            }]
        }
        
        update_data["formatting_data"] = updated_formatting
        
        # Save updated template
        template_result = client.table("templates").update(update_data).eq("id", template_id).execute()
        
        if not template_result.data:
            raise RuntimeError("Failed to save updated template to database")
        
        # Track template usage
        try:
            usage_data = {
                "template_id": template_id,
                "action_type": "updated",
                "metadata": {
                    "updated_files": len(file_ids),
                    "update_method": "template_update_celery_enhanced",
                    "drafting_notes_added": len(content_json) if content_json else 0
                }
            }
            client.table("template_usage_stats").insert(usage_data).execute()
        except Exception as e:
            logger.warning(f"Failed to track template usage: {e}")
        
        logger.info(f"Template {template_id} updated successfully with {len(file_ids)} files and {len(content_json) if content_json else 0} drafting notes")
        return template_id
    
    except Exception as e:
        logger.error(f"Failed to save updated template: {e}")
        raise RuntimeError(f"Failed to save updated template: {str(e)}") from e

# ============================================================================
# JOB MANAGEMENT FUNCTIONS (ENHANCED)
# ============================================================================

def create_file_processing_job(file_id: str, user_id: str) -> str:
    """Create a file processing job with all steps"""
    if not file_id or not user_id:
        raise ValueError("file_id and user_id are required")
    
    try:
        # Create the main job
        job_data = {
            "user_id": user_id,
            "job_type": "file_processing",
            "status": "pending",
            "total_steps": 3,
            "current_step": 0,
            "metadata": {"file_id": file_id},
            "created_at": datetime.utcnow().isoformat()
        }
        
        job_id = db_manager.create_job(job_data)
        if not job_id:
            raise RuntimeError("Failed to create job")
        
        # Create job steps
        steps = [
            {"step_name": "extract_markdown", "step_order": 1},
            {"step_name": "extract_metadata", "step_order": 2},
            {"step_name": "extract_clauses", "step_order": 3}
        ]
        
        for step in steps:
            step_data = {
                "job_id": job_id,
                "step_name": step["step_name"],
                "step_order": step["step_order"],
                "status": "pending",
                "created_at": datetime.utcnow().isoformat()
            }
            db_manager.create_job_step(step_data)
        
        logger.info(f"Created file processing job {job_id} for file {file_id}")
        return job_id
        
    except Exception as e:
        logger.error(f"Failed to create file processing job for {file_id}: {e}")
        raise

def start_file_processing_pipeline(file_id: str, user_id: str = None) -> Dict[str, str]:
    """Start the complete file processing pipeline with job tracking"""
    if not file_id:
        raise ValueError("file_id is required")
    
    logger.info(f"Starting enhanced file processing pipeline for: {file_id}")
    
    # Get user_id if not provided
    if not user_id:
        file_record = db_manager.get_file_record(file_id)
        if not file_record:
            raise ValueError(f"File {file_id} not found")
        user_id = file_record["user_id"]
    
    # Create job with steps
    job_id = create_file_processing_job(file_id, user_id)
    
    # Create a chain of tasks with enhanced error handling
    processing_chain = chain(
        extract_markdown_task.s(file_id, job_id),
        extract_metadata_task.si(file_id, job_id),
        extract_clauses_task.si(file_id, job_id)
    )
    
    # Execute the chain with options
    result = processing_chain.apply_async(
        link_error=handle_chain_error.s(job_id),
        retry=True,
        retry_policy={
            'max_retries': 3,
            'interval_start': 0,
            'interval_step': 0.2,
            'interval_max': 0.2,
        }
    )
    
    # Update job with celery task ID
    db_manager.update_job_with_task_id(job_id, result.id)
    
    logger.info(f"Started enhanced processing chain for file {file_id} with job ID: {job_id}, chain ID: {result.id}")
    
    return {"job_id": job_id, "chain_id": result.id}

@celery_app.task(bind=True)
def handle_chain_error(self, task_id, err, traceback, job_id):
    """Handle errors in task chains"""
    logger.error(f"Task chain error for job {job_id}: {err}")
    try:
        # Update job status to failed
        get_supabase_client().table("jobs").update({
            "status": "failed",
            "error_message": str(err),
            "completed_at": datetime.utcnow().isoformat()
        }).eq("id", job_id).execute()
    except Exception as e:
        logger.error(f"Failed to update job status for failed chain: {e}")

@celery_app.task(base=BaseTaskWithRetry)
def batch_process_files_task(file_ids: List[str], user_id: str):
    """Process multiple files in batch with enhanced error handling"""
    if not file_ids or not user_id:
        return {"status": "error", "message": "file_ids and user_id are required"}
    
    try:
        logger.info(f"Starting enhanced batch processing for {len(file_ids)} files")
        
        results = []
        for file_id in file_ids:
            try:
                job_result = start_file_processing_pipeline(file_id, user_id)
                results.append({
                    "file_id": file_id,
                    "job_id": job_result["job_id"],
                    "chain_id": job_result["chain_id"],
                    "status": "started"
                })
            except Exception as e:
                logger.error(f"Failed to start processing for file {file_id}: {e}")
                results.append({
                    "file_id": file_id,
                    "status": "failed",
                    "error": str(e)
                })
        
        return {
            "status": "batch_started",
            "total_files": len(file_ids),
            "successful_starts": len([r for r in results if r["status"] == "started"]),
            "failed_starts": len([r for r in results if r["status"] == "failed"]),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        return {"status": "error", "message": str(e)}

# ============================================================================
# MONITORING AND HEALTH CHECK TASKS
# ============================================================================

@celery_app.task
def health_check():
    """Enhanced health check task to verify Celery and dependencies"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'worker': 'celery_tasks_enhanced',
        'checks': {}
    }
    
    # Test database connection
    try:
        result = get_supabase_client().table("files").select("id").limit(1).execute()
        health_status['checks']['database'] = 'connected' if result else 'error'
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status['checks']['database'] = f'error: {str(e)}'
        health_status['status'] = 'unhealthy'
    
    # Test Redis/broker connection
    try:
        i = celery_app.control.inspect()
        stats = i.stats()
        health_status['checks']['broker'] = 'connected' if stats else 'disconnected'
    except Exception as e:
        logger.error(f"Broker health check failed: {e}")
        health_status['checks']['broker'] = f'error: {str(e)}'
        health_status['status'] = 'unhealthy'
    
    # Test external API availability (without making actual calls)
    health_status['checks']['mistral_api_configured'] = bool(MISTRAL_API_KEY)
    
    # Test document processor initialization
    try:
        get_document_processor()
        health_status['checks']['document_processor'] = 'available'
    except Exception as e:
        health_status['checks']['document_processor'] = f'error: {str(e)}'
        health_status['status'] = 'degraded'
    
    return health_status

@celery_app.task
def cleanup_stale_jobs():
    """Clean up stale jobs that have been running too long"""
    try:
        # Find jobs that have been processing for more than 4 hours
        cutoff_time = (datetime.utcnow() - timedelta(hours=4)).isoformat()
        
        stale_jobs = get_supabase_client().table("jobs").select("*").eq("status", "processing").lt("started_at", cutoff_time).execute()
        
        cleanup_count = 0
        if stale_jobs.data:
            for job in stale_jobs.data:
                try:
                    # Mark as failed
                    get_supabase_client().table("jobs").update({
                        "status": "failed",
                        "error_message": "Job timed out - exceeded maximum processing time",
                        "completed_at": datetime.utcnow().isoformat()
                    }).eq("id", job["id"]).execute()
                    
                    # Try to revoke the Celery task if possible
                    if job.get("celery_task_id"):
                        celery_app.control.revoke(job["celery_task_id"], terminate=True)
                    
                    cleanup_count += 1
                    logger.info(f"Cleaned up stale job: {job['id']}")
                except Exception as e:
                    logger.error(f"Failed to cleanup job {job['id']}: {e}")
        
        logger.info(f"Cleaned up {cleanup_count} stale jobs")
        return {"cleaned_up": cleanup_count, "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Job cleanup task failed: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}

# ============================================================================
# UTILITY FUNCTIONS (ENHANCED)
# ============================================================================

def test_atomic_job_progress(job_id: str, step_name: str = "test_step", status: str = "processing") -> Dict[str, Any]:
    """Test the atomic job progress update function (for debugging)"""
    try:
        # Test the atomic update
        db_manager._update_job_progress(job_id, step_name, status)
        
        # Get the updated status
        job_status = db_manager.get_job_status(job_id)
        
        return {
            "status": "success",
            "job_status": job_status,
            "message": f"Successfully tested atomic update for job {job_id}"
        }
    except Exception as e:
        logger.error(f"Error testing atomic job progress: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": f"Failed to test atomic update for job {job_id}"
        }

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a Celery task with enhanced information"""
    if not task_id:
        return {'state': 'ERROR', 'error': 'task_id is required'}
    
    try:
        result = celery_app.AsyncResult(task_id)
        
        response = {
            'task_id': task_id,
            'state': result.state,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if result.state == 'PENDING':
            response.update({
                'status': 'Task is waiting to be processed',
                'progress': 0
            })
        elif result.state == 'PROGRESS':
            response.update({
                'current': result.info.get('current', 0),
                'total': result.info.get('total', 1),
                'status': result.info.get('status', 'Processing...'),
                'progress': result.info.get('progress', 0)
            })
        elif result.state == 'SUCCESS':
            response.update({
                'result': result.result,
                'status': 'Task completed successfully',
                'progress': 100
            })
        elif result.state == 'RETRY':
            response.update({
                'status': f'Task is being retried: {result.info}',
                'retry_count': getattr(result.info, 'retries', 0) if hasattr(result.info, 'retries') else 0
            })
        else:  # FAILURE
            response.update({
                'error': str(result.info),
                'status': 'Task failed',
                'traceback': getattr(result, 'traceback', None)
            })
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {e}")
        return {
            'state': 'ERROR',
            'error': str(e),
            'status': 'Error getting task status',
            'timestamp': datetime.utcnow().isoformat()
        }

def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get comprehensive job status including steps"""
    return db_manager.get_job_status(job_id)

def cancel_task(task_id: str) -> Dict[str, str]:
    """Cancel a running Celery task with enhanced feedback"""
    if not task_id:
        return {"status": "error", "message": "task_id is required"}
    
    try:
        # Revoke the task
        celery_app.control.revoke(task_id, terminate=True)
        
        # Try to get the task result to check if it was actually cancelled
        result = celery_app.AsyncResult(task_id)
        
        return {
            "status": "success",
            "message": f"Task {task_id} has been cancelled",
            "task_state": result.state,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        return {
            "status": "error",
            "message": f"Error cancelling task {task_id}: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }

# ============================================================================
# PERIODIC TASKS
# ============================================================================

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Set up periodic tasks for maintenance"""
    
    # Health check every 5 minutes
    sender.add_periodic_task(
        300.0,  # 5 minutes
        health_check.s(),
        name='health_check_every_5_minutes'
    )
    
    # Cleanup stale jobs every hour
    sender.add_periodic_task(
        3600.0,  # 1 hour
        cleanup_stale_jobs.s(),
        name='cleanup_stale_jobs_hourly'
    )

# ============================================================================
# CELERY SIGNAL HANDLERS
# ============================================================================

@celery_app.signals.task_prerun.connect
def task_prerun_handler(signal, sender, task_id, task, args, kwargs, **kwds):
    """Log when tasks start"""
    logger.info(f"Task {task.name} [{task_id}] starting with args: {args}, kwargs: {kwargs}")

@celery_app.signals.task_postrun.connect
def task_postrun_handler(signal, sender, task_id, task, args, kwargs, retval, state, **kwds):
    """Log when tasks complete"""
    logger.info(f"Task {task.name} [{task_id}] completed with state: {state}")

@celery_app.signals.task_failure.connect
def task_failure_handler(signal, sender, task_id, exception, traceback, einfo, **kwds):
    """Log when tasks fail"""
    logger.error(f"Task {sender.name} [{task_id}] failed: {exception}")

# Module initialization logging
logger.info("Enhanced Celery tasks module loaded successfully")
logger.info(f"Broker: {BROKER_URL}")
logger.info(f"Result Backend: {RESULT_BACKEND}")