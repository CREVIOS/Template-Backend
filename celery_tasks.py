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

# Use sync Supabase client for Celery tasks
from supabase import create_client, Client

# Configuration
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY', 'jhJDPTCJ5ZsDd9lez0jxMQRBs5Qc1UKH')
SUPABASE_URL = os.environ.get('SUPABASE_URL', 'https://knqkunivquuuvnfwrqrn.supabase.co')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtucWt1bml2cXV1dXZuZndycXJuIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0OTY5NjY2NSwiZXhwIjoyMDY1MjcyNjY1fQ.axhQBEv4lAnxmqkIDIKT8O72QwX6ypFk04do5eAPKdw')

# Path configuration
sys.path.insert(0, os.path.dirname(__file__))
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

parent_root = project_root.parent
if str(parent_root) not in sys.path:
    sys.path.insert(0, str(parent_root))

os.makedirs("logs", exist_ok=True)

# Celery configuration
celery_app = Celery(
    'celery_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0',
    include=['celery_tasks']
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=900,  # 15 minutes
    task_time_limit=1200,      # 20 minutes
    task_default_retry_delay=60,
    task_max_retries=3,
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
)

logger.add("logs/celery_tasks.log", rotation="100 MB", level="DEBUG")

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
    """Centralized database operations with proper error handling"""
    
    def __init__(self):
        self.client = get_supabase_client()
    
    def update_file_status(self, file_id: str, status: str, error_message: Optional[str] = None) -> bool:
        """Update file status"""
        if not file_id or not status:
            logger.error("file_id and status are required")
            return False
        
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if error_message:
                update_data["error_message"] = error_message if status == "error" else None
            
            if status in ["markdown_ready", "metadata_ready", "processed"]:
                update_data["processed_at"] = datetime.utcnow().isoformat()
            
            result = self.client.table("files").update(update_data).eq("id", file_id).execute()
            logger.info(f"Updated file {file_id} status to '{status}'")
            return bool(result.data)
            
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
        """Update job step"""
        if not job_id or not step_name or not status:
            logger.error("job_id, step_name, and status are required")
            return False
        
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
            self.client.table("job_steps").update(update_data).eq("job_id", job_id).eq("step_name", step_name).execute()
            
            # Update the main job
            self._update_job_progress(job_id, step_name, status, error_message)
            
            logger.info(f"Updated job {job_id} step '{step_name}' to status '{status}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update job step for {job_id}.{step_name}: {e}")
            return False
    
    def _update_job_progress(self, job_id: str, step_name: str, status: str, error_message: Optional[str] = None):
        """Update main job progress based on steps"""
        try:
            job_update = {
                "current_step_name": step_name,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            if status == "processing" and "started_at" not in job_update:
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
            
        except Exception as e:
            logger.error(f"Failed to update job progress for {job_id}: {e}")
    
    def get_file_record(self, file_id: str) -> Optional[Dict]:
        """Get file record"""
        if not file_id:
            logger.error("file_id is required")
            return None
        
        try:
            result = self.client.table("files").select("*").eq("id", file_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get file record {file_id}: {e}")
            return None
    
    def save_markdown_content(self, file_id: str, user_id: str, content: str) -> bool:
        """Save markdown content"""
        if not all([file_id, user_id, content]):
            logger.error("file_id, user_id, and content are required")
            return False
        
        try:
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
        except Exception as e:
            logger.error(f"Failed to save markdown content for {file_id}: {e}")
            return False
    
    def get_markdown_content(self, file_id: str) -> Optional[str]:
        """Get markdown content for file"""
        if not file_id:
            return None
        
        try:
            result = self.client.table("markdown_content").select("content").eq("file_id", file_id).execute()
            return result.data[0]["content"] if result.data else None
        except Exception as e:
            logger.error(f"Failed to get markdown content for {file_id}: {e}")
            return None
    
    def save_clause_to_library(self, clause_data: Dict) -> bool:
        """Save clause to clause_library"""
        try:
            result = self.client.table("clause_library").insert(clause_data).execute()
            return bool(result.data)
        except Exception as e:
            logger.error(f"Failed to save clause: {e}")
            return False
    
    def create_job(self, job_data: Dict) -> Optional[str]:
        """Create job and return job_id"""
        try:
            result = self.client.table("jobs").insert(job_data).execute()
            return result.data[0]["id"] if result.data else None
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            return None
    
    def create_job_step(self, step_data: Dict) -> bool:
        """Create job step"""
        try:
            result = self.client.table("job_steps").insert(step_data).execute()
            return bool(result.data)
        except Exception as e:
            logger.error(f"Failed to create job step: {e}")
            return False
    
    def update_job_with_task_id(self, job_id: str, task_id: str) -> bool:
        """Update job with Celery task ID"""
        try:
            result = self.client.table("jobs").update({
                "celery_task_id": task_id,
                "status": "processing",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", job_id).execute()
            return bool(result.data)
        except Exception as e:
            logger.error(f"Failed to update job {job_id} with task_id: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get comprehensive job status including steps"""
        if not job_id:
            return {"error": "job_id is required"}
        
        try:
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
            
        except Exception as e:
            logger.error(f"Error getting job status for {job_id}: {e}")
            return {"error": f"Error retrieving job status: {str(e)}"}

# Global database manager instance
db_manager = DatabaseManager()

# ============================================================================
# DOCUMENT PROCESSING UTILITIES
# ============================================================================

def get_document_processor():
    """Get document processor instance"""
    try:
        from core.api_config import APIConfiguration
        from core.document_processor import DocumentProcessor
        
        api_config = APIConfiguration()
        return DocumentProcessor(api_config)
    except Exception as e:
        logger.error(f"Failed to initialize document processor: {e}")
        raise

def _extract_content_from_ocr_response(ocr_data: Dict) -> str:
    """Extract text content from OCR response"""
    markdown_content = ""
    
    if not isinstance(ocr_data, dict):
        return str(ocr_data).strip()
    
    # Check various response formats
    if 'pages' in ocr_data and isinstance(ocr_data['pages'], list):
        logger.info(f"🔍 Found pages array with {len(ocr_data['pages'])} pages")
        for page in ocr_data['pages']:
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
    
    # Check for direct content fields
    elif any(key in ocr_data for key in ['content', 'text', 'markdown', 'extracted_text', 'ocr_text']):
        markdown_content = (
            ocr_data.get('content') or 
            ocr_data.get('text') or 
            ocr_data.get('markdown') or 
            ocr_data.get('extracted_text') or
            ocr_data.get('ocr_text') or ''
        )
    
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
    
    return str(markdown_content).strip()

def _process_with_mistral_ocr(download_url: str, file_record: Dict, job_id: Optional[str] = None) -> str:
    """Process document with Mistral OCR API"""
    try:
        logger.info(f"🤖 Starting Mistral OCR processing...")
        
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistral-ocr-latest",
            "id": f"ocr-{int(time.time())}-{file_record.get('id')}",
            "document": {
                "document_url": download_url,
                "document_name": file_record.get("original_filename", "document"),
                "type": "document_url"
            },
            "include_image_base64": False,
            "image_limit": 0,
            "image_min_size": 0
        }
        
        # Make request with proper timeout
        ocr_response = requests.post(
            "https://api.mistral.ai/v1/ocr",
            headers=headers,
            json=payload,
            timeout=240  # 4 minutes timeout
        )
        
        logger.info(f"📊 Mistral OCR API response status: {ocr_response.status_code}")
        
        if ocr_response.status_code != 200:
            error_message = f"Mistral OCR API error: {ocr_response.text}"
            logger.error(error_message)
            raise requests.RequestException(error_message)
        
        # Update progress
        if job_id:
            db_manager.update_job_step(job_id, "extract_markdown", "processing", 70)
        
        # Parse and extract content
        ocr_data = ocr_response.json()
        markdown_content = _extract_content_from_ocr_response(ocr_data)
        
        if not markdown_content:
            error_message = f"No content extracted from document"
            raise ValueError(error_message)
        
        logger.info(f"✅ OCR completed, generated {len(markdown_content)} characters")
        return markdown_content
        
    except requests.RequestException as e:
        error_message = f"Mistral OCR API request failed: {str(e)}"
        logger.error(error_message)
        raise ConnectionError(error_message) from e

# ============================================================================
# MAIN CELERY TASKS
# ============================================================================

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def extract_markdown_task(self, file_id: str, job_id: Optional[str] = None):
    """Extract markdown content from document using Mistral OCR API"""
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
        
        # Check if already processed
        if file_record.get("status") in ["markdown_ready", "processed"]:
            logger.info(f"File {file_id} already processed, skipping")
            return {"status": "already_processed", "file_id": file_id}
        
        # Update status
        db_manager.update_file_status(file_id, "converting", "Converting document to markdown")
        
        # Get signed URL
        try:
            signed_url_response = get_supabase_client().storage.from_("documents").create_signed_url(
                file_record["storage_path"], 
                expires_in=3600
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
        
        # Process with Mistral OCR
        markdown_content = _process_with_mistral_ocr(download_url, file_record, job_id)
        
        # Save markdown content
        if not db_manager.save_markdown_content(
            file_id, 
            file_record["user_id"], 
            markdown_content
        ):
            raise RuntimeError("Failed to save markdown content")
        
        # Update file status
        db_manager.update_file_status(file_id, "markdown_ready")
        
        if job_id:
            db_manager.update_job_step(job_id, "extract_markdown", "completed", 100, metadata={
                "word_count": len(markdown_content.split()),
                "content_length": len(markdown_content)
            })
        
        logger.info(f"✅ Markdown extraction completed successfully for file: {file_id}")
        return {
            "status": "success", 
            "word_count": len(markdown_content.split()), 
            "content_length": len(markdown_content),
            "file_id": file_id
        }
        
    except (ValueError, ConnectionError) as e:
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
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying markdown extraction for file {file_id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60, exc=e)
        
        raise RuntimeError(f"Markdown extraction failed after {self.max_retries} retries: {str(e)}") from e

@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def extract_metadata_task(self, file_id: str, job_id: Optional[str] = None):
    """Extract metadata from markdown content using AI"""
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
        
        # Check if already processed
        if file_record.get("status") in ["metadata_ready", "processed"]:
            logger.info(f"File {file_id} metadata already processed, skipping")
            return {"status": "already_processed", "file_id": file_id}
        
        # Get markdown content
        markdown_content = db_manager.get_markdown_content(file_id)
        if not markdown_content:
            raise ValueError("No markdown content found for metadata extraction")
        
        if job_id:
            db_manager.update_job_step(job_id, "extract_metadata", "processing", 50)
        
        # Extract metadata using AI
        processor = get_document_processor()
        metadata = processor.extract_metadata_from_text(
            markdown_content, 
            file_record.get("original_filename", "unknown")
        )
        
        # Save metadata
        result = get_supabase_client().table("files").update({
            "extracted_metadata": metadata,
            "status": "metadata_ready",
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", file_id).execute()
        
        if not result.data:
            raise RuntimeError("Failed to save metadata")
        
        if job_id:
            db_manager.update_job_step(job_id, "extract_metadata", "completed", 100, metadata={
                "extracted_metadata": metadata
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
        
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying metadata extraction for file {file_id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=30, exc=e)
        
        raise RuntimeError(f"Metadata extraction failed after {self.max_retries} retries: {str(e)}") from e

@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def extract_clauses_task(self, file_id: str, job_id: Optional[str] = None):
    """Extract legal clauses from markdown content and save to clause_library"""
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
        
        # Check if already processed
        if file_record.get("status") == "processed":
            logger.info(f"File {file_id} clauses already processed, skipping")
            return {"status": "already_processed", "file_id": file_id}
        
        metadata = file_record.get("extracted_metadata", {})
        markdown_content = db_manager.get_markdown_content(file_id)
        
        if not markdown_content:
            raise ValueError("No markdown content found for clause extraction")
        
        if job_id:
            db_manager.update_job_step(job_id, "extract_clauses", "processing", 30)
        
        # Extract clauses using AI
        try:
            processor = get_document_processor()
            clauses = processor.extract_clauses_from_text(markdown_content, metadata)
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
                        "clause_hash": clause_hash
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
                "clauses_saved": clauses_saved
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
        
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying clause extraction for file {file_id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=30, exc=e)
        
        raise RuntimeError(f"Clause extraction failed after {self.max_retries} retries: {str(e)}") from e

# ============================================================================
# TEMPLATE GENERATION TASK
# ============================================================================

@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def template_generation_task(self, job_id: str):
    """Celery task to handle template generation in the background"""
    if not job_id:
        raise ValueError("job_id is required")
    
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
        
        # Step 1: Ensure all files are processed
        db_manager.update_job_step(job_id, "process_files", "processing", 10)
        _ensure_files_processed(file_ids, job_id)
        
        # Step 2: Wait for all files to be processed
        processed_count = _wait_for_files_processed(file_ids, job_id)
        db_manager.update_job_step(job_id, "process_files", "completed", 100, metadata={
            "processed_files": processed_count,
            "total_files": len(file_ids)
        })
        
        # Step 3: Generate template
        db_manager.update_job_step(job_id, "generate_template", "processing", 20)
        template_content = _generate_template_content(
            folder_id, priority_template_id, file_ids, job_id
        )
        db_manager.update_job_step(job_id, "generate_template", "completed", 100, metadata={
            "template_length": len(template_content)
        })
        
        # Step 4: Save template with clause extraction
        db_manager.update_job_step(job_id, "save_template", "processing", 50)
        template_id = _save_template_to_database(
            user_id, folder_id, priority_template_id, template_name, template_content, file_ids
        )
        db_manager.update_job_step(job_id, "save_template", "completed", 100, metadata={
            "template_id": template_id
        })
        
        # Update job result
        get_supabase_client().table("jobs").update({
            "result": {
                "template_id": template_id,
                "template_name": template_name,
                "processed_files": processed_count
            },
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", job_id).execute()
        
        logger.info(f"Template generation task {job_id} completed successfully")
        
        return {
            "job_id": job_id,
            "template_id": template_id,
            "status": "completed",
            "processed_files": processed_count
        }
        
    except ValueError as e:
        logger.error(f"❌ Validation error in template generation for {job_id}: {str(e)}")
        raise
        
    except Exception as e:
        logger.error(f"Template generation task {job_id} failed: {str(e)}")
        
        # Update job status
        try:
            current_step_name = job_status.get("current_step_name", "unknown")
            db_manager.update_job_step(job_id, current_step_name, "failed", 0, str(e))
        except:
            pass
        
        # Retry logic
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying template generation for job: {job_id}")
            raise self.retry(countdown=60, exc=e)
        
        raise

# ============================================================================
# TEMPLATE GENERATION HELPER FUNCTIONS
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
    """Generate template content using AI"""
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
                
                # Limit content size
                max_content_length = 50000
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
            template_content, processed_files
        )
        
        logger.info(f"Template generation completed for job {job_id}")
        return final_template
        
    except Exception as e:
        logger.error(f"Template generation failed: {str(e)}")
        raise RuntimeError(f"AI template generation failed: {str(e)}") from e

def _save_template_to_database(
    user_id: str,
    folder_id: str, 
    priority_template_id: str, 
    template_name: str, 
    template_content: str,
    file_ids: List[str]
) -> str:
    """Save template to database with clause extraction"""
    try:
        client = get_supabase_client()
        
        # Get source filenames
        files_response = client.table("files").select("original_filename").in_("id", file_ids).execute()
        source_files = [f["original_filename"] for f in files_response.data] if files_response.data else []
        
        # Extract clauses from the generated template content
        content_json = None
        extracted_clauses = []
        
        try:
            from core.template_generator import TemplateGenerator
            from core.api_config import APIConfiguration
            
            api_config = APIConfiguration()
            
            if api_config.is_configured():
                template_generator = TemplateGenerator(api_config)
                
                logger.info(f"Extracting clauses from template content (length: {len(template_content)} chars)")
                
                # Extract clauses used in the template
                extracted_clauses = template_generator.extract_used_clauses(template_content)
                
                if extracted_clauses and len(extracted_clauses) > 0:
                    content_json = {
                        "clauses": extracted_clauses,
                        "extraction_metadata": {
                            "extracted_at": datetime.utcnow().isoformat(),
                            "total_clauses": len(extracted_clauses),
                            "source_files": source_files,
                            "template_name": template_name,
                            "extraction_method": "celery_background"
                        }
                    }
                    logger.info(f"✅ Successfully extracted {len(extracted_clauses)} clauses for template content_json")
                else:
                    logger.warning("❌ No clauses extracted from template content")
            else:
                logger.warning("API not configured for clause extraction, skipping content_json population")
                
        except Exception as e:
            logger.error(f"❌ Failed to extract clauses for content_json: {str(e)}", exc_info=True)
            # Continue without content_json
        
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
                "generation_method": "celery_background"
            },
            "word_compatible": True,
            "is_active": True,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Add content_json if clauses were extracted
        if content_json:
            template_data["content_json"] = content_json
        
        template_result = client.table("templates").insert(template_data).execute()
        
        if not template_result.data:
            raise RuntimeError("Failed to save template to database")
        
        template_id = template_result.data[0]["id"]
        
        # Track template usage
        try:
            usage_data = {
                "template_id": template_id,
                "user_id": user_id,
                "action_type": "generated",
                "metadata": {
                    "source_files": len(file_ids),
                    "generation_method": "celery_background",
                    "folder_id": folder_id,
                    "clauses_extracted": len(extracted_clauses) if extracted_clauses else 0
                }
            }
            client.table("template_usage_stats").insert(usage_data).execute()
        except Exception as e:
            logger.warning(f"Failed to track template usage: {e}")
        
        logger.info(f"Template {template_id} created successfully from {len(file_ids)} files with {len(extracted_clauses) if extracted_clauses else 0} clauses")
        return template_id
    
    except Exception as e:
        logger.error(f"Failed to save template: {e}")
        raise RuntimeError(f"Failed to save template: {str(e)}") from e

# ============================================================================
# JOB MANAGEMENT FUNCTIONS
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
            "metadata": {"file_id": file_id}
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
                "status": "pending"
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
    
    logger.info(f"Starting file processing pipeline for: {file_id}")
    
    # Get user_id if not provided
    if not user_id:
        file_record = db_manager.get_file_record(file_id)
        if not file_record:
            raise ValueError(f"File {file_id} not found")
        user_id = file_record["user_id"]
    
    # Create job with steps
    job_id = create_file_processing_job(file_id, user_id)
    
    # Create a chain of tasks
    processing_chain = chain(
        extract_markdown_task.si(file_id, job_id),
        extract_metadata_task.si(file_id, job_id),
        extract_clauses_task.si(file_id, job_id)
    )
    
    # Execute the chain
    result = processing_chain.apply_async()
    
    # Update job with celery task ID
    db_manager.update_job_with_task_id(job_id, result.id)
    
    logger.info(f"Started processing chain for file {file_id} with job ID: {job_id}, chain ID: {result.id}")
    
    return {"job_id": job_id, "chain_id": result.id}

@celery_app.task
def batch_process_files_task(file_ids: List[str], user_id: str):
    """Process multiple files in batch"""
    if not file_ids or not user_id:
        return {"status": "error", "message": "file_ids and user_id are required"}
    
    try:
        logger.info(f"Starting batch processing for {len(file_ids)} files")
        
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
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        return {"status": "error", "message": str(e)}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a Celery task"""
    if not task_id:
        return {'state': 'ERROR', 'error': 'task_id is required'}
    
    try:
        result = celery_app.AsyncResult(task_id)
        
        if result.state == 'PENDING':
            response = {
                'state': result.state,
                'status': 'Task is waiting to be processed'
            }
        elif result.state == 'PROGRESS':
            response = {
                'state': result.state,
                'current': result.info.get('current', 0),
                'total': result.info.get('total', 1),
                'status': result.info.get('status', 'Processing...')
            }
        elif result.state == 'SUCCESS':
            response = {
                'state': result.state,
                'result': result.result,
                'status': 'Task completed successfully'
            }
        else:  # FAILURE
            response = {
                'state': result.state,
                'error': str(result.info),
                'status': 'Task failed'
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting task status for {task_id}: {e}")
        return {
            'state': 'ERROR',
            'error': str(e),
            'status': 'Error getting task status'
        }

def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get comprehensive job status including steps"""
    return db_manager.get_job_status(job_id)

def cancel_task(task_id: str) -> str:
    """Cancel a running Celery task"""
    if not task_id:
        return "task_id is required"
    
    try:
        celery_app.control.revoke(task_id, terminate=True)
        return f"Task {task_id} has been cancelled"
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {e}")
        return f"Error cancelling task {task_id}: {str(e)}"

@celery_app.task
def health_check():
    """Health check task to verify Celery is working"""
    try:
        # Test database connection
        result = get_supabase_client().table("files").select("id").limit(1).execute()
        db_healthy = bool(result)
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_healthy = False
    
    return {
        'status': 'healthy' if db_healthy else 'unhealthy',
        'timestamp': datetime.utcnow().isoformat(),
        'worker': 'celery_tasks',
        'database': 'connected' if db_healthy else 'disconnected'
    }