from typing import Optional, Generator, List, Dict
from functools import lru_cache
from pydantic_settings import BaseSettings
from supabase import create_client, Client, acreate_client, AsyncClient
from fastapi import Depends, HTTPException
import asyncio


class DatabaseSettings(BaseSettings):
    """Database configuration using environment variables"""
    supabase_url: str = "https://dpaovmacocyatazsnvtx.supabase.co"
    supabase_anon_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRwYW92bWFjb2N5YXRhenNudnR4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTEzODU3MDMsImV4cCI6MjA2Njk2MTcwM30.ht6xRFCukUx1iiewF47l7f0qeLXOxs8yAaeI-ACuOgQ"
    supabase_service_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRwYW92bWFjb2N5YXRhenNudnR4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTM4NTcwMywiZXhwIjoyMDY2OTYxNzAzfQ.3_fKrWMMPCu83pHD-oxZmAyi_pemW6bJdUUuc_-Hg80"
    
    class Config:
        env_file = ".env"
        env_prefix = "DB_"
        extra = "allow"


@lru_cache()
def get_settings() -> DatabaseSettings:
    """Load and cache settings from environment"""
    return DatabaseSettings()


class DatabaseService:
    """
    Provides both anonymous and service-role AsyncClient instances.
    Use `.anon` for public operations, `.service` for elevated permissions.
    """
    def __init__(self, settings: Optional[DatabaseSettings] = None):
        self.settings = settings or get_settings()
        self._anon_client: Optional[AsyncClient] = None
        self._service_client: Optional[AsyncClient] = None
        self._initialized = False

    async def initialize(self):
        """Initialize the async clients. Call this at startup."""
        if not self._initialized:
            try:
                self._anon_client = await acreate_client(
                    self.settings.supabase_url,
                    self.settings.supabase_anon_key
                )
                self._service_client = await acreate_client(
                    self.settings.supabase_url,
                    self.settings.supabase_service_key
                )
                self._initialized = True
                print("✅ Supabase clients initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Supabase clients: {e}")
                raise

    @property
    def anon(self) -> AsyncClient:
        """AsyncClient with anon key for public queries"""
        if not self._initialized or self._anon_client is None:
            raise RuntimeError(
                "DatabaseService not initialized. Make sure to call await initialize_database() "
                "in your FastAPI startup lifespan function."
            )
        return self._anon_client

    @property
    def service(self) -> AsyncClient:
        """AsyncClient with service key for admin operations"""
        if not self._initialized or self._service_client is None:
            raise RuntimeError(
                "DatabaseService not initialized. Make sure to call await initialize_database() "
                "in your FastAPI startup lifespan function."
            )
        return self._service_client

    @property
    def client(self) -> AsyncClient:
        """Default to service-role client for backend use"""
        return self.service


# Singleton instance
db_service: Optional[DatabaseService] = None

def get_database_service() -> DatabaseService:
    """FastAPI dependency to get the DatabaseService singleton"""
    global db_service
    if db_service is None:
        db_service = DatabaseService()
    return db_service

async def initialize_database():
    """Initialize the database service. Call this at app startup."""
    db = get_database_service()
    await db.initialize()

async def get_db_service(db: DatabaseService = Depends(get_database_service)) -> DatabaseService:
    """Alias dependency for directly injecting DatabaseService"""
    return db


# ----------------------
# Async CRUD functions
# ----------------------

async def create_job(job_data: Dict, db: DatabaseService = Depends(get_database_service)) -> str:
    """Insert a new job and return its ID"""
    resp = await db.client.table("jobs").insert(job_data).execute()
    if not resp.data:
        raise HTTPException(status_code=500, detail="Failed to create job")
    return str(resp.data[0]["id"])

async def update_job(job_id: str, updates: Dict, db: DatabaseService = Depends(get_database_service)) -> bool:
    """Update a job record"""
    if not updates:
        return True
    await db.client.table("jobs").update(updates).eq("id", job_id).execute()
    return True

async def get_job(job_id: str, db: DatabaseService = Depends(get_database_service)) -> Optional[Dict]:
    """Fetch a job record"""
    resp = await db.client.table("jobs").select("*").eq("id", job_id).single().execute()
    return resp.data

async def get_jobs_by_user(
    user_id: str,
    job_type: Optional[str] = None,
    db: DatabaseService = Depends(get_database_service)
) -> List[Dict]:
    """List jobs for a user, optionally filtered by type"""
    query = db.client.table("jobs").select("*").eq("user_id", user_id)
    if job_type:
        query = query.eq("job_type", job_type)
    resp = await query.order("created_at", desc=True).execute()
    return resp.data or []

async def create_job_step(
    step_data: Dict,
    db: DatabaseService = Depends(get_database_service)
) -> str:
    """Insert a new job step and return its ID"""
    resp = await db.client.table("job_steps").insert(step_data).execute()
    if not resp.data:
        raise HTTPException(status_code=500, detail="Failed to create job step")
    return str(resp.data[0]["id"])

async def update_job_step(
    step_id: str,
    updates: Dict,
    db: DatabaseService = Depends(get_database_service)
) -> bool:
    """Update a job step record"""
    if not updates:
        return True
    await db.client.table("job_steps").update(updates).eq("id", step_id).execute()
    return True

async def get_job_steps(
    job_id: str,
    db: DatabaseService = Depends(get_database_service)
) -> List[Dict]:
    """Get all steps for a job"""
    resp = await db.client.table("job_steps").select("*").eq("job_id", job_id).order("step_order").execute()
    return resp.data or []

async def get_file_by_id(file_id: str, db: DatabaseService = Depends(get_database_service)) -> Optional[Dict]:
    """Get file details by ID"""
    resp = await db.client.table("files").select("*").eq("id", file_id).single().execute()
    return resp.data

async def update_file_status(
    file_id: str,
    status: str,
    error_message: Optional[str] = None,
    db: DatabaseService = Depends(get_database_service)
) -> bool:
    """Update file status and optional error message"""
    updates = {"status": status, "updated_at": "now()"}
    if error_message:
        updates["error_message"] = error_message
    await db.client.table("files").update(updates).eq("id", file_id).execute()
    return True

async def get_files_by_folder(
    folder_id: str,
    db: DatabaseService = Depends(get_database_service)
) -> List[Dict]:
    """Get all files in a specific folder"""
    resp = await db.client.table("files").select("*").eq("folder_id", folder_id).execute()
    return resp.data or []


