from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import os
import platform
import sys
from contextlib import asynccontextmanager
from core.database import initialize_database
from core.redis_cache import initialize_cache_service
from core.background_tasks import start_cache_refresh_background, stop_cache_refresh_background
from core.clause_library import router as clause_library_router
# Import routers directly 
from core.folders import router as folders_router
from core.files import router as files_router
from core.template import router as template_router
from core.api_config import APIConfiguration

# Configure logging
logger.add(
    "logs/api.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time} {level} {message}",
    backtrace=True,
    diagnose=True
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Unified application startup / shutdown."""

    # ---------------------- STARTUP ----------------------
    logger.info("🚀 Starting Legal Template Generator API")

    # 1. Config object (make available via dependency)
    api_config = APIConfiguration()
    app.state.api_config = api_config  # optional global access

    try:
        # 2. External services
        await initialize_database()
        await initialize_cache_service()
        await start_cache_refresh_background()
        logger.info("✅ All services initialized")
    except Exception as e:
        logger.error(f"❌ Failed during startup: {e}")
        raise

    # Give control back to FastAPI
    try:
        yield
    finally:
        # -------------------- SHUTDOWN --------------------
        logger.info("🛑 Shutting down API …")
        try:
            stop_cache_refresh_background()
            # Close Supabase clients gracefully
            from core.database import get_database_service
            db = get_database_service()
            await db.close()
            logger.info("✅ External connections closed")
        except Exception as e:
            logger.warning(f"⚠️  Shutdown issue: {e}")

# Create FastAPI app
app = FastAPI(
    title="Legal Template Generator API",
    description="AI-powered legal document template generation system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware with explicit configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000", 
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "*"  # Allow all origins for development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Content-Type",
        "Authorization", 
        "Accept",
        "Origin",
        "X-Requested-With",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
        "*"
    ],
    expose_headers=["*"],
)

# Dependency to get API configuration
def get_api_config() -> APIConfiguration:
    if hasattr(app.state, 'api_config'):
        return app.state.api_config
    raise HTTPException(status_code=500, detail="API configuration not initialized")

# Include routers with correct paths
app.include_router(
    folders_router,
    prefix="/api/folders",
    tags=["folders"]
)

app.include_router(
    files_router,
    prefix="/api/files",
    tags=["files"]
)

app.include_router(
    template_router,
    prefix="/api/templates",
    tags=["templates"]
)

app.include_router(
    clause_library_router,
    prefix="/api/clause-library",
    tags=["clause-library"]
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Legal Template Generator API",
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Legal Template Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "workflow": {
            "1": "POST /api/files/upload - Upload files",
            "2": "POST /api/folders/ - Create folder", 
            "3": "Background processing - Extract text, metadata, clauses",
            "4": "POST /api/templates/generate - Generate template",
            "5": "AI processing - Create template with drafting notes",
            "6": "Template saved to database",
            "7": "GET /api/templates/{id}/export/{format} - Export template"
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception handler caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc) if hasattr(app, 'debug') and app.debug else "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    try:
        if "--with-gunicorn" in sys.argv and platform.system() != "Windows":
            # Use Gunicorn on non-Windows platforms
            import subprocess
            subprocess.run([
                "gunicorn", 
                "-w", "4",  # Number of workers
                "-k", "uvicorn.workers.UvicornWorker",  # Worker class
                "--reload",  # Enable hot reloading
                "-b", "0.0.0.0:8000", 
                "main:app"
            ])
        else:
            import uvicorn
            logger.info("Starting with Uvicorn" + 
                        (" (Gunicorn not supported on Windows)" if platform.system() == "Windows" and "--with-gunicorn" in sys.argv else ""))
            uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError as e:
        logger.error(f"Server startup error: {str(e)}")