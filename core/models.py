from pydantic import BaseModel, Field, UUID4
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from uuid import UUID

# Folder models
class FolderBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    color: Optional[str] = None

class FolderCreate(FolderBase):
    pass

class FolderUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    color: Optional[str] = None

class Folder(FolderBase):
    id: UUID4
    user_id: UUID4
    created_at: datetime
    updated_at: datetime

class FolderWithCount(Folder):
    file_count: int

# File models
class FileBase(BaseModel):
    original_filename: str
    file_size: int
    file_type: str
    folder_id: UUID4

class FileCreate(FileBase):
    pass

class File(FileBase):
    id: UUID4
    user_id: UUID4
    storage_path: Optional[str] = None
    storage_url: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None

# Statistics model
class FolderStats(BaseModel):
    totalFolders: int
    totalFiles: int
    totalProcessed: int
    totalPending: int
    totalErrors: int

# Response models
class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

class FoldersResponse(BaseModel):
    folders: List[FolderWithCount]

class FilesResponse(BaseModel):
    files: List[File]

# Template models - Updated to match exact database schema
class TemplateBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    folder_id: UUID4
    content: str
    template_type: str = Field(default="general", max_length=50)
    file_extension: str = Field(default=".docx", max_length=10)
    formatting_data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    word_compatible: bool = True
    is_active: bool = True

class TemplateCreate(TemplateBase):
    pass

class TemplateUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    content: Optional[str] = None
    template_type: Optional[str] = Field(None, max_length=50)
    file_extension: Optional[str] = Field(None, max_length=10)
    formatting_data: Optional[Dict[str, Any]] = None
    word_compatible: Optional[bool] = None
    is_active: Optional[bool] = None

class Template(TemplateBase):
    id: UUID4
    created_at: datetime
    updated_at: datetime

class TemplateWithDetails(Template):
    folder_name: str
    folder_color: Optional[str] = None
    files_count: int
    last_action_type: Optional[
        Literal['viewed', 'downloaded', 'edited', 'generated', 'exported']
    ] = None
    last_action_date: Optional[datetime] = None

class TemplatePreview(BaseModel):
    id: UUID4
    folder_id: UUID4
    name: str
    content: str
    template_type: str
    file_extension: str
    formatting_data: Optional[Dict[str, Any]] = None
    folder_name: str
    created_at: datetime
    updated_at: datetime

# Template Generation Response
class TemplateGenerationResponse(BaseModel):
    templatesCreated: int
    filesProcessed: int
    duration: int  # in milliseconds

# Template Usage models - Updated to match exact database schema
class TemplateUsageStatsBase(BaseModel):
    template_id: UUID4
    user_id: Optional[UUID4] = None  # Made optional as per schema
    action_type: str = Field(..., max_length=50)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class TemplateUsageStatsCreate(TemplateUsageStatsBase):
    pass

class TemplateUsageStats(TemplateUsageStatsBase):
    id: UUID4
    created_at: datetime

# Response models
class TemplatesResponse(BaseModel):
    templates: List[TemplateWithDetails]
    total: int

class TemplatePreviewResponse(BaseModel):
    template: TemplatePreview

# Template Statistics Summary
class TemplateStatsSummary(BaseModel):
    total_templates: int
    template_types: int
    total_views: int
    total_downloads: int
    total_generated: int
    total_exported: int
    recent_activity: List[Dict[str, Any]]

# Export Request Model
class TemplateExportRequest(BaseModel):
    format: Literal['html', 'docx', 'pdf']
    include_metadata: bool = True
    include_alternatives: bool = True

# Template Search and Filter Models
class TemplateSearchRequest(BaseModel):
    query: Optional[str] = None
    template_type: Optional[str] = None
    folder_id: Optional[UUID4] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    is_active: Optional[bool] = None
    sort_by: str = Field(default="created_at", pattern="^(name|created_at|updated_at|template_type)$")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

# Bulk Operations
class TemplateBulkDeleteRequest(BaseModel):
    template_ids: List[UUID4] = Field(..., min_items=1, max_items=50)

class TemplateBulkUpdateRequest(BaseModel):
    template_ids: List[UUID4] = Field(..., min_items=1, max_items=50)
    update_data: TemplateUpdate

# Analytics Models
class TemplateAnalytics(BaseModel):
    template_id: UUID4
    template_name: str
    total_views: int
    total_downloads: int
    total_exports: int
    last_activity: Optional[datetime] = None
    activity_trend: List[Dict[str, Any]]  # Daily/weekly activity data

class UserAnalytics(BaseModel):
    user_id: UUID4
    total_templates: int
    total_folders: int
    total_files: int
    activity_summary: Dict[str, int]
    most_used_templates: List[TemplateAnalytics]
    recent_activity: List[Dict[str, Any]]

# Error Response Models
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Validation Models
class TemplateValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

# Configuration Models
class TemplateConfig(BaseModel):
    max_file_size: int = Field(default=50 * 1024 * 1024)  # 50MB
    allowed_file_types: List[str] = Field(default=['pdf', 'docx', 'doc'])
    max_templates_per_folder: int = Field(default=100)
    retention_days: int = Field(default=365)
    enable_ai_suggestions: bool = Field(default=True)
    enable_version_control: bool = Field(default=False)

# Job tracking models
class JobCreate(BaseModel):
    user_id: str
    job_type: str
    total_steps: int = 1
    metadata: Optional[Dict[str, Any]] = None

class JobUpdate(BaseModel):
    status: Optional[str] = None
    progress: Optional[int] = None
    current_step: Optional[int] = None
    current_step_name: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    celery_task_id: Optional[str] = None

class Job(BaseModel):
    id: str
    user_id: str
    job_type: str
    status: str = "pending"
    progress: int = 0
    total_steps: int = 1
    current_step: int = 0
    current_step_name: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    celery_task_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class JobStepCreate(BaseModel):
    job_id: str
    step_name: str
    step_order: int

class JobStepUpdate(BaseModel):
    status: Optional[str] = None
    progress: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class JobStep(BaseModel):
    id: str
    job_id: str
    step_name: str
    step_order: int
    status: str = "pending"
    progress: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

# Document processing models
class ExtractedMetadata(BaseModel):
    word_count: int
    character_count: int
    line_count: int
    extracted_at: str
    extraction_method: str = "basic_analysis"