from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str = "queued"
    created_at: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    stage: Optional[str] = None  # "queued", "validating", "segmenting_tissues", "locating_l3", "packaging"
    progress: Optional[float] = None  # 0.0 to 1.0
    error_code: Optional[str] = None
    detail: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    diagnostics: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str = "healthy"
    ready: bool
    contract_version: str = "2.0"
    totalsegmentator_version: str
    license_valid: bool
    tasks_available: List[str]
    gpu_available: bool
    gpu_name: Optional[str] = None
    queue_len: int
    uptime_seconds: float
    ram_total_mb: Optional[float] = None
    ram_available_mb: Optional[float] = None


class ErrorResponse(BaseModel):
    detail: str
    error_code: str


class PurgeResponse(BaseModel):
    detail: str
    job_id: str
