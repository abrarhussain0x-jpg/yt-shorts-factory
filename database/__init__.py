"""
database/__init__.py — Package initializer for database module.
"""

from database.db import (
    # Core initialisation
    init_db,
    session_scope,
    # Job CRUD
    create_job,
    update_job_status,
    get_job,
    list_jobs,
    cancel_job,
    # Video & Export
    save_video_record,
    save_export_record,
    # Transcription cache
    get_transcription_cache,
    save_transcription_cache,
    # Pipeline metrics
    record_pipeline_metric,
    get_pipeline_metrics,
    # Analytics
    save_analytics_snapshot,
    get_stats,
    get_daily_analytics,
    get_top_channels,
    get_processing_time_percentiles,
    # Maintenance
    cleanup_old_jobs,
    cleanup_old_cache,
    vacuum_database,
    # Advanced queries
    get_job_with_details,
    search_videos,
    get_disk_usage_summary,
    get_failure_analysis,
    # Models
    Job,
    Video,
    ExportRecord,
    TranscriptionCache,
    PipelineMetric,
    AnalyticsSnapshot,
    # Constants
    JOB_STATUS_VALUES,
    PRIORITY_VALUES,
    PLATFORM_VALUES,
)

__all__ = [
    # Core
    "init_db",
    "session_scope",
    # Job CRUD
    "create_job",
    "update_job_status",
    "get_job",
    "list_jobs",
    "cancel_job",
    # Video & Export
    "save_video_record",
    "save_export_record",
    # Transcription cache
    "get_transcription_cache",
    "save_transcription_cache",
    # Pipeline metrics
    "record_pipeline_metric",
    "get_pipeline_metrics",
    # Analytics
    "save_analytics_snapshot",
    "get_stats",
    "get_daily_analytics",
    "get_top_channels",
    "get_processing_time_percentiles",
    # Maintenance
    "cleanup_old_jobs",
    "cleanup_old_cache",
    "vacuum_database",
    # Advanced queries
    "get_job_with_details",
    "search_videos",
    "get_disk_usage_summary",
    "get_failure_analysis",
    # Models
    "Job",
    "Video",
    "ExportRecord",
    "TranscriptionCache",
    "PipelineMetric",
    "AnalyticsSnapshot",
    # Constants
    "JOB_STATUS_VALUES",
    "PRIORITY_VALUES",
    "PLATFORM_VALUES",
]
