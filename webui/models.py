from typing import Literal, Optional

from pydantic import BaseModel, Field

from .video_conversion import DEFAULT_DATASET_CONFIG


class TrainRequest(BaseModel):
    title_prefix: str = Field(default="mylora", min_length=1)
    author: str = Field(default="authorName", min_length=1)
    dataset_path: str = Field(default=str(DEFAULT_DATASET_CONFIG))
    save_every: int = Field(default=100, ge=1)
    max_epochs: int = Field(default=100, ge=1)
    cpu_threads_per_process: Optional[int] = Field(default=None, ge=1)
    max_data_loader_workers: Optional[int] = Field(default=None, ge=1)
    upload_cloud: bool = True
    shutdown_instance: bool = True
    auto_confirm: bool = True
    training_mode: Literal["t2v", "i2v"] = "t2v"
    noise_mode: Literal["both", "high", "low", "combined"] = "both"
    convert_videos_to_16fps: bool = False
    cloud_connection_id: Optional[str] = None


class ApiKeyRequest(BaseModel):
    api_key: str = Field(min_length=1)


class DeleteDatasetItem(BaseModel):
    media_path: str = Field(min_length=1)


class UpdateCaptionRequest(BaseModel):
    media_path: str = Field(min_length=1)
    caption_text: Optional[str] = None
    caption_path: Optional[str] = None


class BulkCaptionRequest(BaseModel):
    caption_text: str = Field(min_length=1)
    apply_to: Literal["all_images", "uncaptioned_images"] = "all_images"
