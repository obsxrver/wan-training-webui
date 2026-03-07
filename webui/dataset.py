from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from fastapi import HTTPException, UploadFile

from .config import (
    CAPTION_EXTENSIONS,
    CAPTION_PREVIEW_LIMIT,
    CAPTION_PRIORITY,
    DATASET_ROOT,
    IMAGE_EXTENSIONS,
    MEDIA_EXTENSIONS,
    PROTECTED_DATASET_DIRS,
    VIDEO_EXTENSIONS,
)


def clear_dataset_directory() -> None:
    try:
        if DATASET_ROOT.exists() and not DATASET_ROOT.is_dir():
            raise HTTPException(status_code=500, detail="Dataset path is not a directory")
        if not DATASET_ROOT.exists():
            DATASET_ROOT.mkdir(parents=True, exist_ok=True)
            return
        for entry in DATASET_ROOT.iterdir():
            try:
                if entry.is_dir() and entry.name.lower() in PROTECTED_DATASET_DIRS:
                    continue
                if entry.is_dir():
                    import shutil

                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            except FileNotFoundError:
                continue
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to prepare dataset directory: {exc}") from exc
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)


def resolve_dataset_file(path: str) -> Path:
    dataset_root = DATASET_ROOT.resolve()
    target_path = (dataset_root / path).resolve()
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if dataset_root not in target_path.parents:
        raise HTTPException(status_code=400, detail="Invalid dataset path")
    return target_path


def collect_dataset_items() -> List[Dict[str, Any]]:
    if not DATASET_ROOT.exists() or not DATASET_ROOT.is_dir():
        return []

    dataset_root = DATASET_ROOT.resolve()
    captions: Dict[tuple[str, str], Dict[str, Any]] = {}

    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in CAPTION_EXTENSIONS:
            continue

        relative = file_path.relative_to(dataset_root)
        parent_key = relative.parent.as_posix()
        stem_key = file_path.stem.lower()
        priority = CAPTION_PRIORITY.get(suffix, 999)
        existing = captions.get((parent_key, stem_key))
        if existing and existing["priority"] <= priority:
            continue

        try:
            text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            text = ""
        if len(text) > CAPTION_PREVIEW_LIMIT:
            text = text[:CAPTION_PREVIEW_LIMIT].rstrip() + "…"

        captions[(parent_key, stem_key)] = {
            "caption_path": relative.as_posix(),
            "caption_text": text,
            "priority": priority,
        }

    items: List[Dict[str, Any]] = []

    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in MEDIA_EXTENSIONS:
            continue

        relative = file_path.relative_to(dataset_root)
        parent_key = relative.parent.as_posix()
        stem_key = file_path.stem.lower()
        caption_info = captions.get((parent_key, stem_key), {})
        media_path = relative.as_posix()
        media_url = f"/dataset/media/{media_path}"
        media_kind = "video" if suffix in VIDEO_EXTENSIONS else "image"

        item: Dict[str, Any] = {
            "media_path": media_path,
            "media_url": media_url,
            "media_kind": media_kind,
            "caption_path": caption_info.get("caption_path"),
            "caption_text": caption_info.get("caption_text"),
        }

        if media_kind == "image":
            item["image_path"] = media_path
            item["image_url"] = media_url
        else:
            item["video_path"] = media_path
            item["video_url"] = media_url

        items.append(item)

    items.sort(key=lambda item: item["media_path"].lower())
    return items


def _is_in_protected_dir(path: Path) -> bool:
    try:
        relative_parts = path.relative_to(DATASET_ROOT).parts
    except ValueError:
        return False
    return any(part.lower() in PROTECTED_DATASET_DIRS for part in relative_parts)


def find_existing_caption_file(media_file: Path) -> Optional[Path]:
    for suffix, _ in sorted(CAPTION_PRIORITY.items(), key=lambda item: item[1]):
        candidate = media_file.with_suffix(suffix)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def resolve_dataset_relative_path(relative_path: str) -> Path:
    dataset_root = DATASET_ROOT.resolve()
    normalized = normalize_export_path(relative_path)
    target_path = (dataset_root / normalized).resolve()
    if dataset_root not in target_path.parents and target_path != dataset_root:
        raise HTTPException(status_code=400, detail="Invalid dataset path")
    return target_path


def normalize_export_path(value: str) -> Path:
    normalized = value.replace("\\", "/").strip().lstrip("/")
    if not normalized:
        raise HTTPException(status_code=400, detail="Invalid file name")
    candidate = Path(normalized)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise HTTPException(status_code=400, detail="Invalid file path")
    return candidate


async def write_upload_to_path(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def delete_dataset_item(relative_path: str) -> Dict[str, Any]:
    file_path = resolve_dataset_file(relative_path)
    deleted: List[str] = []
    captions_removed: List[str] = []
    relative_display = file_path.relative_to(DATASET_ROOT).as_posix()

    try:
        file_path.unlink()
        deleted.append(relative_display)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to delete '{relative_display}': {exc}") from exc

    for suffix in CAPTION_EXTENSIONS:
        caption_path = file_path.with_suffix(suffix)
        if not caption_path.exists() or not caption_path.is_file():
            continue
        try:
            caption_path.unlink()
            captions_removed.append(caption_path.relative_to(DATASET_ROOT).as_posix())
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Removed media but failed to delete caption '{caption_path.name}': {exc}",
            ) from exc

    removed_count = 1 + len(captions_removed)
    message = f"Removed '{relative_display}' from the dataset."
    if captions_removed:
        caption_count = len(captions_removed)
        plural = "s" if caption_count != 1 else ""
        message += f" Deleted {caption_count} caption file{plural}."

    return {
        "message": message,
        "deleted": deleted,
        "captions_removed": captions_removed,
        "removed_count": removed_count,
    }


def get_caption(caption_path: Optional[str], media_path: Optional[str]) -> Dict[str, Any]:
    if caption_path:
        relative_caption = caption_path.strip()
        if not relative_caption:
            raise HTTPException(status_code=400, detail="caption_path is required")
        caption_file = resolve_dataset_relative_path(relative_caption)
        if not caption_file.exists() or not caption_file.is_file():
            raise HTTPException(status_code=404, detail="Caption file not found")
    elif media_path:
        normalized_media = media_path.strip()
        if not normalized_media:
            raise HTTPException(status_code=400, detail="media_path is required")
        media_file = resolve_dataset_file(normalized_media)
        caption_file = find_existing_caption_file(media_file)
        if caption_file is None:
            return {"caption_text": "", "caption_path": None}
        relative_caption = caption_file.relative_to(DATASET_ROOT).as_posix()
    else:
        raise HTTPException(status_code=400, detail="caption_path or media_path is required")

    try:
        caption_text = caption_file.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read caption: {exc}") from exc

    return {"caption_text": caption_text, "caption_path": relative_caption}


def update_caption(media_path: str, caption_text_raw: Optional[str], caption_path: Optional[str]) -> Dict[str, Any]:
    media_path = media_path.strip()
    if not media_path:
        raise HTTPException(status_code=400, detail="media_path is required")

    media_file = resolve_dataset_file(media_path)
    caption_text = (caption_text_raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    if caption_path:
        relative_caption = caption_path.strip()
        if not relative_caption:
            raise HTTPException(status_code=400, detail="caption_path is invalid")
        suffix = Path(relative_caption).suffix.lower()
        if suffix not in CAPTION_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Caption file must use a supported extension")
        caption_file = resolve_dataset_relative_path(relative_caption)
    else:
        caption_file = media_file.with_suffix(".txt")
        relative_caption = caption_file.relative_to(DATASET_ROOT).as_posix()

    caption_file.parent.mkdir(parents=True, exist_ok=True)

    if not caption_text:
        if caption_file.exists():
            try:
                caption_file.unlink()
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Failed to delete caption file: {exc}") from exc
            message = f"Caption removed for '{media_path}'."
        else:
            message = f"No caption text provided for '{media_path}'."
        return {"message": message, "caption_text": "", "caption_path": None}

    try:
        caption_file.write_text(caption_text, encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save caption: {exc}") from exc

    message = f"Caption updated for '{media_path}'."
    return {"message": message, "caption_text": caption_text, "caption_path": relative_caption}


def bulk_caption(caption_text_raw: str, apply_to: str) -> Dict[str, Any]:
    caption_text = caption_text_raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not caption_text:
        raise HTTPException(status_code=400, detail="caption_text is required")

    if not DATASET_ROOT.exists() or not DATASET_ROOT.is_dir():
        return {
            "message": "Dataset directory is missing.",
            "updated_count": 0,
            "skipped_count": 0,
            "total_images": 0,
            "caption_text": caption_text,
        }

    total_images = 0
    updated_count = 0
    skipped_count = 0
    dataset_root = DATASET_ROOT.resolve()

    for file_path in dataset_root.rglob("*"):
        if not file_path.is_file() or _is_in_protected_dir(file_path):
            continue
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS and file_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        total_images += 1
        existing_caption = find_existing_caption_file(file_path)
        if apply_to == "uncaptioned_images" and existing_caption is not None:
            skipped_count += 1
            continue

        caption_file = file_path.with_suffix(".txt")
        caption_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            caption_file.write_text(caption_text, encoding="utf-8")
        except OSError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save caption for '{file_path.name}': {exc}",
            ) from exc

        updated_count += 1

    if total_images == 0:
        message = "No images found to caption."
    else:
        plural = "s" if total_images != 1 else ""
        message = f"Applied caption to {updated_count} of {total_images} image{plural}."
        if apply_to == "uncaptioned_images" and skipped_count:
            skipped_plural = "s" if skipped_count != 1 else ""
            message += f" Skipped {skipped_count} image{skipped_plural} with existing captions."

    return {
        "message": message,
        "updated_count": updated_count,
        "skipped_count": skipped_count,
        "total_images": total_images,
        "caption_text": caption_text,
    }


def iter_file_chunks(
    file_path: Path,
    start: int = 0,
    end: Optional[int] = None,
    chunk_size: int = 1024 * 1024,
) -> Generator[bytes, None, None]:
    with file_path.open("rb") as stream:
        stream.seek(start)
        bytes_remaining = (end - start + 1) if end is not None else None
        while True:
            if bytes_remaining is not None and bytes_remaining <= 0:
                break
            read_size = chunk_size if bytes_remaining is None else min(chunk_size, bytes_remaining)
            data = stream.read(read_size)
            if not data:
                break
            if bytes_remaining is not None:
                bytes_remaining -= len(data)
            yield data
