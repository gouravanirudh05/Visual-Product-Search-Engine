from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper


APP_TITLE = "Visual Product Search Engine"
DEFAULT_INDEX_NAME = "vr-clothing-gallery"
SUPPORTED_FINETUNED_SEEDS = ("104", "541")
SUPPORTED_FINETUNED_ALPHAS = ("0.7", "0.5")
DEFAULT_FINETUNED_SEED = "104"
DEFAULT_FINETUNED_ALPHA = "0.7"
DEFAULT_NAMESPACE = f"finetuned-alpha-{DEFAULT_FINETUNED_ALPHA}-seed{DEFAULT_FINETUNED_SEED}"
SEED_NAMESPACE_PATTERN = re.compile(r"^finetuned-alpha-(?P<alpha>\d+(?:\.\d+)?)-seed(?P<seed>\d+)$")
CHECKPOINT_SEED_PATTERN = re.compile(r"seed(?P<seed>\d+)", re.IGNORECASE)
NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}
BLIP2_SCORE_FLOOR = 0.05


@dataclass(frozen=True)
class Settings:
    blip2_server_url: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_namespace: str
    finetuned_seed: str
    finetuned_alpha: str
    gallery_csv: str
    captions_csv: str
    image_root: str
    yolo_model_path: str
    clip_checkpoint: str
    clip_model: str
    clip_pretrained: str
    candidate_k: int
    blip2_rerank_k: int
    timeout_seconds: int
    health_timeout_seconds: int


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def normalize_seed(raw_seed: str | None) -> str:
    seed = str(raw_seed or "").strip()
    return seed if seed in SUPPORTED_FINETUNED_SEEDS else DEFAULT_FINETUNED_SEED


def normalize_alpha(raw_alpha: str | None) -> str:
    alpha = str(raw_alpha or "").strip()
    return alpha if alpha in SUPPORTED_FINETUNED_ALPHAS else DEFAULT_FINETUNED_ALPHA


def format_seed_namespace(seed: str, alpha: str) -> str:
    return f"finetuned-alpha-{alpha}-seed{seed}"


def parse_seed_namespace(namespace: str) -> dict[str, str] | None:
    match = SEED_NAMESPACE_PATTERN.fullmatch(namespace.strip())
    if not match:
        return None
    return {
        "seed": match.group("seed"),
        "alpha": match.group("alpha"),
    }


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        blip2_server_url=os.getenv("BLIP2_SERVER_URL", "").rstrip("/"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", DEFAULT_INDEX_NAME),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "").strip(),
        finetuned_seed=normalize_seed(os.getenv("FINETUNED_SEED", DEFAULT_FINETUNED_SEED)),
        finetuned_alpha=normalize_alpha(os.getenv("FINETUNED_ALPHA", DEFAULT_FINETUNED_ALPHA)),
        gallery_csv=os.getenv("GALLERY_CSV", ""),
        captions_csv=os.getenv("CAPTIONS_CSV", ""),
        image_root=os.getenv("IMAGE_ROOT", ""),
        yolo_model_path=os.getenv("YOLO_MODEL_PATH", "yolov8n.pt"),
        clip_checkpoint=os.getenv("CLIP_CHECKPOINT", "").strip(),
        clip_model=os.getenv("CLIP_MODEL", "ViT-L-14"),
        clip_pretrained=os.getenv("CLIP_PRETRAINED", "openai"),
        candidate_k=env_int("CANDIDATE_K", 50),
        blip2_rerank_k=env_int("BLIP2_RERANK_K", 10),
        timeout_seconds=env_int("BLIP2_TIMEOUT_SECONDS", 120),
        health_timeout_seconds=env_int("BLIP2_HEALTH_TIMEOUT_SECONDS", 180),
    )


def to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def raise_for_status_with_detail(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        detail = payload.get("detail") or response.text
        raise RuntimeError(f"{response.status_code}: {detail}") from exc


def normalize_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def center_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def manual_crop_controls(image: Image.Image) -> Image.Image:
    # Removed in favor of streamlit_cropper
    return center_crop(image)


@st.cache_resource(show_spinner=False)
def load_yolo(model_path: str):
    try:
        from ultralytics import YOLO

        return YOLO(model_path)
    except Exception as exc:  # noqa: BLE001 - shown to user in sidebar.
        return exc


def crop_with_yolo(image: Image.Image, model_path: str) -> tuple[tuple[int, int, int, int] | None, str]:
    if not model_path:
        return None, "YOLO disabled. Enable manual crop if needed."

    model = load_yolo(model_path)
    if isinstance(model, Exception):
        return None, f"YOLO unavailable ({model}). Using full image."

    results = model.predict(image, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, "No YOLO box found. Using full image."

    areas = []
    for box in boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box
        areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    x1, y1, x2, y2 = boxes.xyxy[int(np.argmax(areas))].cpu().numpy()
    width, height = image.size
    pad_x = 0.04 * (x2 - x1)
    pad_y = 0.04 * (y2 - y1)
    
    # st_cropper expects (left, right, top, bottom)
    box = (
        max(0, int(x1 - pad_x)),
        min(width, int(x2 + pad_x)),
        max(0, int(y1 - pad_y)),
        min(height, int(y2 + pad_y)),
    )
    return box, "YOLO crop selected from the largest detected product region."


@st.cache_resource(show_spinner=False)
def load_clip(model_name: str, pretrained: str, checkpoint_path: str):
    import torch
    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    if checkpoint_path:
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"CLIP checkpoint not found: {checkpoint}. "
                "Set CLIP_CHECKPOINT explicitly or make sure the seed-specific file is available locally."
            )
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict):
            state = (
                state.get("model_state_dict")
                or state.get("model_state")
                or state.get("state_dict")
                or state
            )
            state = {key.replace("module.", ""): value for key, value in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, preprocess, device


def candidate_config_roots(settings: Settings) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for raw_path in (
        settings.image_root,
        settings.gallery_csv,
        settings.captions_csv,
        settings.clip_checkpoint,
    ):
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        root = path if path.is_dir() else path.parent
        if root not in seen:
            roots.append(root)
            seen.add(root)
    for root in (Path.cwd(), Path.cwd() / "archive"):
        if root not in seen:
            roots.append(root)
            seen.add(root)
    return roots


def remap_checkpoint_for_seed(checkpoint_path: str, seed: str) -> str | None:
    path = Path(checkpoint_path).expanduser()
    if not CHECKPOINT_SEED_PATTERN.search(path.name):
        return None
    remapped_name = CHECKPOINT_SEED_PATTERN.sub(f"seed{seed}", path.name, count=1)
    return str(path.with_name(remapped_name))


def resolve_clip_checkpoint(settings: Settings, namespace: str) -> str:
    namespace_bits = parse_seed_namespace(namespace)
    if settings.clip_checkpoint:
        if namespace_bits:
            remapped = remap_checkpoint_for_seed(settings.clip_checkpoint, namespace_bits["seed"])
            if remapped:
                return remapped
        return settings.clip_checkpoint

    if not namespace_bits:
        return ""

    seed = namespace_bits["seed"]
    checkpoint_names = (
        f"clip_best_seed{seed}.pt",
        f"clip_seed_{seed}.pt",
        f"clip_seed{seed}.pt",
    )
    roots = candidate_config_roots(settings)
    for root in roots:
        for checkpoint_name in checkpoint_names:
            candidate = root / checkpoint_name
            if candidate.exists():
                return str(candidate)

    if roots:
        return str(roots[0] / checkpoint_names[0])
    return checkpoint_names[0]


def checkpoint_seed_from_path(checkpoint_path: str) -> str | None:
    match = CHECKPOINT_SEED_PATTERN.search(Path(checkpoint_path).name)
    if not match:
        return None
    return match.group("seed")


def build_runtime_settings(base_settings: Settings, seed: str, alpha: str) -> Settings:
    namespace = format_seed_namespace(seed, alpha)
    checkpoint = resolve_clip_checkpoint(base_settings, namespace)
    return replace(
        base_settings,
        pinecone_namespace=namespace,
        finetuned_seed=seed,
        finetuned_alpha=alpha,
        clip_checkpoint=checkpoint,
    )


def runtime_config_warnings(settings: Settings) -> list[str]:
    warnings: list[str] = []
    namespace_bits = parse_seed_namespace(settings.pinecone_namespace)
    if namespace_bits:
        if not settings.clip_checkpoint:
            warnings.append(
                "This seed-specific namespace needs the matching fine-tuned CLIP checkpoint. "
                "Set CLIP_CHECKPOINT or keep the expected file next to your dataset."
            )
        elif not Path(settings.clip_checkpoint).expanduser().exists():
            warnings.append(f"Expected CLIP checkpoint not found: {settings.clip_checkpoint}")
        checkpoint_seed = checkpoint_seed_from_path(settings.clip_checkpoint)
        if checkpoint_seed and checkpoint_seed != namespace_bits["seed"]:
            warnings.append(
                "Namespace seed and checkpoint seed do not match. "
                "Pick the same seed for Pinecone and CLIP before searching."
            )
    elif not settings.clip_checkpoint:
        warnings.append(
            "CLIP_CHECKPOINT is empty, so query encoding will use the pretrained OpenAI CLIP weights."
        )
    return warnings


def encode_query_image(image: Image.Image, settings: Settings) -> list[float]:
    import torch

    model, preprocess, device = load_clip(
        settings.clip_model,
        settings.clip_pretrained,
        settings.clip_checkpoint,
    )
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.squeeze(0).cpu().numpy().astype(float).tolist()


@st.cache_resource(show_spinner=False)
def get_pinecone_index(api_key: str, index_name: str):
    from pinecone import Pinecone

    return Pinecone(api_key=api_key).Index(index_name)


def query_index(vector: list[float], settings: Settings) -> list[dict[str, Any]]:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set.")

    index = get_pinecone_index(settings.pinecone_api_key, settings.pinecone_index_name)
    response = index.query(
        vector=vector,
        top_k=settings.candidate_k,
        namespace=settings.pinecone_namespace,
        include_metadata=True,
    )
    matches = response.get("matches", []) if isinstance(response, dict) else response.matches
    candidates = []
    for match in matches:
        metadata = match.get("metadata", {}) if isinstance(match, dict) else (match.metadata or {})
        score = match.get("score", 0.0) if isinstance(match, dict) else match.score
        match_id = match.get("id", "") if isinstance(match, dict) else match.id
        candidates.append({"id": match_id, "clip_score": float(score), "metadata": metadata})
    return candidates


@st.cache_data(show_spinner=False)
def load_caption_lookup(captions_csv: str) -> dict[str, str]:
    if not captions_csv or not Path(captions_csv).exists():
        return {}

    df = pd.read_csv(captions_csv)
    if "image_name" not in df.columns:
        return {}
    caption_col = "blip2_caption" if "blip2_caption" in df.columns else df.columns[-1]
    return dict(zip(df["image_name"].astype(str), df[caption_col].fillna("").astype(str)))


def enrich_candidates(candidates: list[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    captions = load_caption_lookup(settings.captions_csv)
    enriched = []
    for candidate in candidates:
        metadata = dict(candidate.get("metadata") or {})
        image_name = str(
            metadata.get("image_name")
            or metadata.get("filename")
            or metadata.get("path")
            or candidate.get("id")
        )
        caption = str(metadata.get("caption") or metadata.get("blip2_caption") or captions.get(image_name, ""))
        enriched.append(
            {
                **candidate,
                "image_name": image_name,
                "item_id": metadata.get("item_id", metadata.get("product_id", "")),
                "caption": caption,
            }
        )
    return enriched


def coerce_remote_score(remote: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        raw_score = remote.get(key)
        if raw_score is None:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(score):
            continue
        return max(BLIP2_SCORE_FLOOR, score) if score <= 0 else score
    return None


def request_blip2_rerank(
    query_crop: Image.Image,
    candidates: list[dict[str, Any]],
    settings: Settings,
) -> tuple[list[dict[str, Any]], str]:
    if not settings.blip2_server_url:
        return candidates, "BLIP2_SERVER_URL is not set. Showing CLIP/Pinecone ranking only."

    rerank_count = max(0, min(settings.blip2_rerank_k, len(candidates)))
    if rerank_count == 0:
        return candidates, "BLIP2_RERANK_K is 0. Showing CLIP/Pinecone ranking only."

    rerank_candidates = candidates[:rerank_count]
    payload_candidates = [
        {
            "id": row["id"],
            "caption": row.get("caption", ""),
            "clip_score": row.get("clip_score", 0.0),
            "metadata": row.get("metadata", {}),
        }
        for row in rerank_candidates
    ]

    files = {"image": ("query_crop.png", to_png_bytes(query_crop), "image/png")}
    data = {"candidates": json.dumps(payload_candidates)}
    try:
        response = requests.post(
            f"{settings.blip2_server_url}/rerank",
            files=files,
            data=data,
            headers=NGROK_HEADERS,
            timeout=settings.timeout_seconds,
        )
        raise_for_status_with_detail(response)
        by_id = {str(row["id"]): row for row in response.json().get("results", [])}
    except Exception as exc:  # noqa: BLE001 - this keeps the demo alive during viva.
        return candidates, f"Remote BLIP-2 re-rank failed: {exc}. Showing CLIP/Pinecone ranking only."

    reranked = []
    for candidate in rerank_candidates:
        remote = by_id.get(str(candidate["id"]), {})
        clip_score = float(candidate.get("clip_score", 0.0))
        blip2_score = coerce_remote_score(remote, ("blip2_score", "blip_score", "score"))
        final_score = coerce_remote_score(remote, ("final_score",))
        reranked.append(
            {
                **candidate,
                "blip2_score": blip2_score,
                "final_score": final_score if final_score is not None else clip_score,
            }
        )
    reranked.extend(
        {
            **candidate,
            "final_score": candidate.get("clip_score", 0.0),
        }
        for candidate in candidates[rerank_count:]
    )
    reranked[:rerank_count] = sorted(
        reranked[:rerank_count],
        key=lambda row: row.get("final_score", row.get("clip_score", 0.0)),
        reverse=True,
    )
    return reranked, f"Remote BLIP-2 image-text matching re-rank applied to top {rerank_count} candidates."


def blip2_health(settings: Settings) -> tuple[bool, str]:
    if not settings.blip2_server_url:
        return False, "BLIP2_SERVER_URL is not set."

    try:
        response = requests.post(
            f"{settings.blip2_server_url}/warmup",
            headers=NGROK_HEADERS,
            timeout=settings.health_timeout_seconds,
        )
        raise_for_status_with_detail(response)
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - surfaced in the sidebar.
        return False, str(exc)

    model_name = payload.get("model_id") or payload.get("model_name") or "BLIP-2"
    backend = payload.get("backend")
    score_mode = payload.get("final_score_mode")
    details = ", ".join(str(value) for value in (backend, score_mode) if value)
    return True, f"{model_name} warmed and ready" + (f" ({details})." if details else ".")


def local_image_path(image_name: str, image_root: str) -> Path | None:
    if not image_root:
        return None
    root = Path(image_root)
    direct = root / image_name
    if direct.exists():
        return direct
    matches = list(root.rglob(Path(image_name).name))
    return matches[0] if matches else None


def render_candidate(row: dict[str, Any], rank: int, settings: Settings) -> None:
    with st.container(border=True):
        cols = st.columns([1, 2])
        with cols[0]:
            path = local_image_path(row.get("image_name", ""), settings.image_root)
            if path and path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.caption("Catalog image path not available")
        with cols[1]:
            st.subheader(f"Rank {rank}")
            if row.get("item_id"):
                st.write(f"Item ID: `{row['item_id']}`")
            st.write(f"Image: `{row.get('image_name', row.get('id', ''))}`")
            if row.get("caption"):
                st.caption(row["caption"])
            metric_cols = st.columns(3)
            metric_cols[0].metric("CLIP", f"{row.get('clip_score', 0.0):.4f}")
            blip_score = row.get("blip2_score")
            metric_cols[1].metric("BLIP-2", "N/A" if blip_score is None else f"{blip_score:.4f}")
            metric_cols[2].metric("Final", f"{row.get('final_score', row.get('clip_score', 0.0)):.4f}")


def sidebar(base_settings: Settings) -> tuple[int, Settings]:
    with st.sidebar:
        st.header("Runtime")
        top_k = st.slider("Results", 5, 30, 10, step=5)

        namespace_bits = parse_seed_namespace(base_settings.pinecone_namespace)
        if base_settings.pinecone_namespace and namespace_bits is None:
            runtime_settings = replace(
                base_settings,
                clip_checkpoint=resolve_clip_checkpoint(base_settings, base_settings.pinecone_namespace),
            )
            st.caption("Using custom `PINECONE_NAMESPACE` from environment.")
        else:
            default_seed = normalize_seed(
                namespace_bits["seed"] if namespace_bits else base_settings.finetuned_seed
            )
            default_alpha = normalize_alpha(
                namespace_bits["alpha"] if namespace_bits else base_settings.finetuned_alpha
            )
            seed = st.selectbox(
                "Fine-tuned seed",
                options=list(SUPPORTED_FINETUNED_SEEDS),
                index=SUPPORTED_FINETUNED_SEEDS.index(default_seed),
            )
            alpha = st.selectbox(
                "Namespace alpha",
                options=list(SUPPORTED_FINETUNED_ALPHAS),
                index=SUPPORTED_FINETUNED_ALPHAS.index(default_alpha),
            )
            runtime_settings = build_runtime_settings(base_settings, seed, alpha)

        st.text_input("BLIP-2 server", value=runtime_settings.blip2_server_url or "not set", disabled=True)
        st.text_input("Pinecone index", value=runtime_settings.pinecone_index_name, disabled=True)
        st.text_input("Namespace", value=runtime_settings.pinecone_namespace or "not set", disabled=True)
        st.text_input(
            "CLIP checkpoint",
            value=runtime_settings.clip_checkpoint or "pretrained OpenAI CLIP",
            disabled=True,
        )
        for warning in runtime_config_warnings(runtime_settings):
            st.warning(warning)
        st.caption("BLIP-2 re-ranking stays the same across seeds; the namespace and CLIP checkpoint change together here.")
        if st.button("Check BLIP-2 server"):
            ok, message = blip2_health(runtime_settings)
            if ok:
                st.success(message)
            else:
                st.error(message)
        st.caption("Local config is read from .env or environment variables.")
    return top_k, runtime_settings


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    base_settings = load_settings()
    top_k, settings = sidebar(base_settings)

    st.title(APP_TITLE)
    st.write("Upload a fashion product image, confirm the detected crop, then retrieve visually and semantically similar catalog items.")

    uploaded = st.file_uploader("Query image", type=["jpg", "jpeg", "png", "webp"])
    if not uploaded:
        st.info("Waiting for an input image.")
        return

    image = normalize_image(Image.open(uploaded))
    upload_signature = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("upload_signature") != upload_signature:
        st.session_state.upload_signature = upload_signature
        st.session_state.confirmed_crop = False
        st.session_state.manual_crop = False

    yolo_box, crop_note = crop_with_yolo(image, settings.yolo_model_path)
    st.session_state.setdefault("confirmed_crop", False)
    st.session_state.setdefault("manual_crop", True)

    st.session_state.manual_crop = st.checkbox("Adjust crop manually", value=st.session_state.manual_crop)
    
    st.subheader("Product crop")
    if st.session_state.manual_crop:
        st.write("Adjust the bounding box below to refine the product crop.")
        # Try to use YOLO default coordinates if available
        crop = st_cropper(
            image, 
            realtime_update=True, 
            box_color='#00FF00',
            aspect_ratio=None,
            default_coords=yolo_box if yolo_box else None
        )
    else:
        if yolo_box:
            # yolo_box is (left, right, top, bottom)
            crop = image.crop((yolo_box[0], yolo_box[2], yolo_box[1], yolo_box[3]))
        else:
            crop = center_crop(image)
        st.image(crop, use_container_width=True)
    st.caption(crop_note)

    actions = st.columns([1, 1, 4])
    if actions[0].button("Confirm crop", type="primary"):
        st.session_state.confirmed_crop = True
    if actions[1].button("Re-crop"):
        st.session_state.confirmed_crop = False
        st.session_state.manual_crop = True
        st.rerun()

    if not st.session_state.confirmed_crop:
        st.stop()

    with st.spinner("Encoding query, searching the ANN index, and re-ranking candidates..."):
        try:
            vector = encode_query_image(crop, settings)
            candidates = query_index(vector, settings)
            candidates = enrich_candidates(candidates, settings)
            results, rerank_note = request_blip2_rerank(crop, candidates, settings)
        except Exception as exc:  # noqa: BLE001 - Streamlit should show setup gaps cleanly.
            st.error(f"Search pipeline could not run: {exc}")
            st.stop()

    st.success(rerank_note)
    st.subheader("Retrieved products")
    if not results:
        st.warning("No results returned from the vector index.")
        return

    for rank, row in enumerate(results[:top_k], start=1):
        render_candidate(row, rank, settings)


if __name__ == "__main__":
    main()
