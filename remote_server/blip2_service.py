from __future__ import annotations

import io
import json
import os
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps


BACKEND = os.getenv("BLIP2_BACKEND", "lavis_itm").strip().lower()
LAVIS_MODEL_NAME = os.getenv("BLIP2_LAVIS_MODEL_NAME", "blip2_image_text_matching")
LAVIS_MODEL_TYPE = os.getenv("BLIP2_LAVIS_MODEL_TYPE", "pretrain")
TRANSFORMERS_MODEL_ID = os.getenv("BLIP2_MODEL_ID", "Salesforce/blip2-opt-2.7b")
FINAL_SCORE_MODE = os.getenv("BLIP2_FINAL_SCORE_MODE", "blip2").strip().lower()
CLIP_WEIGHT = float(os.getenv("CLIP_WEIGHT", "0.7"))
BATCH_SIZE = max(1, int(os.getenv("BLIP2_BATCH_SIZE", "16")))
BLIP2_SCORE_FLOOR = float(os.getenv("BLIP2_SCORE_FLOOR", "0.05"))

app = FastAPI(title="BLIP-2 Re-ranking Service")
_model_bundle: dict[str, Any] | None = None


class PatchedTokenizer:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["padding"] = True
        return self.tokenizer(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)


def model_label() -> str:
    if BACKEND == "lavis_itm":
        return f"{LAVIS_MODEL_NAME}:{LAVIS_MODEL_TYPE}"
    return TRANSFORMERS_MODEL_ID


def load_model():
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if BACKEND == "lavis_itm":
        try:
            from lavis.models import load_model_and_preprocess
        except ImportError as exc:
            raise RuntimeError(
                "BLIP2_BACKEND=lavis_itm requires salesforce-lavis. "
                "Install the LAVIS dependencies from requirements-blip2-server.txt, "
                "then run: pip install --no-deps salesforce-lavis"
            ) from exc

        model, vis_processors, txt_processors = load_model_and_preprocess(
            name=LAVIS_MODEL_NAME,
            model_type=LAVIS_MODEL_TYPE,
            is_eval=True,
            device=device,
        )
        model.tokenizer = PatchedTokenizer(model.tokenizer)
        _model_bundle = {
            "backend": BACKEND,
            "device": device,
            "model": model,
            "vis_processors": vis_processors,
            "txt_processors": txt_processors,
        }
        return _model_bundle

    if BACKEND != "transformers_loss":
        raise RuntimeError(
            "Unsupported BLIP2_BACKEND. Use 'lavis_itm' for notebook-aligned ITM scoring "
            "or 'transformers_loss' for the older caption-likelihood fallback."
        )

    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = Blip2Processor.from_pretrained(TRANSFORMERS_MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        TRANSFORMERS_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    _model_bundle = {
        "backend": BACKEND,
        "device": device,
        "dtype": dtype,
        "model": model,
        "processor": processor,
    }
    return _model_bundle


def read_image(raw: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")


def batched(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def clean_caption(caption: Any) -> str:
    return str(caption or "").strip() or "clothing product"


@torch.no_grad()
def score_candidates_lavis_itm(image: Image.Image, captions: list[str]) -> list[float]:
    bundle = load_model()
    model = bundle["model"]
    device = bundle["device"]
    vis_processors = bundle["vis_processors"]
    txt_processors = bundle["txt_processors"]

    img_tensor = vis_processors["eval"](image).unsqueeze(0).to(device)
    scores: list[float] = []
    for caption_batch in batched(captions, BATCH_SIZE):
        clean_captions = [clean_caption(caption) for caption in caption_batch]
        img_batch = img_tensor.repeat(len(clean_captions), 1, 1, 1)
        txt_batch = [txt_processors["eval"](caption) for caption in clean_captions]
        itm_output = model({"image": img_batch, "text_input": txt_batch}, match_head="itm")
        batch_scores = torch.nn.functional.softmax(itm_output.float(), dim=1)[:, 1]
        scores.extend(float(score) for score in batch_scores.cpu().tolist())
    return scores


@torch.no_grad()
def score_candidates_transformers_loss(image: Image.Image, captions: list[str]) -> list[float]:
    bundle = load_model()
    processor = bundle["processor"]
    model = bundle["model"]
    device = bundle["device"]
    dtype = bundle["dtype"]
    losses: list[float] = []
    for caption_batch in batched(captions, BATCH_SIZE):
        clean_captions = [clean_caption(caption) for caption in caption_batch]
        images = [image] * len(clean_captions)
        inputs = processor(
            images=images,
            text=clean_captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        inputs = {
            key: value.to(device, dtype=dtype) if value.is_floating_point() else value.to(device)
            for key, value in inputs.items()
        }
        labels = inputs["input_ids"].clone()
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        outputs = model(**inputs, labels=labels)
        logits = outputs.logits[:, :-1, :].float()
        target = labels[:, 1:]
        valid = target.ne(-100)
        safe_target = target.masked_fill(~valid, 0)
        token_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            safe_target.reshape(-1),
            reduction="none",
        ).reshape(target.shape)
        sequence_loss = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

        # Lower caption loss means a better image-text match.
        losses.extend(sequence_loss.detach().cpu().tolist())

    if len(losses) == 1:
        return [1.0]

    min_loss = min(losses)
    max_loss = max(losses)
    if max_loss == min_loss:
        return [0.5 for _ in losses]

    # Convert losses to relative scores for this candidate set.
    # Keep a small floor/ceiling so the demo does not show hard 0/1 endpoints
    # just because an item is the worst/best within the current candidate batch.
    score_floor = min(max(BLIP2_SCORE_FLOOR, 0.0), 0.49)
    score_span = 1.0 - (2.0 * score_floor)
    return [
        float(score_floor + score_span * ((max_loss - loss) / (max_loss - min_loss)))
        for loss in losses
    ]


def score_candidates(image: Image.Image, captions: list[str]) -> list[float]:
    if not captions:
        return []
    if BACKEND == "lavis_itm":
        return score_candidates_lavis_itm(image, captions)
    return score_candidates_transformers_loss(image, captions)


def parse_clip_score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("clip_score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def combine_scores(clip_score: float, blip_score: float) -> float:
    if FINAL_SCORE_MODE == "blend":
        clip_weight = min(max(CLIP_WEIGHT, 0.0), 1.0)
        return (clip_weight * clip_score) + ((1.0 - clip_weight) * blip_score)
    return blip_score


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "backend": BACKEND,
        "model_id": model_label(),
        "batch_size": BATCH_SIZE,
        "final_score_mode": FINAL_SCORE_MODE,
        "loaded": _model_bundle is not None,
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "BLIP-2 re-ranking service",
        "backend": BACKEND,
        "health": "/health",
        "warmup": "/warmup",
        "rerank": "/rerank",
    }


@app.post("/warmup")
def warmup() -> dict[str, Any]:
    try:
        bundle = load_model()
    except Exception as exc:  # noqa: BLE001 - expose setup errors to the Streamlit health check.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "ok": True,
        "backend": bundle["backend"],
        "model_id": model_label(),
        "device": str(bundle["device"]),
        "final_score_mode": FINAL_SCORE_MODE,
    }


@app.post("/rerank")
async def rerank(
    image: UploadFile = File(...),
    candidates: str = Form(...),
) -> dict[str, Any]:
    query_image = read_image(await image.read())
    try:
        rows = json.loads(candidates)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Field 'candidates' must be a JSON list.",
        ) from exc

    if not isinstance(rows, list):
        raise HTTPException(status_code=400, detail="Field 'candidates' must be a JSON list.")
    if not rows:
        return {"results": []}

    typed_rows = [row if isinstance(row, dict) else {} for row in rows]
    captions = [clean_caption(row.get("caption")) for row in typed_rows]
    try:
        blip_scores = score_candidates(query_image, captions)
    except Exception as exc:  # noqa: BLE001 - keep model/runtime failures explicit for the caller.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results = []
    for row, blip_score in zip(typed_rows, blip_scores):
        clip_score = parse_clip_score(row)
        blip_score = float(blip_score)
        final_score = combine_scores(clip_score, blip_score)
        results.append(
            {
                "id": row.get("id"),
                "clip_score": clip_score,
                "blip2_score": blip_score,
                "final_score": final_score,
                "metadata": row.get("metadata") or {},
                "caption": row.get("caption") or "",
            }
        )
    results.sort(key=lambda row: row["final_score"], reverse=True)
    return {
        "backend": BACKEND,
        "final_score_mode": FINAL_SCORE_MODE,
        "results": results,
    }
