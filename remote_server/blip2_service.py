from __future__ import annotations

import io
import json
import os
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps


MODEL_ID = os.getenv("BLIP2_MODEL_ID", "Salesforce/blip2-opt-2.7b")
CLIP_WEIGHT = float(os.getenv("CLIP_WEIGHT", "0.7"))
BATCH_SIZE = int(os.getenv("BLIP2_BATCH_SIZE", "4"))

app = FastAPI(title="BLIP-2 Re-ranking Service")
_model_bundle: tuple[Any, Any, torch.device, torch.dtype] | None = None


def load_model():
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    processor = Blip2Processor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    _model_bundle = processor, model, device, dtype
    return _model_bundle


def read_image(raw: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")


def batched(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def score_candidates(image: Image.Image, captions: list[str]) -> list[float]:
    processor, model, device, dtype = load_model()
    if not captions:
        return []

    losses: list[float] = []
    for caption_batch in batched(captions, max(1, BATCH_SIZE)):
        clean_captions = [caption.strip() or "clothing product" for caption in caption_batch]
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

        with torch.no_grad():
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

    # Convert losses to relative 0..1 scores for this candidate set.
    # Best caption gets 1.0, worst gets 0.0.
    return [float((max_loss - loss) / (max_loss - min_loss)) for loss in losses]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "model_id": MODEL_ID,
        "batch_size": BATCH_SIZE,
        "backend": "transformers",
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "BLIP-2 re-ranking service",
        "backend": "transformers",
        "health": "/health",
        "rerank": "/rerank",
    }


@app.post("/warmup")
def warmup() -> dict[str, Any]:
    load_model()
    return {"ok": True, "model_id": MODEL_ID}


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

    captions = [str(row.get("caption") or "") for row in rows]
    blip_scores = score_candidates(query_image, captions)

    results = []
    for row, blip_score in zip(rows, blip_scores):
        clip_score = float(row.get("clip_score") or 0.0)
        final_score = CLIP_WEIGHT * clip_score + (1.0 - CLIP_WEIGHT) * float(blip_score)
        results.append(
            {
                "id": row.get("id"),
                "clip_score": clip_score,
                "blip2_score": float(blip_score),
                "final_score": final_score,
                "metadata": row.get("metadata") or {},
            }
        )
    results.sort(key=lambda row: row["final_score"], reverse=True)
    return {"results": results}
