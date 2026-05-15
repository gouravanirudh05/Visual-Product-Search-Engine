# Run Streamlit locally and BLIP-2 on Kaggle

This is the intended setup when your laptop has no GPU:

- Laptop: Streamlit app, YOLO crop, CLIP query encoding, Pinecone search, result display.
- Kaggle GPU: BLIP-2 image-text matching server only.

## 1. Prepare the local dataset paths

Download the Kaggle dataset version shown in your screenshot to your laptop.

Your local dataset folder should contain:

```text
version_5/
  img/
  blip2_captions_gallery.csv
  blip2_captions_train.csv
  clip_best_seed104.pt
  clip_best_seed541.pt
  gallery.csv
  query.csv
  train.csv
```

`IMAGE_ROOT` must point to `version_5`, because the CSV image paths look like:

```text
img/img/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg
```

## 2. Start BLIP-2 on Kaggle GPU

In a Kaggle notebook with GPU enabled:

```bash
git clone YOUR_REPO_URL
cd Visual-Product-Search-Engine
pip install -r requirements-blip2-server.txt
pip install --no-deps salesforce-lavis
```

The server defaults to LAVIS `blip2_image_text_matching`, which is the same BLIP-2 ITM scorer used by `finetuned-multiseed-eval.ipynb`. Installing `salesforce-lavis` with `--no-deps` avoids pulling older dependency pins over the packages listed in `requirements-blip2-server.txt`.

Then restart the Kaggle kernel if you installed these packages after importing Python libraries.

If you have an ngrok auth token, set it:

```bash
export NGROK_AUTHTOKEN="YOUR_NGROK_TOKEN"
```

Set the BLIP-2 backend before launch:

```bash
export BLIP2_BACKEND="lavis_itm"
export BLIP2_FINAL_SCORE_MODE="blip2"
export BLIP2_BATCH_SIZE=16
```

Start the BLIP-2 API and ngrok tunnel:

```bash
python remote_server/run_blip2_ngrok.py
```

Copy the printed URL:

```text
BLIP-2 service public URL: https://xxxx.ngrok-free.app
```

Keep this Kaggle notebook running.

## 3. Configure the local Streamlit app

On your laptop:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
BLIP2_SERVER_URL=https://xxxx.ngrok-free.app
PINECONE_API_KEY=YOUR_PINECONE_KEY
PINECONE_INDEX_NAME=vr-clothing-gallery
GALLERY_CSV=/absolute/path/to/version_5/gallery.csv
CAPTIONS_CSV=/absolute/path/to/version_5/blip2_captions_gallery.csv
IMAGE_ROOT=/absolute/path/to/version_5
FINETUNED_SEED=104
FINETUNED_ALPHA=0.7
CLIP_CHECKPOINT=/absolute/path/to/version_5/clip_best_seed104.pt
CANDIDATE_K=50
BLIP2_RERANK_K=10
BLIP2_TIMEOUT_SECONDS=180
```

## 4. Run Streamlit locally

Install local app dependencies:

```bash
pip install -r requirements-streamlit-cpu.txt
```

Use `requirements-streamlit-cpu.txt` on your laptop. It installs CPU-only PyTorch and avoids the huge CUDA downloads.

Run the app:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## 5. Demo flow

1. Click **Check BLIP-2 server** in the sidebar.
2. Upload a query image.
3. Confirm the YOLO crop, or use manual crop.
4. The local app runs CLIP and Pinecone search.
5. The local app sends only the cropped query image and candidate captions to the Kaggle BLIP-2 URL.
6. Results are shown locally with CLIP, BLIP-2, and final scores.

To switch to the other multi-seed run, change:

```bash
FINETUNED_SEED=541
CLIP_CHECKPOINT=/absolute/path/to/version_5/clip_best_seed541.pt
```

The sidebar server check warms the BLIP-2 model on Kaggle. Run it once before the first search so the first re-rank request does not spend its timeout loading model weights.

`CANDIDATE_K` controls how many Pinecone matches are fetched. `BLIP2_RERANK_K` controls how many of those matches are sent to Kaggle for the slower BLIP-2 pass. Keep `BLIP2_RERANK_K` at least as large as the number of results you plan to display, such as `10`, on a Kaggle T4. Results beyond `BLIP2_RERANK_K` remain CLIP-only fallback entries.

## YOLO weights note

`YOLO_MODEL_PATH` can be left empty:

```bash
YOLO_MODEL_PATH=
```

In that mode the app uses a center crop first, and you can turn on manual crop in the UI. If you want automatic YOLO cropping, set:

```bash
YOLO_MODEL_PATH=yolov8n.pt
```

Then install the optional YOLO dependency locally:

```bash
pip install -r requirements-yolo.txt
```

Ultralytics will download `yolov8n.pt` on first run if your local machine has internet access.
