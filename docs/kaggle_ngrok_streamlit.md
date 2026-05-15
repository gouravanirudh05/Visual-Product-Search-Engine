# Kaggle BLIP-2 + local Streamlit setup

Run BLIP-2 as a separate Kaggle GPU service and keep Streamlit local on your laptop.

For the full step-by-step version, see `docs/local_streamlit_kaggle_blip.md`.

## 1. Start the BLIP-2 server on a GPU runtime

```bash
pip install -r requirements-blip2-server.txt
pip install --no-deps salesforce-lavis
```

The remote service uses the same LAVIS BLIP-2 ITM scorer as `finetuned-multiseed-eval.ipynb` by default. Set these before launch:

```bash
export BLIP2_BACKEND="lavis_itm"
export BLIP2_FINAL_SCORE_MODE="blip2"
export BLIP2_BATCH_SIZE=16
```

Then start the service:

```bash
uvicorn remote_server.blip2_service:app --host 0.0.0.0 --port 8001
```

Expose it with ngrok and copy the public URL:

```python
from pyngrok import ngrok
print(ngrok.connect(8001).public_url)
```

Or launch both uvicorn and ngrok together:

```bash
python remote_server/run_blip2_ngrok.py
```

## 2. Start Streamlit locally

Set the project paths and service URLs first:

```bash
export BLIP2_SERVER_URL="https://YOUR-BLIP2-SERVER.ngrok-free.app"
export PINECONE_API_KEY="YOUR_KEY"
export PINECONE_INDEX_NAME="vr-clothing-gallery"
export GALLERY_CSV="/local/path/to/version_5/gallery.csv"
export CAPTIONS_CSV="/local/path/to/version_5/blip2_captions_gallery.csv"
export IMAGE_ROOT="/local/path/to/version_5"
export FINETUNED_SEED="104"
export FINETUNED_ALPHA="0.7"
export CLIP_CHECKPOINT="/local/path/to/version_5/clip_best_seed104.pt"
```

Install and run:

```bash
pip install -r requirements-streamlit.txt
streamlit run app.py
```

## Expected demo flow

1. Upload a product query image.
2. The app runs YOLO and displays the cropped product.
3. Click **Confirm crop**.
4. The app encodes the crop with CLIP, queries Pinecone/HNSW, sends candidate captions to the remote BLIP-2 service, and displays the top-K ranked products with CLIP, BLIP-2, and final scores.
