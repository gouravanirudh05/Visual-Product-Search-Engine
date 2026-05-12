import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1
markdown_1 = """# Batch Evaluation & Semantic Re-ranking
This notebook runs the end-to-end evaluation of the Visual Product Search Engine on Kaggle.
It computes Recall@K, NDCG@K, and mAP@K for the 3 ablation conditions:
- **A**: Vision-only CLIP ($\alpha=1.0$)
- **B**: Frozen CLIP + Frozen BLIP-2
- **C**: Fine-tuned CLIP + Frozen BLIP-2
"""

# Cell 2
code_deps = """!pip install -q omegaconf timm fairscale iopath decord webdataset pycocotools pycocoevalcap
!pip install -q --no-dependencies salesforce-lavis
!pip install -q transformers==4.38.2 open_clip_torch pinecone pandas Pillow tqdm scikit-learn accelerate"""

# Cell 3
code_imports = """import os
import torch
import open_clip
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from pinecone import Pinecone
from lavis.models import load_model_and_preprocess
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")"""

# Cell 4
code_config = """# ==============================================================================
# CONFIGURATIONS
# ==============================================================================
PINECONE_API_KEY  = "YOUR_PINECONE_API_KEY"  # Replace with your key
INDEX_NAME        = "vr-clothing-gallery"

DATASET_ROOT      = "/kaggle/input/datasets/vinay1706/deepfashion-cropped" # Verify path
CLIP_CHECKPOINT   = f"{DATASET_ROOT}/clip_best.pt"
QUERY_CSV         = f"{DATASET_ROOT}/query.csv"
GALLERY_CSV       = f"{DATASET_ROOT}/gallery.csv"
CAPTIONS_CSV      = f"{DATASET_ROOT}/blip2_captions_gallery.csv"
IMAGE_ROOT        = DATASET_ROOT

# Top-K settings
TOP_K_RETRIEVAL = 15
K_VALUES = [5, 10, 15]

# Ablation Configs to Evaluate: (Config Name, Namespace, Use Finetuned CLIP, Use BLIP-2 Reranking)
CONFIGS = [
    ("Config A (Vision-Only)", "frozen-alpha-1.0", False, False),
    ("Config B (Frozen CLIP, alpha=0.7)", "frozen-alpha-0.7", False, True),
    ("Config B (Frozen CLIP, alpha=0.5)", "frozen-alpha-0.5", False, True),
    ("Config C (Finetuned CLIP, alpha=0.7)", "finetuned-alpha-0.7", True, True),
    ("Config C (Finetuned CLIP, alpha=0.5)", "finetuned-alpha-0.5", True, True)
]
"""

# Cell 5
code_data = """# ==============================================================================
# LOAD DATA
# ==============================================================================
query_df = pd.read_csv(QUERY_CSV)
gallery_df = pd.read_csv(GALLERY_CSV)
captions_df = pd.read_csv(CAPTIONS_CSV)

# Create a mapping from image_name to caption for the gallery
caption_map = dict(zip(captions_df['image_name'], captions_df['blip2_caption']))

print(f"Loaded {len(query_df)} query images.")
print(f"Loaded {len(gallery_df)} gallery images.")
"""

# Cell 6
code_models = """# ==============================================================================
# LOAD MODELS
# ==============================================================================
print("Loading LAVIS BLIP-2 ITM Model...")
blip_model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_image_text_matching", 
    model_type="pretrain", 
    is_eval=True, 
    device=device
)

# Fix LAVIS missing padding in batched inference
class PatchedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, *args, **kwargs):
        kwargs["padding"] = True
        return self.tokenizer(*args, **kwargs)
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

blip_model.tokenizer = PatchedTokenizer(blip_model.tokenizer)

# CLIP loading helper
def load_clip(use_finetuned):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    if use_finetuned:
        ckpt = torch.load(CLIP_CHECKPOINT, map_location=device)
        # BUG FIX: The checkpoint uses the key 'model_state', not 'model_state_dict'
        state_dict = ckpt.get("model_state", ckpt)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval(), preprocess

print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
"""

# Cell 7
code_itm = """# ==============================================================================
# BLIP-2 ITM SCORING (LAVIS Batch Method)
# ==============================================================================
def compute_itm_scores_batched(raw_image, candidate_captions):
    \"\"\"
    Computes true ITM probabilities for a batch of candidate captions against a single image.
    Returns a list of probabilities (0.0 to 1.0).
    \"\"\"
    # Preprocess image once
    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    
    # Duplicate image tensor to match the batch size of captions
    img_batch = img.repeat(len(candidate_captions), 1, 1, 1)
    
    # Preprocess all candidate text captions
    txt_batch = [txt_processors["eval"](c) for c in candidate_captions]
    
    # Pass through the ITM model in a single batch
    with torch.no_grad():
        itm_output = blip_model({"image": img_batch, "text_input": txt_batch}, match_head="itm")
        
        # The model outputs logits for [Not Match, Match]. 
        # We take the softmax and extract the probability of Match (index 1).
        itm_scores = torch.nn.functional.softmax(itm_output, dim=1)[:, 1].cpu().tolist()
        
    return itm_scores
"""

# Cell 8
code_metrics = """# ==============================================================================
# METRICS CALCULATION
# ==============================================================================
def calculate_metrics(ranked_item_ids, ground_truth_id, k_values=[5, 10, 15]):
    metrics = {}
    for k in k_values:
        top_k_items = ranked_item_ids[:k]
        
        # Recall@K: 1 if ground truth is in top-K, else 0
        recall = 1 if ground_truth_id in top_k_items else 0
        metrics[f'Recall@{k}'] = recall
        
        # NDCG@K
        dcg = 0
        for i, item in enumerate(top_k_items):
            if item == ground_truth_id:
                dcg += 1 / math.log2(i + 2)
                break
                
        # Ideal DCG is 1 because we only consider the first matching item
        idcg = 1 
        metrics[f'NDCG@{k}'] = dcg / idcg
        
        # mAP@K (Average Precision)
        # Since there is only 1 relevant item, AP is just 1/(rank) if found, else 0
        ap = 0
        for i, item in enumerate(top_k_items):
            if item == ground_truth_id:
                ap = 1 / (i + 1)
                break
        metrics[f'mAP@{k}'] = ap
        
    return metrics
"""

# Cell 9
code_eval = """# ==============================================================================
# MAIN EVALUATION LOOP
# ==============================================================================

# Subset for testing (set to None for full evaluation)
NUM_QUERIES_TO_EVALUATE = 50 

if NUM_QUERIES_TO_EVALUATE:
    eval_df = query_df.head(NUM_QUERIES_TO_EVALUATE)
else:
    eval_df = query_df

results_summary = []

for config_name, namespace, use_finetuned, use_reranking in CONFIGS:
    print(f"\\n{'='*60}\\nEvaluating {config_name}\\n{'='*60}")
    
    # Load appropriate CLIP model
    clip_model, clip_preprocess = load_clip(use_finetuned)
    
    config_metrics = {f'Recall@{k}': [] for k in K_VALUES}
    config_metrics.update({f'NDCG@{k}': [] for k in K_VALUES})
    config_metrics.update({f'mAP@{k}': [] for k in K_VALUES})
    
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        query_image_path = os.path.join(IMAGE_ROOT, row['image_name'])
        gt_item_id = row['item_id']
        
        # 1. Load and process query image
        try:
            raw_image = Image.open(query_image_path).convert('RGB')
            # Assuming images are already cropped in this dataset.
            processed_image = clip_preprocess(raw_image).unsqueeze(0).to(device)
        except Exception as e:
            continue
            
        # 2. Get CLIP Visual Embedding
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                query_embedding = clip_model.encode_image(processed_image)
                query_embedding = torch.nn.functional.normalize(query_embedding, dim=-1).cpu().numpy().tolist()[0]
                
        # 3. Retrieve Top-K candidates from Pinecone
        retrieval_response = index.query(
            vector=query_embedding,
            top_k=TOP_K_RETRIEVAL,
            namespace=namespace,
            include_metadata=True
        )
        
        candidates = retrieval_response['matches']
        
        # 4. Semantic Re-ranking
        if use_reranking:
            cand_images = [cand['metadata']['image_name'] for cand in candidates]
            cand_captions = [caption_map.get(img, "") for img in cand_images]
            
            # Get ITM Scores in one batch
            itm_scores = compute_itm_scores_batched(raw_image, cand_captions)
            
            scored_candidates = list(zip(candidates, itm_scores))
                
            # Re-sort candidates by ITM score (descending probability)
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = [item[0] for item in scored_candidates]
            
        # 5. Extract Ranked Item IDs
        ranked_item_ids = [cand['metadata']['item_id'] for cand in candidates]
        
        # 6. Calculate Metrics
        q_metrics = calculate_metrics(ranked_item_ids, gt_item_id, K_VALUES)
        
        for k, v in q_metrics.items():
            config_metrics[k].append(v)
            
    # Aggregate and print results
    print(f"\\nResults for {config_name}:")
    summary_row = {"Config": config_name}
    for metric, values in config_metrics.items():
        mean_val = np.mean(values)
        summary_row[metric] = mean_val
        print(f"{metric}: {mean_val:.4f}")
        
    results_summary.append(summary_row)
    
    # Clean up CLIP model memory before next config
    del clip_model
    torch.cuda.empty_cache()

# Display Final Summary Table
results_df = pd.DataFrame(results_summary)
display(results_df)
"""

nb.cells = [
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_code_cell(code_deps),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_config),
    nbf.v4.new_code_cell(code_data),
    nbf.v4.new_code_cell(code_models),
    nbf.v4.new_code_cell(code_itm),
    nbf.v4.new_code_cell(code_metrics),
    nbf.v4.new_code_cell(code_eval)
]

with open('batch_evaluation.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully as batch_evaluation.ipynb")
