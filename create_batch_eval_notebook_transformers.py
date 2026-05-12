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
code_deps = """!pip install -q transformers open_clip_torch pinecone pandas Pillow tqdm scikit-learn accelerate"""

# Cell 3
code_imports = """import os
import torch
import open_clip
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from pinecone import Pinecone
from transformers import Blip2Processor, Blip2ForConditionalGeneration
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
print("Loading BLIP-2 Model for Re-ranking...")
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-6.7b", 
    device_map="balanced", 
    torch_dtype=torch.float16
)
blip_model.eval()

# CLIP loading helper
def load_clip(use_finetuned):
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    if use_finetuned:
        ckpt = torch.load(CLIP_CHECKPOINT, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval(), preprocess

print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
"""

# Cell 7
code_itm = """# ==============================================================================
# BLIP-2 ITM SCORING (Cross-Entropy Method)
# ==============================================================================
@torch.no_grad()
def compute_itm_score(query_image, candidate_caption):
    \"\"\"
    Computes a matching score using the model's loss when generating 
    the candidate caption given the query image. Lower loss = higher score.
    \"\"\"
    inputs = blip_processor(images=query_image, text=candidate_caption, return_tensors="pt").to(device, torch.float16)
    
    # We pass the input_ids as labels so the model computes the cross-entropy loss
    inputs["labels"] = inputs["input_ids"]
    
    outputs = blip_model(**inputs)
    loss = outputs.loss.item()
    
    # Return negative loss so that higher score means better match
    return -loss
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
            scored_candidates = []
            for cand in candidates:
                cand_image_name = cand['metadata']['image_name']
                cand_caption = caption_map.get(cand_image_name, "")
                
                # Get ITM Score
                itm_score = compute_itm_score(raw_image, cand_caption)
                scored_candidates.append((cand, itm_score))
                
            # Re-sort candidates by ITM score (descending)
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

with open('batch_evaluation_transformers.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully as batch_evaluation_transformers.ipynb")
