import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
import torch

print("torch:", torch.__version__)
print("torch file:", torch.__file__)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
emb = model.encode(["hello world"], convert_to_numpy=True, show_progress_bar=False)
print("ok", emb.shape)