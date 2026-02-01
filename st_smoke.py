from sentence_transformers import SentenceTransformer
import torch

print("torch:", torch.__version__)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(["hello world"], convert_to_numpy=True)
print("ok", emb.shape)