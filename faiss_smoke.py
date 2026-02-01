import faiss
import numpy as np

x = np.random.rand(10, 5).astype("float32")
faiss.normalize_L2(x)
idx = faiss.IndexFlatIP(5)
idx.add(x)
D, I = idx.search(x[:1], 3)
print("ok", D, I)
print("faiss file:", faiss.__file__)