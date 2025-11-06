import numpy as np
import faiss

class KnowledgeRetriever:
    def __init__(self, embeddings, texts):
        self.texts = texts
        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(np.array(embeddings, dtype="float32"))

    def search(self, query_embedding, top_k=3):
        D, I = self.index.search(np.array([query_embedding], dtype="float32"), top_k)
        return [self.texts[i] for i in I[0]]
