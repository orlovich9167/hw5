MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"

COLLECTION_NAME = "squad_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

BATCH_SIZE_EMBED = 64
BATCH_SIZE_INSERT = 1000

DATA_DIR = "data"
DOCUMENTS_PATH = f"{DATA_DIR}/documents.json"
EMBEDDINGS_PATH = f"{DATA_DIR}/embeddings.npy"
BENCHMARK_RESULTS_PATH = f"{DATA_DIR}/benchmark_results.json"

MAX_DOCUMENTS = 5000

INDEX_PARAMS = {
    "IVF_FLAT": {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
        "search_params": {"nprobe": 16},
    },
    "HNSW": {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 16, "efConstruction": 200},
        "search_params": {"ef": 64},
    },
    "IVF_PQ": {
        "index_type": "IVF_PQ",
        "metric_type": "L2",
        "params": {"nlist": 128, "m": 16, "nbits": 8},
        "search_params": {"nprobe": 16},
    },
}

# --- Search Demo ---
SEARCH_COLLECTION_NAME = "squad_search_demo"
TFIDF_PATH = f"{DATA_DIR}/tfidf_vectorizer.json"
SEARCH_RESULTS_PATH = f"{DATA_DIR}/search_results.json"
TFIDF_MAX_FEATURES = 30000

SAMPLE_QUERIES = [
    "What is oxygen and how is it used?",
    "How does photosynthesis work in plants?",
    "Who was the first president of the United States?",
    "What causes earthquakes?",
    "How do computers process information?",
]