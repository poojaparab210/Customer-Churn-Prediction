
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "rag" / "retention_docs"
DB_DIR = BASE_DIR / "outputs" / "chroma_db"

MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading embedding model...")
embedding_model = SentenceTransformer(MODEL_NAME)

client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name="retention_knowledge",
    metadata={"hnsw:space": "cosine"},
)


def split_into_chunks(text: str, chunk_size: int = 120, overlap: int = 25):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def build_knowledge_base():
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Document folder not found: {DOCS_DIR}")

    documents = []
    metadatas = []
    ids = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        for index, chunk in enumerate(split_into_chunks(text)):
            documents.append(chunk)
            metadatas.append(
                {"source": file_path.name, "chunk": index}
            )
            ids.append(f"{file_path.stem}-{index}")

    if not documents:
        raise ValueError(
            f"No non-empty .txt documents found in {DOCS_DIR}"
        )

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Indexed {len(documents)} text chunks.")


def retrieve_context(question: str, top_k: int = 3):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if collection.count() == 0:
        raise RuntimeError(
            "Knowledge base is empty. Run this script to index documents first."
        )

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append(
            {
                "text": document,
                "source": metadata["source"],
                "chunk": metadata["chunk"],
                "distance": round(float(distance), 4),
            }
        )

    return retrieved


if __name__ == "__main__":
    build_knowledge_base()

    question = (
        "What should we investigate for a high-risk customer "
        "on a month-to-month contract?"
    )

    print("\nRetrieval test:")
    for item in retrieve_context(question):
        print(f"\nSource: {item['source']} | Chunk: {item['chunk']}")
        print(f"Distance: {item['distance']}")
        print(item["text"])
