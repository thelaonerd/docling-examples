import logging
import os
import uuid
from pathlib import Path
from typing import Iterable, List

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import ollama

# --------------------------------------------------------------------------- #
# 1️⃣  Configuration & Validation
# --------------------------------------------------------------------------- #


class Config:
    """Load and validate all required environment variables."""

    def __init__(self) -> None:
        load_dotenv()  # Load from .env file if present
        self.qdrant_host = os.getenv("QDRANT_HOST")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.ollama_host = os.getenv("OLLAMA_HOST")
        self.embedding_model = "embeddinggemma:latest"
        self.collection_name = "capstone_docling_gpt_oss"

        missing = [k for k, v in self.__dict__.items() if v is None]
        if missing:
            raise ValueError(
                f"Missing required env vars: {', '.join(missing)}")


# --------------------------------------------------------------------------- #
# 2️⃣  Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 3️⃣  Client factory
# --------------------------------------------------------------------------- #


def create_clients(cfg: Config) -> tuple[QdrantClient, ollama.Client]:
    """Instantiate Qdrant and Ollama clients."""
    qclient = QdrantClient(url=cfg.qdrant_host, api_key=cfg.qdrant_api_key)
    oclient = ollama.Client(cfg.ollama_host)
    return qclient, oclient

# --------------------------------------------------------------------------- #
# 4️⃣  Document → chunk → embed → PointStruct pipeline
# --------------------------------------------------------------------------- #


def _process_documents(
    qclient: QdrantClient,
    oclient: ollama.Client,
    file_paths: Iterable[Path],
    cfg: Config
) -> None:
    """
    Convert Markdown files → Document → chunks → embeddings → upsert.
    """
    # 4.1  Convert files to Documents
    doc_convertor = DocumentConverter(allowed_formats=[InputFormat.MD])
    documents = [
        doc_convertor.convert(source=str(p)).document for p in file_paths
    ]

    embedding_model = cfg.embedding_model
    collection = cfg.collection_name

    # 4.2  Chunk each Document
    chunker = HybridChunker()
    chunked_docs = [chunker.chunk(dl_doc=doc) for doc in documents]

    # 4.3  Flatten chunks into a single list
    texts: List[str] = []
    metadatas: List[dict] = []
    for chunks in chunked_docs:
        for chunk in chunks:
            texts.append(chunk.text)
            metadatas.append(chunk.export_json_dict())

    # 4.4  Embed all chunks (synchronously)
    vectors: List[List[float]] = []
    for text in texts:
        try:
            resp = oclient.embeddings(model=embedding_model, prompt=text)
            vectors.append(resp["embedding"])
        except Exception as exc:  # pragma: no cover
            log.error("Embedding failed for chunk: %s – %s", text[:30], exc)
            vectors.append([0.0] * 768)  # fallback vector

    # 4.5  Build PointStructs
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={"document": texts[i], "metadata": metadatas[i]},
        )
        for i in range(len(vectors))
    ]

    # 4.6  Create collection if needed
    if not qclient.collection_exists(collection):
        qclient.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=768, distance=models.Distance.COSINE
            ),
        )

    # 4.7  Upsert with error handling
    try:
        qclient.upsert(
            collection_name=collection,
            points=points,
            wait=True,
        )
        log.info("Successfully upserted %d points into %s",
                 len(points), collection)
    except Exception as exc:  # pragma: no cover
        log.exception("Failed to upsert points into %s: %s", collection, exc)

# --------------------------------------------------------------------------- #
# 5️⃣  Main entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    cfg = Config()
    qclient, oclient = create_clients(cfg)

    # Use pathlib.Path for cross‑platform safety
    file_paths = [
        Path("./scratch/2203.01017v2.md"),
        Path("./scratch/2206.01062.md"),
        Path("./scratch/2305.03393v1.md"),
        Path("./scratch/2408.09869v5.md"),
        Path("./scratch/2501.17887v1.md"),
    ]

    _process_documents(qclient, oclient, file_paths,
                       cfg)


if __name__ == "__main__":
    main()
