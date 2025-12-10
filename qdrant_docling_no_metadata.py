from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from qdrant_client import QdrantClient, models

from dotenv import load_dotenv

import ollama
import os
import uuid


load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = None
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
EMBEDDING_MODEL = "embeddinggemma:latest"
QDRANT_COLLECTION = "capstone_docling_no_metdata"


def main():
    qclient = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
    oclient = ollama.Client(OLLAMA_HOST)

    file_paths = [
        "./scratch/2203.01017v2.md",
        "./scratch/2206.01062.md",
        "./scratch/2305.03393v1.md",
        "./scratch/2408.09869v5.md",
        "./scratch/2501.17887v1.md"
    ]

    doc_convertor = DocumentConverter(allowed_formats=[InputFormat.MD])
    chunker_input = []
    for file in file_paths:
        chunker_input.append(doc_convertor.convert(source=file).document)

    chunker = HybridChunker()
    chunked_docs = []
    for input_doc in chunker_input:
        chunked_docs.append(chunker.chunk(dl_doc=input_doc))

    documents = []
    for chunks in chunked_docs:
        for chunk in chunks:
            documents.append(chunk.text)

    vectors = []
    for document in documents:
        response = oclient.embeddings(
            model=EMBEDDING_MODEL,
            prompt=document)
        vectors.append(response["embedding"])

    upsert_points = []
    for i in range(len(vectors)):
        upsert_points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={
                "document": documents[i],
            }
        ))

    if not qclient.collection_exists(QDRANT_COLLECTION):
        qclient.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=768, distance=models.Distance.COSINE),
        )

    qclient.upsert(
        collection_name=QDRANT_COLLECTION,
        points=upsert_points,
        wait=True,
    )


if __name__ == "__main__":
    main()
