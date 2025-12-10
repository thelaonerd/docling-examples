from sitemap import get_sitemap_urls

from docling.document_converter import DocumentConverter
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
QDRANT_COLLECTION = "capstone_docling_site"


def main():
    qclient = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)

    if not qclient.collection_exists(QDRANT_COLLECTION):
        qclient.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=768, distance=models.Distance.COSINE),
        )

    oclient = ollama.Client(OLLAMA_HOST)
    doc_convertor = DocumentConverter()

    sitemap_urls = get_sitemap_urls(
        "https://docling-project.github.io/docling/")
    conv_results_iter = doc_convertor.convert_all(sitemap_urls)

    site_docs = []
    for result in conv_results_iter:
        if result.document:
            document = result.document
            site_docs.append(document)

    chunker = HybridChunker()
    for site_doc in site_docs:
        chunks = chunker.chunk(dl_doc=site_doc)

        documents = []
        metadatas = []
        for chunk in chunks:
            documents.append(chunk.text)
            metadatas.append(chunk.export_json_dict())

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
                    "metadata": metadatas[i]
                }
            ))

        qclient.upsert(
            collection_name=QDRANT_COLLECTION,
            points=upsert_points,
            wait=True,
        )


if __name__ == "__main__":
    main()
