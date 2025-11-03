from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter


import ollama
import os
import uuid

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_EMBEDDING_MODEL = "embeddinggemma:latest"
OLLAMA_GEN_LLM_MODEL = "llama3.2:latest"
QDRANT_COLLECTION = "capstone_collection"


def main():
    qclient = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
    oclient = ollama.Client(OLLAMA_HOST)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Example chunk size
        chunk_overlap=200,  # Example overlap
        separators=[ "\n", " "]
    )

    file_list = [
                "2203.01017v2.md",
                "2206.01062.md",
                "2305.03393v1.md",
                "2408.09869v5.md",
                "2501.17887v1.md"
                 ]

    file_texts = []
    for file in file_list:
        with open(f"./scratch/{file}", "r", encoding="utf-8") as fp:
            file_texts.append(fp.read())

    file_chunks = []
    for text in file_texts:
        file_chunks.append(text_splitter.split_text(text))

    file_vectors = []
    for chunks in file_chunks:
        vectors = []
        for chunk in chunks:
            response = oclient.embeddings(model=OLLAMA_EMBEDDING_MODEL,prompt=chunk)
            vectors.append(response["embedding"])
        file_vectors.append(vectors)

    upsert_objects = list(zip(file_list, file_chunks, file_vectors))

    upsert_points = []

    for upsert_object in upsert_objects:
        file_name = upsert_object[0]
        for i in range(len(upsert_object[1])):
            upsert_points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=upsert_object[2][i],
                payload={
                    "document": upsert_object[1][i],
                    "metadata": file_name,
                },
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
        wait=True
    )


if __name__ == "__main__":
    main()
