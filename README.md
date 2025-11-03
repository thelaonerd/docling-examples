# Using flowise effectively with qdrant and docling

## Introduction

In this README we will see how to effectively use the docling `HybridChunker`, upsert the chunks to a qdrant cloud instance and use it effectively in Flowise AgentFlows.

The main files that can be used for reference are below
- `batch_convert.py` Used to convert all the PDF files to markdown using the docling engine.
- `qdrant_docling.py` Used to take the markdown documents created in the pervious step to, chunk, vectorize and upsert to a collection.

**Note**: I ran `batch_convert.py` on a CPU only instance on my laptop-server at home as the GPU conversion was freezing up my GPU workstation. All the development was done on a Linux server running Ubuntu 24.04 LTS version at my home.

## How to run this on your own setup

- Copy .env.example to .env file and fill out the parameters in .env file
- Ensure that you have access to ollama server. This is needed to use embeddinggemma:latest embedding model. Else you will find a suitable embedding model to use
- Create a python 3.12 or 3.13 version venv
- pip install -r requirements.txt # Need to migrate to uv later
- Optional: Run `batch_convert.py` to generate the markdown files again
- Run `qdrant_docling.py` to create chunks, vectorize and upsert to your qdrant instance. If you are not using ollama for embedding please follow the documentation to find the right steps. In the following block vector size needs to be adjusted from 768 to whatever size you are using.

```python
    if not qclient.collection_exists(QDRANT_COLLECTION):
        qclient.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=768, distance=models.Distance.COSINE),
        )
```

- Upload flowise agent 'Capstone Docling Agents.json' to flowise instance. I am using a locally hosted version on my laptop-server.
- Change the chat model, update the embedding end point, vector dimensions. Also note that here the key is to ensure that qdrant 'Content Key' is set to `document` and 'metadata' is set to `metaddata`. This is one reason why a lot of people couldn't get good results from the vectordb. This is defined in the following block of code in `qdrant_docling.py` where I am creating the upsert_points to be added to the vectordb
```python
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
```

Hope this helps.

> **PS:** I will create a github gist on this github account to share ways to selfhost the following apps locally and use a zero trust vpn tool to help you personally access it from any where over the internet using a secure tunnel
> - n8n
> - flowise
> - ollama
> - qdrant
> - docling server