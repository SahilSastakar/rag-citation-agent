"""
What this file does: Takes our chunks and their embeddings and stores them in Qdrant — our vector database. 
Also provides a method to search that database given a query vector, returning the most similar chunks. 
This is the "filing cabinet" from our earlier analogy — we're building it here.
"""

from typing import List
from loguru import logger
from qdrant_client import QdrantClient , models
from src.config import get_settings
from src.chunker import Chunk

class QdrantVectorStore:
    """Wrapper around Qdrant in-memory vector store """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = QdrantClient(":memory:")
        self.collection_name = self.settings.collection_name

        self._create_collection()

    def _create_collection(self):
        """
        Creates a collection in Qdrant to store vectors.
        
        - vectors_config specifies the dimensionality and distance metric.
          Here we use 768 dimensions (common for embedding models) and cosine similarity
          (good for semantic similarity).
        - We wrap this in a try/except block so the script doesn't crash if you run it multiple times:
          if the collection already exists, we just move on.
        """
        try:
            self.client.create_collection(
                collection_name = self.collection_name,
                vectors_config = models.VectorParams(size = self.settings.embedding_dimensions,
                distance=models.Distance.COSINE)
            )
            logger.info(f"Collection {self.collection_name} created successfully!")
        
        except:
            logger.info(f"Collection {self.collection_name} already exists...Continiuing!")

    
    def add_chunks(self , chunks: List[Chunk] , embeddings: List[List[float]]) -> None:
        logger.info(f"Adding {len(chunks)} no of chunks to the collection")

        points = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            data_pt = models.PointStruct(
                id = i,
                vector = embedding,
                payload = {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "chunk_index":chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                })
            points.append(data_pt)

        self.client.upsert(
            collection_name=self.collection_name,
            points = points
                
            )
        logger.info(f"Successfully added {len(points)} chunks")


    def search(self, query_vector: List[float], top_k: int) -> List[Chunk]:
        logger.info(f" searching {top_k} nearest neighbours for the query vector")


        
        results = self.client.query_points(   
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        ).points

        retrieved_chunks = []

        for result in results:
            retrieved_chunks.append(Chunk(
                text=result.payload["text"],
                chunk_index=result.payload["chunk_index"],
                source=result.payload.get("source"),
                page_number=result.payload.get("page_number"),
                total_chunks=result.payload.get("total_chunks", 0)
                
            ))

        logger.info(f"Successfully retrieved {len(retrieved_chunks)} chunks")

        return retrieved_chunks

        
        
    
        
                





        

    

