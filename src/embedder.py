"""
What this file does: Takes a list of Chunk objects and 
converts each one's text into a vector (a list of numbers). 
Also converts a user's query string into a vector. Both use the 
same Gemini embedding model so they live in the same "meaning space" 
and can be compared.
"""

from typing import List
from src.chunker import Chunk
from loguru import logger
from google import genai
from src.config import get_settings

class GeminiEmbedder:
    def __init__(self):
        self.settings = get_settings()
        self.model = self.settings.embedding_model
        self.client = genai.Client(api_key=self.settings.gemini_api_key)

    def _embed_single(self , text: str) -> List[float]:
        """Takes a single string and returns its embedding as a list of floats."""

        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )

        return response.embeddings[0].values

    def embed_chunks(self, chunks: List[Chunk]) -> List[List[float]]:
        """takes a list of Chunk objects, extracts their text, embeds them in batch for efficiency."""

        logger.info(f"Embedding {len(chunks)} chunks")
        embeddings = []
    

        for i, chunk in enumerate(chunks):
            embedding = self._embed_single(chunk.text)         
            embeddings.append(embedding)
            
            if i % 10 == 0:
                logger.info(f"Embedded {i}/{len(chunks)}")
        logger.info("Embedding complete")

        return embeddings

    def embed_query(self , query: str) -> List[float]:
        """separate public method for embedding a user's question. 
        Functionally identical to _embed_single but kept separate because in some systems 
        you'd use different parameters for query vs document embedding. 
        Having separate methods makes that easy to change later without touching the rest of the code."""

        logger.info(f"Embedding user query {query}")
        user_inp = self._embed_single(query)
        
        return user_inp







