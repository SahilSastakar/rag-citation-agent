"""
What this file does: Orchestrates the full retrieval pipeline. Takes a user's query, 
embeds it, searches the vector store for candidates, and returns the most relevant chunks. 
Connects embedder.py and vectorstore.py together into one coherent retrieval system.
Implements multi-query retrieval to cast a wider net.
"""

import json
from typing import List
from loguru import logger
from google import genai
from src.config import get_settings
from src.chunker import Chunk
from src.embedder import GeminiEmbedder
from src.vectorstore import QdrantVectorStore


class Retriever:

    def __init__(self, vectorstore: QdrantVectorStore):
        self.settings = get_settings()
        self.embedder = GeminiEmbedder()
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        self.vectorstore = vectorstore

    def _generate_alt_queries(self, query: str) -> List[str]:
        logger.info(f"Generating alternative queries for: {query}")

        prompt = """
        You are an expert research assistant. Your task is to generate 3-5 
        alternative ways of phrasing the following user query. 
        Think like a researcher who would be searching for this information.

        Goal:
        - Keep the same meaning and intent.
        - Use different wording, synonyms, and sentence structures.
        - Make them more specific, more general, or rephrased in different ways.
        - Provide exactly 3-5 alternatives.

        User Query:
        {query}

        Return only a JSON list of strings. No explanations. No extra text.

        ["alternative query 1", "alternative query 2", "alternative query 3"]
        """.format(query=query)

        response = self.client.models.generate_content(
            model=self.settings.generation_model,
            contents=prompt
        )

        text = response.text
        alternatives = json.loads(text)
        alternatives.append(query)

        logger.info(f"Generated {len(alternatives)} alternative queries")
        return alternatives

    def _retrieve_single(self, query: str, top_k: int) -> List[Chunk]:
        embedding = self.embedder.embed_query(query)
        results = self.vectorstore.search(embedding, top_k)
        return results

    def _deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        seen = {}
        unique_chunks = []

        for chunk in chunks:
            key = f"{chunk.source}_{chunk.chunk_index}"

            if key not in seen:
                seen[key] = chunk
                unique_chunks.append(chunk)

        logger.info(f"Removed {len(chunks) - len(unique_chunks)} duplicate chunks")
        return unique_chunks

    def retrieve(self, query: str) -> List[Chunk]:
        logger.info(f"Starting retrieval for query: {query}")

        alt_queries = self._generate_alt_queries(query)

        all_chunks = []

        for alt_query in alt_queries:
            chunks = self._retrieve_single(alt_query, self.settings.top_k_retrieval)
            all_chunks.extend(chunks)

        unique_chunks = self._deduplicate(all_chunks)

        logger.info(f"Retrieval complete. {len(unique_chunks)} unique chunks from {len(all_chunks)} total candidates")
        return unique_chunks