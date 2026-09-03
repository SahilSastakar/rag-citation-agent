"""
What this file does: Takes the retrieved chunks from retriever.py and uses the Gemini 
generation model to produce a final answer grounded in those chunks. 
Adds citation grounding — the answer tells you exactly which source and page 
each piece of information came from. This is the final step in the RAG pipeline.
"""

from typing import List
from loguru import logger
from google import genai
from google.genai import types
from src.config import get_settings
from src.chunker import Chunk
from dataclasses import dataclass


@dataclass
class CitedAnswer:
    answer: str
    sources: List[dict]
    chunks_used: int


class Generator:

    def __init__(self):
        self.settings = get_settings()
        self.client = genai.Client(api_key=self.settings.gemini_api_key)

    def _build_context(self, chunks: List[Chunk]) -> str:
        """
        Takes the retrieved chunks and formats them into a single context string
        that gets injected into the prompt. Each chunk is labelled with its source
        and page number so the model can cite them.
        """
        context_parts = []

        for i, chunk in enumerate(chunks):
            source = chunk.source or "Unknown"
            page = chunk.page_number or "N/A"

            context_parts.append(
                f"[Source {i + 1}: {source}, Page {page}]\n{chunk.text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(self, query: str, context: str) -> str:
        """
        Constructs the full prompt that gets sent to Gemini.
        The prompt instructs the model to answer only from the provided context
        and to cite its sources explicitly.
        """
        return f"""You are a precise and helpful research assistant.

Answer the user's question using ONLY the information provided in the context below.

Rules:
- Only use information that appears in the context. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so clearly.
- Cite your sources by referencing the Source number (e.g. [Source 1], [Source 2]).
- Be concise but complete.
- If multiple sources support a point, cite all of them.

Context:
{context}

User Question:
{query}

Answer:"""

    def _extract_sources(self, chunks: List[Chunk]) -> List[dict]:
        """
        Builds a clean list of unique sources from the chunks used.
        This becomes the citations section of the final answer.
        """
        seen = set()
        sources = []

        for i, chunk in enumerate(chunks):
            source_key = f"{chunk.source}_{chunk.page_number}"

            if source_key not in seen:
                seen.add(source_key)
                sources.append({
                    "source_number": i + 1,
                    "filename": chunk.source or "Unknown",
                    "page": chunk.page_number or "N/A",
                    "chunk_index": chunk.chunk_index,
                })

        return sources

    def generate(self, query: str, chunks: List[Chunk]) -> CitedAnswer:
        """
        Main public method. Takes a query and the retrieved chunks,
        builds a grounded prompt, calls Gemini, and returns a CitedAnswer
        with the answer text and its citations.
        """
        logger.info(f"Generating answer for query: {query}")
        logger.info(f"Using {len(chunks)} chunks as context")

        context = self._build_context(chunks)
        prompt = self._build_prompt(query, context)

        response = self.client.models.generate_content(
            model=self.settings.generation_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
            )
        )

        answer_text = response.text
        sources = self._extract_sources(chunks)

        logger.info(f"Answer generated successfully using {len(sources)} unique sources")

        return CitedAnswer(
            answer=answer_text,
            sources=sources,
            chunks_used=len(chunks)
        )