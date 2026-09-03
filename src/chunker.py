#here is where the chunking mechanisms and strategies lie

from posixpath import sep
from dataclasses import dataclass , field
from typing import Optional
import tiktoken
from loguru import logger
from src.config import get_settings

@dataclass #decorator that automatically generates boilerplate code such as __init__ , __repr__ , __eq__ , etc
class Chunk:
    """Represents one individual chunk of text"""
    text: str
    chunk_index: int
    source: Optional[str] = None
    page_number: Optional[int] = None
    total_chunks: int = 0  
    metadata: dict = field(default_factory=dict)

class RecursiveChunker:
    """ Recursive text splitting strategies to chunk texts"""
    def __init__(self):
        self.settings = get_settings()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.separators = ["\n\n", "\n", ". ", " ", ""]
    """
    a list of separators in priority order. 
    We try to split at double newlines (paragraph boundaries) first. 
    If a piece is still too big, we try single newlines. Then sentence endings. 
    Then spaces. As a last resort, we split anywhere. 
    This is how we preserve natural language boundaries as much as possible.
    """
    def _count_tokens(self , text: str) -> int:
        """Tokenizes the text and returns the number of tokens using tiktoken"""
        return len(self.tokenizer.encode(text))

    def _split_text(self , text: str , separators: list[str]) -> list[str]:
        """
        recursively split text on the given separators until all chunks 
        are under the max_tokens limit
        """
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            splits = list(text)

        else:
            splits = text.split(separator)

        final_chunks = []
        curr_chunk = ""

        for split in splits:
            tentative = curr_chunk + separator + split if curr_chunk else split

            if self._count_tokens(tentative) <= self.settings.chunk_size:
                curr_chunk = tentative

            else:
                if curr_chunk:
                    final_chunks.append(curr_chunk)
                if self._count_tokens(split) > self.settings.chunk_size:
                    sub_chunks = self._split_text(split , remaining_separators)
                    final_chunks.extend(sub_chunks)
                    curr_chunk = ""

                else:
                    curr_chunk = split

        if curr_chunk:
            final_chunks.append(curr_chunk)

        return [c for c in final_chunks if c.strip()]

    def _add_overlap(self , chunks: list[str]) -> list[str]:
        if len(chunks) <= 1:
            return chunks

        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
                continue

            prev_chunk = chunks[i - 1]
            prev_tokens = self.tokenizer.encode(prev_chunk)
            overlap_tokens = prev_tokens[-self.settings.chunk_overlap:]
            overlap_text = self.tokenizer.decode(overlap_tokens)
            overlapped.append(overlap_text + " " + chunk)

        return overlapped

    def chunk_text(self,text: str,source: Optional[str] = None,page_number: Optional[int] = None) -> list[Chunk]:
        logger.info(f"Chunking text from source: {source}, length: {len(text)} chars")

        raw_chunks = self._split_text(text, self.separators)
        overlapped_chunks = self._add_overlap(raw_chunks)

        chunks = []
        for i, chunk_text in enumerate(overlapped_chunks):
            chunk = Chunk(
            text=chunk_text.strip(),
            chunk_index=i,
            source=source,
            page_number=page_number,
            total_chunks=len(overlapped_chunks),
            metadata={"source": source, "page": page_number}
        )
        chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks from source: {source}")
        return chunks               

        
        
        
        