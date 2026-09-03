"""
main.py — Entry point for the RAG Citation Agent.
Wires together the full pipeline:
1. Load and chunk a document
2. Embed the chunks
3. Store in Qdrant
4. Take a user query
5. Retrieve relevant chunks
6. Generate a cited answer
"""

import os
from pathlib import Path
from loguru import logger
from src.chunker import RecursiveChunker
from src.embedder import GeminiEmbedder
from src.vectorstore import QdrantVectorStore
from src.retriever import Retriever
from src.generator import Generator


def load_text_file(filepath: str) -> str:
    """
    Reads a plain text file and returns its contents as a string.
    We start with plain text files to keep ingestion simple.
    PDF support can be added later via pypdf.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if not path.suffix == ".txt":
        raise ValueError(f"Expected a .txt file, got: {path.suffix}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    logger.info(f"Loaded {len(text)} characters from {filepath}")
    return text


def ingest_document(filepath: str, vectorstore: QdrantVectorStore) -> int:
    """
    Full ingestion pipeline for one document:
    1. Load the text
    2. Chunk it
    3. Embed the chunks
    4. Store in the vectorstore

    Returns the number of chunks created.
    """
    logger.info(f"Starting ingestion for: {filepath}")

    # Step 1 — load
    text = load_text_file(filepath)

    # Step 2 — chunk
    chunker = RecursiveChunker()
    chunks = chunker.chunk_text(
        text=text,
        source=Path(filepath).name,
    )
    logger.info(f"Created {len(chunks)} chunks")

    # Step 3 — embed
    embedder = GeminiEmbedder()
    embeddings = embedder.embed_chunks(chunks)
    logger.info(f"Created {len(embeddings)} embeddings")

    # Step 4 — store
    vectorstore.add_chunks(chunks, embeddings)
    logger.info(f"Stored {len(chunks)} chunks in vectorstore")

    return len(chunks)


def ask(query: str, retriever: Retriever, generator: Generator) -> None:
    """
    Full query pipeline:
    1. Retrieve relevant chunks
    2. Generate a cited answer
    3. Print the answer and its citations
    """
    logger.info(f"Processing query: {query}")

    # Step 1 — retrieve
    chunks = retriever.retrieve(query)
    logger.info(f"Retrieved {len(chunks)} chunks")

    # Step 2 — generate
    cited_answer = generator.generate(query, chunks)

    # Step 3 — print results
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(cited_answer.answer)

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)
    for source in cited_answer.sources:
        print(
            f"[Source {source['source_number']}] "
            f"{source['filename']} "
            f"(Page {source['page']}, Chunk {source['chunk_index']})"
        )

    print("\n" + "=" * 60)
    print(f"Chunks used: {cited_answer.chunks_used}")
    print("=" * 60 + "\n")


def main():
    # --- SETUP ---
    logger.info("Initialising RAG Citation Agent")

    # Initialise the vectorstore once — shared across ingestion and retrieval
    vectorstore = QdrantVectorStore()

    # Initialise retriever and generator
    retriever = Retriever(vectorstore=vectorstore)
    generator = Generator()

    # --- INGESTION ---
    # Create a sample document in the data/ folder to test with
    sample_path = "data/sample.txt"

    if not os.path.exists(sample_path):
        logger.info("No sample document found — creating one")
        os.makedirs("data", exist_ok=True)
        with open(sample_path, "w") as f:
            f.write("""
Artificial Intelligence Safety

AI safety refers to the research area focused on ensuring that artificial 
intelligence systems behave in accordance with human intentions and values. 
As AI systems become more capable, the importance of safety research grows.

Key concepts in AI safety include alignment, which is the challenge of 
ensuring AI systems pursue goals that humans actually want. Misaligned AI 
systems might pursue proxy goals that diverge from true human intentions.

Interpretability is another critical area, focusing on understanding how 
AI models arrive at their outputs. Without interpretability, it is difficult 
to verify whether a model is reasoning correctly or for the right reasons.

Robustness refers to AI systems performing reliably across a wide range of 
inputs, including adversarial inputs designed to fool the system. A model 
that is not robust can fail in unexpected and dangerous ways in deployment.

AI governance covers the policies, regulations, and institutional structures 
needed to ensure AI development benefits humanity broadly. This includes 
questions of accountability, transparency, and equitable access to AI benefits.

Anthropic is an AI safety company founded in 2021. Its mission is the 
responsible development and maintenance of advanced AI for the long-term 
benefit of humanity. Anthropic conducts research into AI interpretability, 
alignment, and robustness.

Claude is an AI assistant developed by Anthropic. Claude is designed with 
safety as a core priority, including honesty, harmlessness, and helpfulness 
as guiding principles. Claude models are trained using Constitutional AI, 
a technique developed by Anthropic to make AI systems more aligned.
            """.strip())
        logger.info(f"Sample document created at {sample_path}")

    # Ingest the document
    num_chunks = ingest_document(sample_path, vectorstore)
    logger.info(f"Ingestion complete — {num_chunks} chunks ready for retrieval")

    # --- QUERIES ---
    queries = [
        "What is AI alignment and why does it matter?",
        "Who founded Anthropic and what is its mission?",
        "What is Constitutional AI?",
    ]

    for query in queries:
        ask(query, retriever, generator)


if __name__ == "__main__":
    main()