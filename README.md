# Retrieval-Augmented Generation (RAG) with ChromaDB and OpenAI Embeddings

This repository contains a simple implementation of Retrieval-Augmented Generation (RAG) using ChromaDB and OpenAI embeddings.

RAG is a technique that combines the power of large language models (LLMs) with the ability to retrieve relevant information from a knowledge base.

## How to use this repository

1. Clone the repository: `git clone https://github.com/yourusername/rag-chromadb-openai.git`
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate` (on Windows, use `venv\Scripts\activate`)
4. Install the required packages: `pip install -r requirements.txt`
5. Rename `example.env` to `.env` and fill in your OpenAI API key.
6. Install Ollama models: `ollama install llama3.1:8b` - see [Ollama](https://ollama.ai/) for more information.
7. Run the script: `python local.py` to test the RAG system with a local LLM.
8. Run the script: `python open_ai.py` to test the RAG system with OpenAI embeddings.

## References

- [ChromaDB](https://docs.trychroma.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
