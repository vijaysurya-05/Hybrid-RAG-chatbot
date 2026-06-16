# Hybrid-RAG-chatbot
An intelligent PDF Question-Answering application built with Streamlit that combines semantic vector search and keyword-based retrieval for accurate document understanding

## Workflow
Upload a PDF document.
Extract text from the PDF.
Split the document into chunks.
Generate embeddings using Hugging Face models.
Store embeddings in ChromaDB.
Retrieve relevant chunks using both BM25 and Vector Search.
Combine results through Hybrid Retrieval.
Generate answers using a Groq-hosted Large Language Model.

## Use Cases :-
Research paper analysis
Technical documentation querying
Academic study assistance
Report summarization
Enterprise knowledge retrieval