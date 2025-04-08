# Chatbot
# TumorTalk Chatbot

TumorTalk is an AI-powered chatbot designed to assist with cancer research by answering questions based on provided PDF documents. It uses advanced language models and vector search to retrieve and generate accurate responses from the context of the documents.

## Features
- **PDF Document Parsing**: Automatically loads and processes PDF files from a specified folder.
- **Text Chunking**: Splits documents into manageable chunks for efficient processing.
- **Vector Search**: Uses FAISS for similarity search to retrieve the most relevant document chunks.
- **Language Model Integration**: Leverages Groq-powered language models for generating responses.
- **Customizable Embeddings**: Supports HuggingFace embeddings for document vectorization.

## Dependencies
- langchain
- langchain-community
- langchain-groq
- transformers
- torch
- faiss-cpu
- python-dotenv

