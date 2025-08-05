import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


# Set path to your folder with PDFs
pdf_folder = "data/"

# Load all PDFs
all_docs = []
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, filename))
        docs = loader.load()
        all_docs.extend(docs)

#Chunk the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

split_docs = text_splitter.split_documents(all_docs)
print(f"Split into {len(split_docs)} chunks")

#embedding
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"  # or "hkunlp/instructor-xl"
)

# Store the embeddings in FAISS
vectorstore = FAISS.from_documents(split_docs, embedding_model)
vectorstore.save_local("faiss_index")
print("FAISS index saved")
print(f"Vectorstore has {vectorstore.index.ntotal} vectors")

# Load existing index
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Perform a search
query = "What are recent advancements for cancer treatment?"
results = vectorstore.similarity_search(query, k=3)

# Print top chunks
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:\n", doc.page_content)


llm = ChatGroq(
    groq_api_key=os.environ['GROQ_API_KEY'],
    model_name="llama-3.2-1b-preview",  
)

# Combine the top documents into one context string
context = "\n\n".join(doc.page_content for doc in results)

# Create a system prompt with that context
system_prompt = f"""You are a cancer research assistant.
Use the following context to answer the user's question.
Only use information from the context. If the answer is not in the context, say you don't know.

Context:
{context}
"""

# Create the full message chain
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=query)
]

# Invoke LLM
response = llm.invoke(messages)
print("\n\n Answer:\n", response.content)




