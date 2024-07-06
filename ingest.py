import chromadb
from llama_index.core import (
    VectorStoreIndex, 
    Settings, 
    Document,
    StorageContext,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from constants import (
    DOCUMENTS_FOLDER,
    PERSIST_DIRECTORY,
    LLM_MODEL,
    EMBEDDINGS_MODEL,
)
from documents_info import docs_info
from utils import pdf_to_text

# configure LLM and embeddings
llm = Ollama(model=LLM_MODEL, request_timeout=360.0, temperature=0.4)
embed_model = OllamaEmbedding(model_name=EMBEDDINGS_MODEL)
db_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# configure settings for the vector store
Settings.embed_model = embed_model
Settings.llm = llm

def process_documents():
    collections = []
    for doc in docs_info:
        file_name = doc['file_name']
        collection_name = doc['collection']

        print('Processing:', file_name)
        pdf_url = f"{DOCUMENTS_FOLDER}/{file_name}" 

        collection = db_client.create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        doc = pdf_to_text(pdf_url)
        chunks = []
        for page in doc:
            chunks.append(page)
        overlapped_chunks = add_overlap(chunks, 500)

        documents = []
        for chunk in overlapped_chunks:
            documents.append(Document(text=chunk, metadata={'file_name': file_name}))

        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, embed_model=Settings.embed_model
        )

        print('Documents processed successfully!')

def add_overlap(chunks, overlap_size):
    overlapped_chunks = []

    for i in range(len(chunks)):
        if i > 0:
            overlap_part_start = chunks[i-1][-overlap_size:]
        else:
            overlap_part_start = ""

        if i < len(chunks) - 1:
            overlap_part_end = chunks[i][:overlap_size]
        else:
            overlap_part_end = ""

        new_chunk = overlap_part_start + chunks[i] + overlap_part_end
        overlapped_chunks.append(new_chunk)

    return overlapped_chunks

if __name__ == "__main__":
    process_documents()
