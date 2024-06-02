import os

PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', 'db_processes')
DOCUMENTS_FOLDER = os.environ.get('DOCUMENTS_FOLDER', 'source_documents')
EMBEDDINGS_MODEL = os.environ.get('EMBEDDINGS_MODEL', 'nomic-embed-text')
LLM_MODEL = os.environ.get('MODEL', 'llama3')

ACCOUNT_NAME = 'privategpt'
CONTAINER_NAME = 'c6d03ed8-2e3f-4cda-84cf-02a8f9d2b91c'
