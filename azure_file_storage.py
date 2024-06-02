import io 
import uuid
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from constants import (
    ACCOUNT_NAME,
    CONTAINER_NAME
)

class AzureFileStorage:
    def __init__(self):
        account_url = f"https://{ACCOUNT_NAME}.blob.core.windows.net"
        default_credential = DefaultAzureCredential()
        self.blob_service_client = BlobServiceClient(account_url, credential=default_credential)

    def upload(self, file_content, file_name):
        self.blob_client = self.blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=file_name)
        file_stream = io.BytesIO(file_content.encode('utf-8'))
        self.blob_client.upload_blob(file_stream, overwrite=True)

        return f'https://{ACCOUNT_NAME}.blob.core.windows.net/{CONTAINER_NAME}/{file_name}'
