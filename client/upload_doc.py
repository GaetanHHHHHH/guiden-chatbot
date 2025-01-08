import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
storage_account_name = ""
container_name = ""

files_path = Path("")
files = list(files_path.rglob('*'))
file = files[0]

blob_name = str(file).split('/')[-1]
try:
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)    
    container_client = blob_service_client.get_container_client(container_name)
    
    with open(file, "rb") as data:
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(data, overwrite=True)
    print(f"Successfully uploaded {blob_name} to container {container_name}")
    
except FileNotFoundError:
    print(f"Error: The file {file} was not found")
except Exception as e:
    print(f"An error occurred: {str(e)}")
