import logging
import os
from typing import Tuple

import azure.functions as func
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.ai.formrecognizer._models import AnalyzeResult
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.documents import Document

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Log startup progress
logging.info("Starting function initialization...")

try:
    load_dotenv()
    logging.info("Environment variables loaded")

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_DEPLOYMENT_MODEL"),
        api_version="2024-02-01",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    logging.info("Model initialized")

    # Initialize services with connection testing
    document_client = DocumentAnalysisClient(
        endpoint=os.getenv("AZURE_DOC_INT_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_DOC_INT_API_KEY"))
    )
    logging.info("Document Intelligence client initialized")

    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-small",
        azure_endpoint=os.getenv("AZURE_OPENAI_EMBED_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_EMBED_API_KEY")
    )
    logging.info("Embeddings client initialized")

    vector_store = AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_SEARCH_SERVICE"),
        azure_search_key=os.environ["AZURE_SEARCH_API_KEY"],
        index_name="",
        embedding_function=embeddings.embed_query,
        search_options={
            "select": "id, content, metadata",  # Specify which fields to return
            "vector_fields": None  # Don't include vector fields in results
        },
        additional_search_client_options={
            "retry_total": 4,
            "connection_timeout": 5,
            "read_timeout": 30
        }
    )
    logging.info("Vector store initialized")

except Exception as e:
    logging.error(f"Error during function initialization: {str(e)}")
    raise


def process_blob_document(container_name: str, blob_name: str) -> AnalyzeResult:
    """
    Simulates the blob trigger function by processing a blob document file.
    This helps us test our document processing logic without needing Azure Functions.
    """
    blob_service_client = BlobServiceClient.from_connection_string(
            os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        )
    blob_client = blob_service_client.get_container_client(container_name).get_blob_client(blob_name)
    blob_content = blob_client.download_blob().readall()
    
    # Start the document analysis - notice we're using begin_analyze_document
    # instead of begin_analyze_document_from_url since we have a local file
    result = document_client.begin_analyze_document(
        "prebuilt-document",
        blob_content
    ).result() 
    logging.info(f"Processed document: {blob_name}")
    
    # Return the extracted text for further processing if needed
    return result


def chunk_blob_document(container_name: str, blob_name: str) -> Tuple[list, AnalyzeResult]:
    result = process_blob_document(container_name, blob_name)
    # Analyze the document
    
    document_chunks = []
    current_chunk = []
    current_length = 0
    target_chunk_size = 1000
    
    for paragraph in result.paragraphs:
        paragraph_text = paragraph.content
        
        # If adding this paragraph would exceed our target size
        if current_length + len(paragraph_text) > target_chunk_size and current_chunk:
            # Save the current chunk and start a new one
            document_chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
            
        current_chunk.append(paragraph_text)
        current_length += len(paragraph_text)
    
    # add the last chunk
    if current_chunk:
        document_chunks.append(" ".join(current_chunk))
    
    return document_chunks, result


def get_metadata(result: AnalyzeResult, blob_name: str, i: int, chunks: list) -> dict:
    metadata = {}
    # Get project structure
    metadata['section_titles'] = []
    for elem in result.to_dict()['paragraphs']:
        if elem['role'] in ['sectionHeading', 'title', 'heading']:
            metadata['section_titles'].append(elem['content'])

    # Add other metadata
    if len(metadata['section_titles']) > 0:
        metadata['title'] = metadata['section_titles'][0]
    else:
        metadata['title'] = ''
    metadata['file_type'] = blob_name.split('.')[-1]
    metadata['file_name'] = blob_name
    metadata['page'] = i + 1
    metadata['total_pages'] = len(chunks)

    logging.info(f'Metadata created for chunk {i}')

    return(metadata)


def embed_and_upload_blob_document(container_name: str, blob_name: str) -> None:
    chunks, result = chunk_blob_document(container_name, blob_name)
    logging.info(f"Document analysis result attributes: {dir(result)}")
    
    documents_to_upload = []
    for i, chunk in enumerate(chunks):
        # For the moment, the whole "table of content" is stored in the metadata of each chunk
        metadata = get_metadata(result, blob_name, i, chunks)
        
        doc = Document(
            page_content=chunk,
            metadata=metadata
        )
        documents_to_upload.append(doc)
        
        if len(documents_to_upload) >= 5:
            vector_store.add_texts(
                texts=[doc.page_content for doc in documents_to_upload],
                metadatas=[doc.metadata for doc in documents_to_upload]
            )
            documents_to_upload = []
            
    if documents_to_upload:
        vector_store.add_texts(
            texts=[doc.page_content for doc in documents_to_upload],
            metadatas=[doc.metadata for doc in documents_to_upload]
        )


@app.function_name(name="ProcessNewDocument")
@app.blob_trigger(arg_name="myblob", 
                 path="st-dataroots-guiden-pdfstorage/{name}",
                 connection="AzureWebJobsStorage")
def process_new_document(myblob: func.InputStream):
    logging.info(f"Processing new document: {myblob.name}")
    try:
        container_name, blob_name = myblob.name.split('/')[0], myblob.name.split('/')[-1]
        embed_and_upload_blob_document(container_name, blob_name)  
        logging.info(f"Successfully processed {myblob.name}")
    except Exception as e:
        logging.error(f"Error during processing of document {blob_name}: {str(e)}")
        raise
