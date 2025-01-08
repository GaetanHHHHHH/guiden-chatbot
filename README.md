# Guiden - RAG-Chatbot for guidelines and rules retrieval
## Introduction
This repo is the code used to create a RAG-chatbot on Azure. The goal of the project was to create a chatbot able to discuss the document stored in its vector store. The documents could be uploaded to Azure Blob Storage, which would trigger a first function to index the document automatically in the chatbot's vector store. You could then query the chatbot to discuss the document.

## Azure services needed
- Resource group (attached to a subscription)
- Azure OpenAI service with two deployments: an LLM model (eg. gpt-4o-mini), and an embedding model (eg. text-embedding-3-large)
- Azure AI Search: Used mainly for an Index
- Azure Functions: Two classic scalable functions (one for querying the model (HTTP trigger), one for uploading new documents and indexing them (blob trigger)). The code for these functions is available in the related folder
- Azure AI Services | Document intelligence: Form Recognizer used to process the PDF documents (more lightweight than other libraries like LangChain)
- Storage Account | Containers: A simple blob storage used for the documents before indexing

## Running the chatbot
1. Create all Azure resources (CLI / UI)
2. Create .env file with all needed env variables (endpoints, secrets, api keys, etc)
3. Query the LLM using the first function (simply call the script from the terminal)
4. Try uploading a file using the second script (same as before)
5. Ask the chatbot about the file to see if it's been indexed appropriately