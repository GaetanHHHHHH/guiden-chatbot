import logging
import os
from typing import Optional
from typing_extensions import Annotated, List, TypedDict

import azure.functions as func
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import AzureSearch
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langgraph.graph import START, StateGraph

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

load_dotenv()
oai_model = os.getenv("AZURE_DEPLOYMENT_MODEL")

llm = AzureChatOpenAI(
    azure_deployment=oai_model,
    api_version="2024-02-01",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-small",
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBED_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_EMBED_API_KEY")
)

vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_SERVICE"),
    azure_search_key=os.environ["AZURE_SEARCH_API_KEY"],
    index_name="",
    embedding_function=embeddings.embed_query,
    additional_search_client_options={"retry_total": 4},
)

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use five sentences maximum. Keep the answer as concise as possible. Always say "Would you like more information?" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)


class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Optional[str],
        ...,
        "Project name from PDF source to filter by."
    ]


class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}

# Look into filter(s) later on (could be useful but niche, eg: filter = {"source": {"$regex": state["project"], "$options": "i"}})
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(
        state["question"],
        k=10,
        search_type="hybrid")

    return {"context": retrieved_docs}


def generate(state: State):
    context_pieces = []
    
    for doc in state["context"]:
        # Extract metadata
        metadata = doc.metadata
        source = metadata.get('source', 'Unknown source')
        section = metadata.get('section_titles', [])
        
        # Create a context piece (with metadata)
        context_piece = f"""
            Source: {source}
            Section: {', '.join(section) if section else 'General content'}
            Content: {doc.page_content}
        """
        context_pieces.append(context_piece)
    
    # Join all context pieces with clear separation
    full_context = "\n\n---\n\n".join(context_pieces)
    
    # Update your prompt to make use of the metadata
    messages = custom_rag_prompt.invoke({
        "question": state["question"], 
        "context": full_context
    })
    
    response = llm.invoke(messages)
    return {"answer": response.content}


@app.route(route="query_chatbot")
def query_chatbot(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    user_query = req.params.get('query')

    if not user_query:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            user_query = req_body.get('query')

    if user_query:
        graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
        graph_builder.add_edge(START, "analyze_query")
        graph = graph_builder.compile()

        result = graph.invoke({"question": user_query})

        formatted_context = '\n'.join([
                f"- {doc.metadata.get('file_name', 'Unknown')}, "
                f"page: {doc.metadata.get('page', 'Unknown')} out of {doc.metadata.get('total_pages', 'Unknown')}" 
                for doc in result["context"]
            ]) if result.get("context") else "No context available"

        response_text = (
            f"Answer: {result['answer']}\n\n"
            f"Sources:\n{formatted_context}"
        )
        
        return func.HttpResponse(
            body=response_text,
            mimetype="text/plain",
            status_code=200
        )
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a query in the query string or in the request body for a personalized response.",
             status_code=200
        )
