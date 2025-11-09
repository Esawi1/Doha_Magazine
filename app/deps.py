"""Shared dependencies and configurations"""
import os
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv

# Azure SDK
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# Cosmos DB (optional - can use postgres instead)
try:
    from azure.cosmos import CosmosClient
    COSMOS_AVAILABLE = True
except ImportError:
    COSMOS_AVAILABLE = False

load_dotenv()


class Settings:
    """Application settings from environment variables"""
    
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = os.environ.get("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_API_VERSION: str = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "")
    
    # Azure Search
    AZURE_SEARCH_ENDPOINT: str = os.environ.get("AZURE_SEARCH_ENDPOINT", "")
    AZURE_SEARCH_API_KEY: str = os.environ.get("AZURE_SEARCH_API_KEY", "")
    AZURE_SEARCH_INDEX_NAME: str = os.environ.get("AZURE_SEARCH_INDEX_NAME", "dm-articles")
    EMBED_DIM: int = int(os.environ.get("EMBED_DIM", "1536"))
    
    # Cosmos DB (or use Postgres)
    COSMOS_ENDPOINT: str = os.environ.get("COSMOS_ENDPOINT", "")
    COSMOS_KEY: str = os.environ.get("COSMOS_KEY", "")
    COSMOS_DB: str = os.environ.get("COSMOS_DB", "dm_chat")
    COSMOS_CONTAINER_MESSAGES: str = os.environ.get("COSMOS_CONTAINER", "messages")
    COSMOS_CONTAINER_FEEDBACK: str = "feedback"
    
    
    # App settings
    MAX_TOKENS_CONTEXT: int = 1200
    RETRIEVAL_TOP_K: int = 8
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 80
    
    # Conversation orchestration settings
    MAX_CONVERSATION_HISTORY: int = 10  # Messages to keep in context
    CONVERSATION_MEMORY_TURNS: int = 5  # Turns to include in prompt
    ENABLE_QUERY_ENHANCEMENT: bool = True  # Enhance queries with context
    MIN_RELEVANCE_SCORE: float = 0.4  # Minimum retrieval score


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


@lru_cache()
def get_openai_client() -> AzureOpenAI:
    """Get cached Azure OpenAI client"""
    settings = get_settings()
    return AzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION
    )


@lru_cache()
def get_search_client() -> SearchClient:
    """Get cached Azure AI Search client"""
    settings = get_settings()
    return SearchClient(
        endpoint=settings.AZURE_SEARCH_ENDPOINT,
        index_name=settings.AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY)
    )




@lru_cache()
def get_cosmos_client() -> Optional['CosmosClient']:
    """Get cached Cosmos DB client (if available)"""
    if not COSMOS_AVAILABLE:
        return None
    
    settings = get_settings()
    if not settings.COSMOS_ENDPOINT or not settings.COSMOS_KEY:
        return None
    
    return CosmosClient(settings.COSMOS_ENDPOINT, settings.COSMOS_KEY)


 
