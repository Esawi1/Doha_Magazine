#!/usr/bin/env python3
"""
Recreate Azure Search Index
This script will delete and recreate the index with correct field settings
"""
import os
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchAlgorithmMetric,
    SearchableField
)
from azure.core.credentials import AzureKeyCredential

def recreate_index():
    """Delete and recreate the Azure Search index with correct settings"""
    
    # Load environment variables
    load_dotenv()
    
    # Get settings from environment
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "dm-articles")
    
    if not all([search_endpoint, search_key]):
        print("ERROR: Missing Azure Search credentials in .env file")
        return False
    
    # Create search client
    credential = AzureKeyCredential(search_key)
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
    
    try:
        # Delete existing index
        print(f"Deleting existing index '{index_name}'...")
        index_client.delete_index(index_name)
        print("Index deleted successfully")
        
    except Exception as e:
        print(f"Index '{index_name}' doesn't exist or couldn't be deleted: {str(e)}")
    
    # Create new index with correct settings
    print(f"Creating new index '{index_name}' with correct field settings...")
    
    # Define fields with retrievable=True for all fields
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True, sortable=True, retrievable=True),
        SearchableField(name="title", type=SearchFieldDataType.String, analyzer_name="ar.microsoft", retrievable=True),
        SearchableField(name="summary", type=SearchFieldDataType.String, analyzer_name="ar.microsoft", retrievable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="ar.microsoft", retrievable=True),
        SimpleField(name="url", type=SearchFieldDataType.String, filterable=True, facetable=False, retrievable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,  # Default for text-embedding-3-small
            vector_search_profile_name="dm-hnsw",
            retrievable=True
        ),
        SimpleField(name="lang", type=SearchFieldDataType.String, filterable=True, facetable=True, sortable=True, retrievable=True),
    ]
    
    # Vector search configuration
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(
            name="dm-hnsw",
            kind="hnsw",
            parameters={"m": 30, "efConstruction": 400, "metric": VectorSearchAlgorithmMetric.COSINE}
        )],
        profiles=[{
            "name": "dm-hnsw",
            "algorithm": "dm-hnsw"
        }]
    )
    
    # Create the index
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )
    
    index_client.create_index(index)
    print("Index created successfully with correct field settings")
    
    print("\nNext steps:")
    print("1. Re-index your articles with: python index_dm_articles.py")
    print("2. Test your chatbot - the search errors should be fixed!")
    
    return True

if __name__ == "__main__":
    print("Azure Search Index Recreation Tool")
    print("=" * 50)
    recreate_index()
