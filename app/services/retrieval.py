"""Retrieval service for hybrid search"""
from typing import List, Dict, Optional
from azure.search.documents.models import VectorizedQuery
from app.deps import get_search_client, get_settings
from app.services.embeddings import generate_embedding


def hybrid_search(query: str, top_k: int = 8, min_score: Optional[float] = None) -> List[Dict]:
    """
    Perform hybrid search (vector + keyword) with semantic ranking.
    
    Args:
        query: User query text
        top_k: Number of results to return
        min_score: Minimum score threshold (optional)
    
    Returns:
        List of search result dicts
    """
    client = get_search_client()
    settings = get_settings()
    
    # Generate query embedding
    query_vector = generate_embedding(query)
    
    # Perform hybrid search
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector"
    )
    
    try:
        results = client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top_k,
            select=["id", "title", "url", "summary", "content"]
        )
        
        docs = []
        for result in results:
            score = result.get("@search.score", 0.0)
            
            # Apply min score filter if specified
            if min_score and score < min_score:
                continue
            
            docs.append({
                "id": result.get("id"),
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "summary": result.get("summary", ""),
                "content": result.get("content", ""),
                "score": score
            })
        
        return docs
    
    except Exception as e:
        print(f"[!] Hybrid search error: {str(e)}")
        return []




def format_context_for_llm(search_results: List[Dict], max_tokens: int = 1200) -> str:
    """
    Format search results into context string for LLM.
    
    Args:
        search_results: List of search result dicts
        max_tokens: Maximum tokens for context (rough word limit)
    
    Returns:
        Formatted context string
    """
    if not search_results:
        return "No relevant information found."
    
    context_parts = []
    token_count = 0
    
    for i, doc in enumerate(search_results, 1):
        title = doc.get("title", "Unknown")
        url = doc.get("url", "")
        content = doc.get("content", doc.get("summary", ""))
        
        # Estimate tokens (rough: 1 token ≈ 0.75 words)
        chunk_words = len(content.split())
        if token_count + chunk_words > max_tokens:
            break
        
        context_parts.append(f"[{i}] {title}\n{content}\nSource: {url}\n")
        token_count += chunk_words
    
    return "\n".join(context_parts)

