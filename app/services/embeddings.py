"""Embedding generation service"""
import time
import random
from typing import List
from openai import AzureOpenAI
from app.deps import get_openai_client, get_settings


def generate_embeddings(texts: List[str], max_retries: int = 3) -> List[List[float]]:
    """
    Generate embeddings for a list of texts with rate limit handling.
    
    Args:
        texts: List of text strings to embed
        max_retries: Maximum number of retry attempts
    
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    client = get_openai_client()
    settings = get_settings()
    
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                input=texts
            )
            return [d.embedding for d in response.data]
        
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limit error
            if "rate limit" in error_msg or "too many requests" in error_msg:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"[!] Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[!] Rate limit exceeded after {max_retries} attempts")
                    raise Exception(f"Rate limit exceeded. Please try again later. Original error: {str(e)}")
            else:
                # Non-rate-limit error, don't retry
                print(f"Error generating embeddings: {str(e)}")
                raise


def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text.
    
    Args:
        text: Text string to embed
    
    Returns:
        Embedding vector
    """
    return generate_embeddings([text])[0]

