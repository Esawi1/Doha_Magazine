import os, re, json, uuid
from dotenv import load_dotenv
from typing import List, Dict
from pathlib import Path

# Azure Search SDK
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchAlgorithmMetric,
    SearchableField
)
from azure.core.credentials import AzureKeyCredential

# Azure OpenAI embeddings
from openai import AzureOpenAI

load_dotenv()

# ---- Config ----
AOAI_ENDPOINT  = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_KEY       = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_EMB_DEP   = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]  # e.g., text-embedding-3-small / -large

SEARCH_ENDPOINT   = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_KEY        = os.environ["AZURE_SEARCH_API_KEY"]
INDEX_NAME        = os.environ.get("AZURE_SEARCH_INDEX_NAME", "dm-articles")

# derive dims safely (or allow override via ENV)
_model = AOAI_EMB_DEP.lower()
EMBED_DIM = int(os.getenv("EMBED_DIM") or (3072 if "3-large" in _model else 1536))

from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchAlgorithmMetric,
    SearchableField
)

TXT_PATH   = Path("doha_magazine_articles.txt")
JSON_PATH  = Path("doha_magazine_articles.json")  # if you created it

# ---- Clients ----
search_index_client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))
search_client       = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))
aoai_client         = AzureOpenAI(azure_endpoint=AOAI_ENDPOINT, api_key=AOAI_KEY, api_version="2024-06-01")

# ---- Helpers ----
def clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", x).strip()

def chunk_text(text: str, max_tokens: int = 700, overlap_tokens: int = 80) -> List[str]:
    """
    Simple token-ish chunker by words. For Arabic/English mixed text it works fine.
    Adjust sizes as you like; embeddings tolerate ~8k tokens for text-embedding-3-*,
    but smaller chunks are better for retrieval.
    """
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    step = max_tokens - overlap_tokens
    while i < len(words):
        chunk = " ".join(words[i:i+max_tokens])
        chunks.append(chunk)
        i += step
    return chunks

def embed(texts: List[str]) -> List[List[float]]:
    resp = aoai_client.embeddings.create(
        model=AOAI_EMB_DEP,
        input=texts
    )
    return [d.embedding for d in resp.data]

def ensure_index():
    required_fields = {"id", "title", "summary", "content", "url", "content_vector", "lang"}
    try:
        existing = search_index_client.get_index(INDEX_NAME)
        # basic validation
        field_names = {f.name for f in existing.fields}
        if not required_fields.issubset(field_names):
            print(f"[!] Index '{INDEX_NAME}' exists but missing fields: {required_fields - field_names}")
        # check vector dims
        vec_field = next((f for f in existing.fields if f.name == "content_vector"), None)
        if hasattr(vec_field, "vector_search_dimensions") and vec_field.vector_search_dimensions != EMBED_DIM:
            print(f"[!] WARNING: vector dim mismatch (index={vec_field.vector_search_dimensions}, expected={EMBED_DIM})")
        print(f"[i] Index '{INDEX_NAME}' exists.")
        return
    except Exception:
        pass

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True, sortable=True),
        SearchableField(name="title",   type=SearchFieldDataType.String, analyzer_name="ar.microsoft"),
        SearchableField(name="summary", type=SearchFieldDataType.String, analyzer_name="ar.microsoft"),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="ar.microsoft"),
        SimpleField(name="url", type=SearchFieldDataType.String, filterable=True),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBED_DIM,
            vector_search_configuration="dm-hnsw",
        ),
        SimpleField(name="lang", type=SearchFieldDataType.String, filterable=True, facetable=True, sortable=True),
    ]

    vector_search = VectorSearch(algorithms=[
        HnswAlgorithmConfiguration(
            name="dm-hnsw",
            kind="hnsw",
            parameters={"m": 30, "efConstruction": 400, "metric": VectorSearchAlgorithmMetric.COSINE}
        )
    ])

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )

    search_index_client.create_index(index)
    print(f"[+] Created index '{INDEX_NAME}'.")

def load_records() -> List[Dict]:
    """
    Prefer JSON if you saved it earlier. Fallback to parsing the TXT separator lines.
    JSON schema expected: [{title, link, summary}]
    """
    records = []
    if JSON_PATH.exists():
        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        for row in data:
            records.append({
                "title": clean_text(row.get("title", "")),
                "url": row.get("link", ""),
                "summary": clean_text(row.get("summary", "")),
            })
        return records

    # Very simple TXT parser: blocks separated by "--------"
    if TXT_PATH.exists():
        raw = TXT_PATH.read_text(encoding="utf-8")
        blocks = [b.strip() for b in raw.split("-" * 80) if b.strip()]
        for b in blocks:
            lines = [l for l in b.splitlines() if l.strip()]
            if len(lines) >= 3:
                title = lines[0].strip()
                url   = lines[1].strip()
                summary = " ".join(lines[2:]).strip()
                records.append({"title": title, "url": url, "summary": summary})
    return records

def build_documents(rows: List[Dict]) -> List[Dict]:
    docs = []
    print(f"[*] Processing {len(rows)} articles...")
    
    for i, row in enumerate(rows, 1):
        try:
            content = f"{row.get('title','')}\n\n{row.get('summary','')}"
            content = clean_text(content)
            
            # Skip empty content
            if not content.strip():
                print(f"  [!] Skipping empty article {i}/{len(rows)}")
                continue
            
            # chunk and embed
            chunks = chunk_text(content, max_tokens=700, overlap_tokens=80)
            if not chunks:
                print(f"  [!] No chunks for article {i}/{len(rows)}: {row.get('title','')[:50]}")
                continue
                
            vectors = embed(chunks)

            # One doc per chunk (recommended for granular retrieval)
            for j, (chunk, vec) in enumerate(zip(chunks, vectors)):
                docs.append({
                    "id": f"{uuid.uuid5(uuid.NAMESPACE_URL, row.get('url',''))}-{j}",
                    "title": row.get("title",""),
                    "summary": row.get("summary",""),
                    "content": chunk,
                    "content_vector": vec,
                    "url": row.get("url",""),
                    "lang": "ar",   # Change to 'en' if needed; mixed is fine
                })
            
            # Progress indicator
            if i % 10 == 0:
                print(f"  [+] Processed {i}/{len(rows)} articles -> {len(docs)} chunks so far")
        
        except Exception as e:
            print(f"  [!] Error processing article {i}/{len(rows)}: {str(e)[:100]}")
            continue
    
    return docs

def upload(docs: List[Dict]):
    """Upload documents in batches with error handling."""
    BATCH = 1000
    total_failed = 0
    
    for i in range(0, len(docs), BATCH):
        batch = docs[i:i+BATCH]
        try:
            r = search_client.upload_documents(documents=batch)
            failed = [x for x in r if not x.succeeded]
            total_failed += len(failed)
            
            if failed:
                print(f"[!] Batch {i//BATCH + 1}: Uploaded {len(batch) - len(failed)}/{len(batch)} docs. Failed: {len(failed)}")
                for fail in failed[:3]:  # Show first 3 failures
                    print(f"    - Failed ID: {fail.key}, Error: {fail.error_message if hasattr(fail, 'error_message') else 'Unknown'}")
            else:
                print(f"[+] Batch {i//BATCH + 1}: Successfully uploaded {len(batch)} docs ({i+len(batch)}/{len(docs)})")
        except Exception as e:
            print(f"[!] Error uploading batch {i//BATCH + 1}: {str(e)[:200]}")
            total_failed += len(batch)
    
    if total_failed > 0:
        print(f"\n[!] Total failed uploads: {total_failed}/{len(docs)}")
    else:
        print(f"\n[+] Successfully uploaded all {len(docs)} documents!")

def sample_query(user_query: str, k: int = 5):
    # Hybrid: vector + text
    # 1) embed the query
    qvec = embed([user_query])[0]
    results = search_client.search(
        search_text=user_query,
        top=k,
        vector={"value": qvec, "k": k, "fields": "content_vector"},
        query_type="semantic",
        semantic_configuration_name="dm-semantic"
    )
    print("\n=== Sample results ===")
    for doc in results:
        print(doc["title"], "->", doc["url"])

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    print("=" * 70)
    print("Doha Magazine Article Indexer")
    print("=" * 70)
    
    # Step 1: Ensure index exists
    print("\n[1/5] Setting up Azure AI Search index...")
    ensure_index()
    
    # Step 2: Load articles
    print("\n[2/5] Loading scraped articles...")
    rows = load_records()
    if not rows:
        raise SystemExit("ERROR: No records found. Ensure your TXT/JSON files are present.")
    print(f"[+] Loaded {len(rows)} articles")
    
    # Step 3: Build documents with embeddings
    print("\n[3/5] Building document chunks and generating embeddings...")
    print("    (This may take a while depending on the number of articles)")
    docs = build_documents(rows)
    
    if not docs:
        raise SystemExit("ERROR: No documents were created. Check your data.")
    
    print(f"\n[+] Prepared {len(docs)} document chunks from {len(rows)} articles")
    print(f"    Average chunks per article: {len(docs)/len(rows):.1f}")
    
    # Step 4: Upload to Azure AI Search
    print("\n[4/5] Uploading documents to Azure AI Search...")
    upload(docs)
    
    # Step 5: Test with a sample query
    print("\n[5/5] Testing with a sample query...")
    try:
        sample_query("ما آخر مقالات الدوحة عن الثقافة؟")
    except Exception as e:
        print(f"[!] Sample query failed: {str(e)[:200]}")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Indexing complete! Total time: {elapsed:.1f} seconds")
    print("=" * 70)
