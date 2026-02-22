# src/data_engine/evaluate_models.py
import os
from .db import get_supabase_client
from .embedder import embed_texts, MODEL_CHOICE

# The "Golden Dataset"
# 10 test questions mapping to the expected Android risk ecosystem data
TEST_QUESTIONS = [
    {"query": "What are the rules regarding restricted content and malware?", "keyword": "malware"},
    {"query": "How does the Play Integrity API protect applications?", "keyword": "integrity"},
    {"query": "What recent vulnerabilities have been added to the CISA catalog?", "keyword": "cisa"},
    {"query": "Are there any critical security patches for the Android framework?", "keyword": "framework"},
    {"query": "What happens if a developer violates the impersonation policy?", "keyword": "impersonation"},
]

def evaluate_hit_rate():
    print(f"\n🧪 Starting RAG Evaluation using model: {MODEL_CHOICE.upper()}")
    sb = get_supabase_client()
    hits = 0
    top_k = 5 # We are testing Hit Rate @ 5
    
    for item in TEST_QUESTIONS:
        question = item["query"]
        expected_keyword = item["keyword"].lower()
        
        # 1. Embed the question (notice is_query=True for Nomic)
        query_vector = embed_texts([question], is_query=True)[0]
        
        # 2. Search the Supabase vector database
        # 2. Search the Supabase vector database
        response = sb.rpc(
            os.getenv("VECTOR_RPC_MATCH", "match_vector_chunks"),
            {
                "query_embedding": query_vector, 
                "match_count": top_k,
                "filter_source_id": None,  # By explicitly passing these as None...
                "filter_kind": None        # ...Supabase instantly knows exactly which function to use!
            }
        ).execute()
        
        results = response.data
        found_hit = False
        
        # 3. Check if the retrieved text actually contains the right context
        for rank, res in enumerate(results):
            text = res.get("chunk_text", "").lower()
            if expected_keyword in text:
                found_hit = True
                print(f"✅ Hit at Rank {rank + 1} | Query: '{question}'")
                break
                
        if not found_hit:
            print(f"❌ Miss | Failed to find context for: '{question}'")
            
        if found_hit:
            hits += 1

    hit_rate = (hits / len(TEST_QUESTIONS)) * 100
    print(f"\n📊 FINAL SCORE ({MODEL_CHOICE.upper()}): {hit_rate}% Hit Rate @ {top_k}\n")

if __name__ == "__main__":
    evaluate_hit_rate()