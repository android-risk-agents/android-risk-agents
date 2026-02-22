# src/data_engine/evaluate_models.py
import os
from .db import get_supabase_client
from .embedder import embed_texts, MODEL_CHOICE

# The "Golden Dataset"
# 10 test questions mapping to the expected Android risk ecosystem data
# The "Golden Dataset"
# 10 rigorous semantic test questions mapping to the Android risk ecosystem
TEST_QUESTIONS = [
    # --- Category 1: Google Play Policy (Testing Conceptual Semantic Matching) ---
    {"query": "Is it allowed to offer users in-game currency or cash in exchange for a 5-star rating on the Play Store?", "keyword": "incentivized"},
    {"query": "If my app allows users to create AI-generated deepfakes of public figures, what specific rules must I follow?", "keyword": "manipulated media"},
    {"query": "Can an app download executable code directly from our company's private server instead of the Play Store?", "keyword": "malicious behavior"},
    {"query": "What happens if a developer uses a prominent politician's face in their app icon without permission?", "keyword": "impersonation"},

    # --- Category 2: Android Security Bulletins (Testing Technical Granularity) ---
    {"query": "What is the severity of the latest elevation of privilege bugs found in the Qualcomm closed-source components?", "keyword": "qualcomm"},
    {"query": "Which Android Security Patch Level (SPL) string is required to address the vulnerabilities disclosed in February 2025?", "keyword": "patch level"},
    {"query": "Are there any critical remote code execution flaws affecting the System or Framework components?", "keyword": "remote code execution"},

    # --- Category 3: CISA KEV Catalog (Testing Dynamic JSON & Threat Intel) ---
    {"query": "Which specific vulnerabilities are currently being actively exploited in the wild on Android devices?", "keyword": "cve-"},
    
    # --- Category 4: Play Integrity API (Testing Developer Documentation) ---
    {"query": "How can I programmatically verify if a user's device has been rooted or compromised before processing a financial transaction?", "keyword": "verdict"},
    {"query": "What specific API should a game developer call to defend against unauthorized APK modifications and bot traffic?", "keyword": "play integrity"}
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