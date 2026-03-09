# src/data_engine/evaluate_models.py
import os
import math
from .db import get_supabase_client
from .embedder import embed_texts, MODEL_CHOICE

# The "Golden Dataset"
# 10 rigorous semantic test questions mapped to enterprise Android risk intelligence
TEST_QUESTIONS = [
    # --- Category 1: Google Play Policy (Testing Compliance & De-platforming Risk) ---
    {"query": "Does an application face platform removal risk if it artificially boosts its visibility by offering users in-game currency for 5-star ratings?", "keyword": "incentivized"},
    {"query": "What are the regulatory and compliance risks for an app that generates or distributes unauthorized AI deepfakes of public figures?", "keyword": "manipulated media"},
    {"query": "Is downloading dynamic executable code from a third-party server considered a critical security risk and malware policy violation by Google?", "keyword": "malicious behavior"},
    {"query": "What is the policy violation risk if a developer uses a prominent politician's face or corporate logo in their app icon without authorization?", "keyword": "impersonation"},

    # --- Category 2: Android Security Bulletins (Testing OS-Level Threat Intel) ---
    {"query": "What is the defined risk severity of the latest elevation of privilege (EoP) vulnerabilities discovered in Qualcomm closed-source components?", "keyword": "qualcomm"},
    {"query": "What specific Android Security Patch Level (SPL) string must a device apply to successfully mitigate the risks disclosed in the February 2025 security bulletin?", "keyword": "patch level"},
    {"query": "Are there currently any critical remote code execution (RCE) flaws posing a severe risk to the Android System or Framework components?", "keyword": "remote code execution"},

    # --- Category 3: CISA KEV Catalog (Testing Active Zero-Day Threats) ---
    {"query": "Which specific Android OS and Pixel vulnerabilities are currently recognized by CISA as being actively exploited by threat actors in the wild?", "keyword": "cve-"},
    
    # --- Category 4: Play Integrity API (Testing Anti-Tampering & Compromise Mitigation) ---
    {"query": "How can financial applications mitigate device compromise risks by programmatically verifying if a user's phone has been rooted or unlocked?", "keyword": "verdict"},
    {"query": "What security architecture should be implemented to defend against the risk of unauthorized APK modifications, repackaging, and malicious bot traffic?", "keyword": "play integrity"}
]

def evaluate_hit_rate():
    print(f"\n🧪 Starting Advanced RAG Evaluation using model: {MODEL_CHOICE.upper()}")
    sb = get_supabase_client()
    top_k = 5
    
    # Metric Accumulators
    total_hits = 0
    sum_precision = 0.0
    sum_mrr = 0.0
    sum_ndcg = 0.0
    
    for item in TEST_QUESTIONS:
        question = item["query"]
        expected_keyword = item["keyword"].lower()
        
        # 1. Embed the question
        query_vector = embed_texts([question], is_query=True)[0]
        
        # 2. Search the Supabase vector database
        response = sb.rpc(
            os.getenv("VECTOR_RPC_MATCH", "match_vector_chunks"),
            {
                "query_embedding": query_vector, 
                "match_count": top_k,
                "filter_source_id": None,
                "filter_kind": None
            }
        ).execute()
        
        results = response.data or []
        
        # 3. Build a binary relevance list (1 if keyword found, 0 if not) for the top K results
        relevance_list = []
        for res in results:
            text = res.get("chunk_text", "").lower()
            relevance_list.append(1 if expected_keyword in text else 0)
            
        # 4. Calculate Metrics for this specific query
        hit = 1 if sum(relevance_list) > 0 else 0
        total_hits += hit
        
        precision = sum(relevance_list) / top_k if top_k > 0 else 0.0
        sum_precision += precision
        
        mrr = 0.0
        if hit:
            first_hit_index = relevance_list.index(1)
            mrr = 1.0 / (first_hit_index + 1)
        sum_mrr += mrr
        
        # Calculate nDCG
        dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance_list))
        num_relevant = sum(relevance_list)
        idcg = sum(1 / math.log2(idx + 2) for idx in range(num_relevant))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        sum_ndcg += ndcg
        
        # 5. Log individual query performance
        if hit:
            print(f"✅ Hit | Rank: {relevance_list.index(1) + 1} | MRR: {mrr:.2f} | nDCG: {ndcg:.2f} | Query: '{question}'")
        else:
            print(f"❌ Miss | Failed to find context for: '{question}'")

    # 6. Final Aggregation
    num_q = len(TEST_QUESTIONS)
    avg_hit_rate = (total_hits / num_q) * 100
    avg_precision = (sum_precision / num_q) * 100
    
    # In proxy-based RAG evaluation without a labeled universe of total possible relevant documents, 
    # Recall@K is mathematically equivalent to the binary Hit Rate (Did we recall the necessary fact: Yes/No).
    avg_recall = avg_hit_rate 
    
    avg_mrr = sum_mrr / num_q
    avg_ndcg = sum_ndcg / num_q

    print(f"\n📊 FINAL METRICS ({MODEL_CHOICE.upper()}) @ K={top_k}")
    print("-" * 50)
    print(f"🎯 Hit Rate @ {top_k}:  {avg_hit_rate:.1f}%")
    print(f"🎯 Recall @ {top_k}:    {avg_recall:.1f}% (Proxy assumed 1 necessary document per query)")
    print(f"🎯 Precision @ {top_k}: {avg_precision:.1f}% (Density of relevant chunks in Top K)")
    print(f"🥇 MRR:             {avg_mrr:.3f} (Rank quality of the FIRST relevant chunk)")
    print(f"📈 nDCG @ {top_k}:      {avg_ndcg:.3f} (Overall ranking quality score)")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    evaluate_hit_rate()