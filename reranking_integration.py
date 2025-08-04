import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import OrderedDict
import asyncio


@dataclass
class RerankingResult:
    """Result from reranking stage"""
    entity_id: str
    descriptor: str
    retrieval_score: float
    rerank_score: float
    final_score: float
    features: Dict[str, float]


class LRUCache:
    """Simple LRU cache for reranking results"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[List[RerankingResult]]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: List[RerankingResult]):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        if len(self.cache) > self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)


class MockReranker:
    """Mock reranker to simulate mxbai-rerank-v2 behavior"""
    
    def __init__(self, model_name: str = "mxbai-rerank-v2"):
        self.model_name = model_name
        # Simulate model loading time
        time.sleep(0.1)
    
    def compute_features(self, query: str, document: str) -> Dict[str, float]:
        """Extract features for reranking"""
        query_lower = query.lower()
        doc_lower = document.lower()
        
        # Simple feature extraction
        features = {
            "exact_match": 1.0 if query_lower in doc_lower else 0.0,
            "word_overlap": len(set(query_lower.split()) & set(doc_lower.split())) / len(query_lower.split()),
            "length_ratio": min(len(query), len(document)) / max(len(query), len(document)),
            "position_score": 1.0 if doc_lower.startswith(query_lower) else 0.5,
        }
        
        return features
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """Simulate reranking with realistic latency"""
        # Simulate inference time (8-12ms for 20 docs)
        base_latency = 0.008  # 8ms base
        per_doc_latency = 0.0002  # 0.2ms per doc
        time.sleep(base_latency + per_doc_latency * len(documents))
        
        # Compute reranking scores
        scores = []
        for idx, doc in enumerate(documents):
            features = self.compute_features(query, doc)
            # Simulate cross-attention score
            score = (
                features["exact_match"] * 0.4 +
                features["word_overlap"] * 0.3 +
                features["length_ratio"] * 0.1 +
                features["position_score"] * 0.2
            )
            scores.append((idx, score))
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class TwoStageEntityDisambiguator:
    """Entity disambiguator with optional reranking stage"""
    
    def __init__(self, retriever, reranker=None, use_cache=True, cache_size=10000):
        self.retriever = retriever
        self.reranker = reranker
        self.use_cache = use_cache
        self.cache = LRUCache(cache_size) if use_cache else None
        
        # Metrics tracking
        self.metrics = {
            "retrieval_time": [],
            "rerank_time": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "total_searches": 0
        }
    
    def should_rerank(self, results: List[Dict], query_type: str) -> bool:
        """Determine if reranking would be beneficial"""
        if not self.reranker or len(results) <= 3:
            return False
        
        # Don't rerank if we have a clear winner
        if results and results[0]["similarity"] > 0.95:
            return False
        
        # Check score distribution
        if len(results) >= 5:
            top_scores = [r["similarity"] for r in results[:5]]
            score_variance = np.var(top_scores)
            
            # Rerank if scores are too similar
            if score_variance < 0.01:
                return True
        
        # Rerank for ambiguous queries
        return query_type == "ambiguous"
    
    def extract_reranking_features(self, query: str, result: Dict) -> Dict[str, float]:
        """Extract additional features for reranking"""
        features = {}
        
        # Query-entity interaction features
        query_words = set(query.lower().split())
        entity_words = set(result["descriptor"].lower().split())
        
        features["word_overlap"] = len(query_words & entity_words) / len(query_words)
        features["retrieval_score"] = result["similarity"]
        features["match_type_exact"] = 1.0 if result.get("match_type") == "exact" else 0.0
        features["match_type_fuzzy"] = 1.0 if result.get("match_type") == "fuzzy" else 0.0
        
        return features
    
    def search(self, query: str, entities: List[Dict[str, str]], 
              entity_embeddings: Dict, threshold: float = 0.5,
              retrieval_top_k: int = 20, final_top_k: int = 5) -> Dict:
        """Two-stage search with optional reranking"""
        
        self.metrics["total_searches"] += 1
        
        # Stage 1: Retrieval
        retrieval_start = time.time()
        retrieval_results, _, query_type = self.retriever.search(
            query, entities, entity_embeddings, threshold
        )
        
        # Limit to retrieval_top_k
        retrieval_results = retrieval_results[:retrieval_top_k]
        retrieval_time = time.time() - retrieval_start
        self.metrics["retrieval_time"].append(retrieval_time)
        
        # Check if reranking is needed
        if not self.should_rerank(retrieval_results, query_type):
            return {
                "results": retrieval_results[:final_top_k],
                "search_time": retrieval_time,
                "match_type": query_type,
                "used_reranker": False,
                "retrieval_candidates": len(retrieval_results)
            }
        
        # Stage 2: Reranking
        rerank_start = time.time()
        
        # Check cache
        if self.use_cache:
            cache_key = f"{query}:{','.join([r['id'] for r in retrieval_results])}"
            cached_results = self.cache.get(cache_key)
            
            if cached_results:
                self.metrics["cache_hits"] += 1
                rerank_time = 0.0001  # Simulate cache lookup time
                self.metrics["rerank_time"].append(rerank_time)
                
                return {
                    "results": cached_results[:final_top_k],
                    "search_time": retrieval_time + rerank_time,
                    "match_type": query_type,
                    "used_reranker": True,
                    "from_cache": True,
                    "retrieval_candidates": len(retrieval_results)
                }
            else:
                self.metrics["cache_misses"] += 1
        
        # Perform reranking
        documents = [r["descriptor"] for r in retrieval_results]
        reranked_indices = self.reranker.rerank(query, documents, final_top_k)
        
        # Build final results with reranking scores
        final_results = []
        for idx, rerank_score in reranked_indices:
            original_result = retrieval_results[idx]
            features = self.extract_reranking_features(query, original_result)
            
            # Combine retrieval and reranking scores
            final_score = 0.6 * original_result["similarity"] + 0.4 * rerank_score
            
            reranked_result = RerankingResult(
                entity_id=original_result["id"],
                descriptor=original_result["descriptor"],
                retrieval_score=original_result["similarity"],
                rerank_score=rerank_score,
                final_score=final_score,
                features=features
            )
            
            final_results.append({
                **original_result,
                "similarity": final_score,
                "retrieval_score": original_result["similarity"],
                "rerank_score": rerank_score
            })
        
        rerank_time = time.time() - rerank_start
        self.metrics["rerank_time"].append(rerank_time)
        
        # Cache results
        if self.use_cache:
            self.cache.put(cache_key, final_results)
        
        return {
            "results": final_results,
            "search_time": retrieval_time + rerank_time,
            "match_type": query_type,
            "used_reranker": True,
            "from_cache": False,
            "retrieval_candidates": len(retrieval_results),
            "rerank_time": rerank_time
        }
    
    def get_metrics_summary(self) -> Dict:
        """Get performance metrics summary"""
        return {
            "total_searches": self.metrics["total_searches"],
            "avg_retrieval_time_ms": np.mean(self.metrics["retrieval_time"]) * 1000 if self.metrics["retrieval_time"] else 0,
            "avg_rerank_time_ms": np.mean(self.metrics["rerank_time"]) * 1000 if self.metrics["rerank_time"] else 0,
            "cache_hit_rate": self.metrics["cache_hits"] / self.metrics["total_searches"] if self.metrics["total_searches"] > 0 else 0,
            "searches_with_reranking": len(self.metrics["rerank_time"]) / self.metrics["total_searches"] if self.metrics["total_searches"] > 0 else 0
        }


async def progressive_search(disambiguator: TwoStageEntityDisambiguator,
                           query: str, entities: List[Dict], 
                           embeddings: Dict) -> List[Dict]:
    """Progressive search that returns initial results immediately"""
    
    # First, return quick results
    quick_results = disambiguator.retriever.search(
        query, entities, embeddings, threshold=0.5
    )
    
    yield {
        "stage": "initial",
        "results": quick_results[0][:5],  # Top 5 quick results
        "time": quick_results[1]
    }
    
    # Then, if beneficial, perform reranking
    full_results = disambiguator.search(
        query, entities, embeddings,
        retrieval_top_k=20, final_top_k=5
    )
    
    if full_results["used_reranker"]:
        yield {
            "stage": "refined",
            "results": full_results["results"],
            "time": full_results["search_time"],
            "improvement": "reranked"
        }


# Demonstration
if __name__ == "__main__":
    print("RERANKING INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    # Mock retriever
    class MockRetriever:
        def search(self, query, entities, embeddings, threshold):
            # Simulate retrieval with similar scores
            results = []
            for i, entity in enumerate(entities[:10]):
                # Generate similar scores to simulate ambiguity
                base_score = 0.75 + np.random.uniform(-0.05, 0.05)
                results.append({
                    **entity,
                    "similarity": base_score,
                    "match_type": "semantic"
                })
            
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results, 0.0002, "ambiguous"  # 0.2ms retrieval
    
    # Test entities
    entities = [
        {"id": "1", "descriptor": "Senior Software Engineer - Google Cloud Platform"},
        {"id": "2", "descriptor": "Software Engineer - Google Search"},
        {"id": "3", "descriptor": "Staff Software Engineer - Google AI"},
        {"id": "4", "descriptor": "Software Development Engineer - Amazon AWS"},
        {"id": "5", "descriptor": "Senior Engineer - Microsoft Azure"},
        {"id": "6", "descriptor": "Principal Software Engineer - Meta"},
        {"id": "7", "descriptor": "Software Engineer II - Apple"},
        {"id": "8", "descriptor": "Lead Software Engineer - Netflix"},
        {"id": "9", "descriptor": "Senior Software Developer - Oracle"},
        {"id": "10", "descriptor": "Software Engineering Manager - Uber"},
    ]
    
    # Initialize components
    retriever = MockRetriever()
    reranker = MockReranker()
    
    # Test without reranking
    print("\n1. WITHOUT RERANKING:")
    print("-" * 60)
    
    basic_disambiguator = TwoStageEntityDisambiguator(retriever, reranker=None)
    
    start = time.time()
    results = basic_disambiguator.search("Google Engineer", entities, {}, threshold=0.5)
    print(f"Search time: {results['search_time']*1000:.2f}ms")
    print(f"Results: {len(results['results'])}")
    for r in results["results"][:3]:
        print(f"  - {r['descriptor']} (score: {r['similarity']:.3f})")
    
    # Test with reranking
    print("\n2. WITH RERANKING:")
    print("-" * 60)
    
    reranking_disambiguator = TwoStageEntityDisambiguator(retriever, reranker)
    
    # First search (no cache)
    results = reranking_disambiguator.search("Google Engineer", entities, {}, threshold=0.5)
    print(f"Search time: {results['search_time']*1000:.2f}ms")
    print(f"Used reranker: {results['used_reranker']}")
    print(f"Rerank time: {results.get('rerank_time', 0)*1000:.2f}ms")
    print(f"Results: {len(results['results'])}")
    for r in results["results"][:3]:
        print(f"  - {r['descriptor']} (score: {r['similarity']:.3f})")
    
    # Second search (with cache)
    print("\n3. CACHED SEARCH:")
    print("-" * 60)
    
    results = reranking_disambiguator.search("Google Engineer", entities, {}, threshold=0.5)
    print(f"Search time: {results['search_time']*1000:.2f}ms")
    print(f"From cache: {results.get('from_cache', False)}")
    
    # Performance comparison
    print("\n4. PERFORMANCE COMPARISON:")
    print("-" * 60)
    
    queries = [
        "Software Engineer",
        "Google",
        "Senior Engineer", 
        "Cloud Platform",
        "Engineering Manager"
    ]
    
    # Run multiple searches
    for query in queries * 10:  # Run each query 10 times
        reranking_disambiguator.search(query, entities, {}, threshold=0.5)
    
    metrics = reranking_disambiguator.get_metrics_summary()
    
    print(f"Total searches: {metrics['total_searches']}")
    print(f"Average retrieval time: {metrics['avg_retrieval_time_ms']:.2f}ms")
    print(f"Average rerank time: {metrics['avg_rerank_time_ms']:.2f}ms")
    print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")
    print(f"Searches with reranking: {metrics['searches_with_reranking']:.1%}")
    
    # Latency breakdown
    print("\n5. LATENCY BREAKDOWN:")
    print("-" * 60)
    
    print("Without reranking: ~0.2ms")
    print("With reranking (no cache): ~10-15ms")
    print("With reranking (cached): ~0.3ms")
    print("\nExpected production performance:")
    print("- 70% cache hits: avg ~3.3ms per search")
    print("- 90% cache hits: avg ~1.3ms per search")