from entity_disambiguation_improved_flexible import ImprovedFlexibleEntityDisambiguator
from entity_disambiguation_flexible import FlexibleEntityDisambiguator
import time
import numpy as np


def analyze_search_patterns():
    """Analyze which matching strategies are used and their impact on speed"""
    
    print("ANALYZING SEARCH SPEED: Why Improved is Faster")
    print("="*80)
    
    # Initialize models
    potion_baseline = FlexibleEntityDisambiguator(
        model_name="minishlab/potion-multilingual-128M",
        model_type="static"
    )
    potion_improved = ImprovedFlexibleEntityDisambiguator(
        model_name="minishlab/potion-multilingual-128M", 
        model_type="static"
    )
    
    # Test entities
    entities = [
        {"id": "1", "descriptor": "John Smith - Software Engineer at Google"},
        {"id": "2", "descriptor": "John Michael Smith - Professor at MIT"},
        {"id": "3", "descriptor": "John Doe - Data Scientist at Microsoft"},
        {"id": "4", "descriptor": "Jane Smith - Product Manager at Apple"},
        {"id": "5", "descriptor": "Michael Johnson - Olympic Athlete"},
        {"id": "6", "descriptor": "Sarah Johnson - CEO of Tech Startup"},
    ]
    
    # Create embeddings
    baseline_embeddings = potion_baseline.create_entity_embeddings(entities)
    improved_embeddings = potion_improved.create_entity_embeddings(entities)
    
    # Test queries categorized by type
    test_queries = {
        "Exact matches": ["John Smith", "Michael Johnson", "Jane Smith"],
        "Partial names": ["John", "Smith", "Johnson"],
        "Typos": ["Jhon Smith", "Micheal Johnson"],
        "Case variations": ["john smith", "JANE SMITH"],
        "Semantic": ["Software Engineer", "CEO startup", "Olympic sport"],
    }
    
    print("\nSEARCH TIME BREAKDOWN BY QUERY TYPE:")
    print("-"*80)
    
    # Track which matching strategies are used
    strategy_counts = {
        "exact_match": 0,
        "fuzzy_match": 0,
        "semantic_search": 0,
        "partial_name_exact": 0,
        "partial_name_semantic": 0
    }
    
    for category, queries in test_queries.items():
        print(f"\n{category}:")
        
        for query in queries:
            # Baseline timing
            baseline_times = []
            for _ in range(100):  # Run 100 times for accuracy
                start = time.time()
                _, _, _ = potion_baseline.search(query, entities, baseline_embeddings, 0.5)
                baseline_times.append(time.time() - start)
            baseline_avg = np.mean(baseline_times) * 1000  # Convert to ms
            
            # Improved timing with strategy tracking
            improved_times = []
            for _ in range(100):
                start = time.time()
                results, _, match_type = potion_improved.search(query, entities, improved_embeddings, 0.5)
                improved_times.append(time.time() - start)
            improved_avg = np.mean(improved_times) * 1000
            
            # Track which strategy was used (from last search)
            if results and 'match_type' in results[0]:
                match_strategy = results[0]['match_type']
                if 'exact' in match_strategy:
                    if match_type == "partial":
                        strategy_counts["partial_name_exact"] += 1
                    else:
                        strategy_counts["exact_match"] += 1
                elif 'fuzzy' in match_strategy:
                    strategy_counts["fuzzy_match"] += 1
                elif 'semantic' in match_strategy:
                    if match_type == "partial":
                        strategy_counts["partial_name_semantic"] += 1
                    else:
                        strategy_counts["semantic_search"] += 1
            
            speedup = baseline_avg / improved_avg if improved_avg > 0 else float('inf')
            print(f"  '{query}': Baseline={baseline_avg:.3f}ms, Improved={improved_avg:.3f}ms, Speedup={speedup:.1f}x")
            if results and 'match_type' in results[0]:
                print(f"    → Strategy used: {results[0]['match_type']}")
    
    print("\n" + "="*80)
    print("MATCHING STRATEGY DISTRIBUTION:")
    print("-"*80)
    
    total_queries = sum(strategy_counts.values())
    for strategy, count in strategy_counts.items():
        percentage = (count / total_queries * 100) if total_queries > 0 else 0
        print(f"{strategy:<25}: {count:3d} queries ({percentage:5.1f}%)")
    
    print("\n" + "="*80)
    print("WHY THE IMPROVED VERSION IS FASTER:")
    print("="*80)
    
    print("""
1. EARLY EXIT STRATEGIES:
   - Exact name matches return immediately without computing embeddings
   - Fuzzy matches for typos avoid semantic search
   - Partial names use string comparison before falling back to embeddings

2. REDUCED EMBEDDING COMPUTATIONS:
   - Baseline: ALWAYS computes cosine similarity for ALL entities
   - Improved: Often finds matches without any embedding computation

3. EFFICIENT DATA STRUCTURES:
   - Pre-extracted name parts avoid repeated string processing
   - Cached normalized versions reduce preprocessing time

4. QUERY TYPE DETECTION:
   - Routes queries to appropriate strategy immediately
   - Avoids unnecessary computation paths
""")
    
    # Detailed timing analysis
    print("\nDETAILED TIMING ANALYSIS:")
    print("-"*80)
    
    # Time individual operations
    test_query = "John Smith"
    n_iterations = 1000
    
    # Time embedding computation
    start = time.time()
    for _ in range(n_iterations):
        query_emb = potion_baseline.encode([test_query])
    embedding_time = (time.time() - start) / n_iterations * 1000
    
    # Time cosine similarity
    query_emb = potion_baseline.encode([test_query]).reshape(1, -1)
    start = time.time()
    for _ in range(n_iterations):
        from sklearn.metrics.pairwise import cosine_similarity
        _ = cosine_similarity(query_emb, baseline_embeddings)
    cosine_time = (time.time() - start) / n_iterations * 1000
    
    # Time string comparison
    start = time.time()
    for _ in range(n_iterations):
        for entity in entities:
            _ = test_query.lower() == entity["descriptor"].lower()
    string_time = (time.time() - start) / n_iterations * 1000
    
    print(f"Embedding computation: {embedding_time:.3f} ms")
    print(f"Cosine similarity (6 entities): {cosine_time:.3f} ms")
    print(f"String comparison (6 entities): {string_time:.3f} ms")
    print(f"\nString comparison is {embedding_time/string_time:.0f}x faster than embedding")
    print(f"String comparison is {(embedding_time + cosine_time)/string_time:.0f}x faster than full semantic search")


if __name__ == "__main__":
    analyze_search_patterns()