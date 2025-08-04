from entity_disambiguation import EntityDisambiguator
from entity_disambiguation_hybrid import HybridEntityDisambiguator
import numpy as np


def analyze_approach_behavior(disambiguator, entities, entity_embeddings, approach_name):
    """Analyze how each approach behaves on specific test cases"""
    
    test_cases = [
        ("John Smith", 0.5, "Should match both John Smiths exactly"),
        ("Jhon Smith", 0.5, "Typo - should ideally match John Smiths only"),
        ("John", 0.4, "Should match all Johns (John Smith x2, John Doe, John Williams)"),
        ("Johnson", 0.4, "Should match all Johnsons (Michael, Sarah, Robert)"),
    ]
    
    print(f"\n{approach_name} BEHAVIOR ANALYSIS")
    print("="*80)
    
    for query, threshold, expected in test_cases:
        results, _, _ = disambiguator.search(query, entities, entity_embeddings, threshold)
        
        print(f"\nQuery: '{query}' (threshold={threshold})")
        print(f"Expected: {expected}")
        print(f"Found {len(results)} matches:")
        
        for i, result in enumerate(results[:5]):
            print(f"  {i+1}. {result['descriptor']} (score: {result['similarity']:.3f})")
        
        if len(results) > 5:
            print(f"  ... and {len(results)-5} more")
    
    return


def main():
    print("Analyzing why metrics differ between approaches...")
    
    # Initialize
    original = EntityDisambiguator()
    hybrid = HybridEntityDisambiguator()
    
    # Entities
    entities = [
        {"id": "1", "descriptor": "John Smith - Software Engineer at Google"},
        {"id": "2", "descriptor": "John Smith - Professor of Physics at MIT"},
        {"id": "3", "descriptor": "John Doe - Data Scientist at Microsoft"},
        {"id": "4", "descriptor": "Jane Smith - Product Manager at Apple"},
        {"id": "5", "descriptor": "Michael Johnson - Olympic Athlete"},
        {"id": "6", "descriptor": "Michael Jordan - Basketball Player"},
        {"id": "7", "descriptor": "Sarah Johnson - CEO of Tech Startup"},
        {"id": "8", "descriptor": "John Williams - Composer"},
        {"id": "9", "descriptor": "Robert Johnson - Blues Musician"},
    ]
    
    # Create embeddings
    original_embeddings = original.create_entity_embeddings(entities)
    hybrid_embeddings = hybrid.create_entity_embeddings(entities)
    
    # Analyze behavior
    analyze_approach_behavior(original, entities, original_embeddings, "ORIGINAL")
    analyze_approach_behavior(hybrid, entities, hybrid_embeddings, "HYBRID")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS ON METRICS")
    print("="*80)
    
    print("""
1. PRECISION vs RECALL TRADE-OFF:
   - Original: High recall (0.957) but lower precision (0.688)
     → Returns many results, including false positives
   - Hybrid: Higher precision (0.720) but lower recall (0.783)
     → More selective, may miss some valid matches

2. WHY HYBRID HAS LOWER RECALL ON SOME QUERIES:
   - "John" query: Hybrid returns only 1 result (John Doe) instead of all 4 Johns
   - "Jhon Smith" typo: Hybrid returns all 9 entities (too broad)
   - The preprocessing and exact matching can sometimes be too restrictive or too broad

3. PERFECT SCORES:
   - Hybrid achieves perfect F1 (1.0) on 10/13 queries (vs 5/13 for original)
   - This shows hybrid is better for exact matches and well-formed queries
   - But it struggles with partial matches and some typos

4. MACRO vs OVERALL METRICS:
   - Macro F1: Hybrid is better (0.866 vs 0.817) - better per-query average
   - Overall F1: Original is better (0.800 vs 0.750) - better across all predictions
   - This suggests hybrid excels on most queries but fails badly on a few

5. THE PROBLEM CASES:
   - Partial name queries ("John", "Johnson") 
   - Some typos trigger fuzzy matching too broadly
   - Trade-off between exact matching and semantic understanding
""")
    
    print("\nRECOMMENDATIONS:")
    print("-"*80)
    print("1. Tune fuzzy matching threshold (currently 0.85) to be more restrictive")
    print("2. Improve partial name matching logic")
    print("3. Add query type detection (exact name vs partial vs semantic)")
    print("4. Consider ensemble approach: use hybrid for exact matches, original for partial")
    print("5. Implement query-specific thresholds based on query characteristics")


if __name__ == "__main__":
    main()