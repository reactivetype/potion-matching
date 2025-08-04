import time
import numpy as np
from entity_disambiguation import EntityDisambiguator


def main():
    # Initialize disambiguator
    print("Testing POTION model on edge cases...")
    disambiguator = EntityDisambiguator()
    
    # Test entities
    entities = [
        {"id": "1", "descriptor": "John Smith - Software Engineer at Google"},
        {"id": "2", "descriptor": "John Smith - Professor of Physics at MIT"},
        {"id": "3", "descriptor": "John Doe - Data Scientist at Microsoft"},
        {"id": "4", "descriptor": "Jane Smith - Product Manager at Apple"},
        {"id": "5", "descriptor": "Michael Johnson - Olympic Athlete"},
        {"id": "6", "descriptor": "Michael Jordan - Basketball Player"},
        {"id": "7", "descriptor": "Sarah Johnson - CEO of Tech Startup"},
        {"id": "8", "descriptor": "Jon Snow - Game of Thrones Character"},
        {"id": "9", "descriptor": "Johnny Depp - Actor"},
        {"id": "10", "descriptor": "Johnson & Johnson - Healthcare Company"},
    ]
    
    # Create embeddings
    entity_embeddings = disambiguator.create_entity_embeddings(entities)
    
    # Test cases focusing on exact matches and typos
    test_cases = [
        # Exact string matches that might fail
        ("John Smith", "Exact name match"),
        ("Michael Johnson", "Exact full name"),
        ("Jane Smith", "Exact female name"),
        ("Sarah Johnson", "Another exact match"),
        
        # Typo variations
        ("Jhon Smith", "Common typo: Jhon instead of John"),
        ("Jon Smith", "Missing 'h' in John"),
        ("John Smth", "Missing 'i' in Smith"),
        ("Jonh Smith", "Transposed letters"),
        ("John Simth", "Common typo in Smith"),
        
        # More typos
        ("Micheal Johnson", "Common misspelling of Michael"),
        ("Michel Johnson", "Missing 'a' in Michael"),
        ("Michael Jonson", "Missing 'h' in Johnson"),
        ("Micahel Johnson", "Transposed letters"),
        
        # Case variations
        ("john smith", "Lowercase"),
        ("JOHN SMITH", "Uppercase"),
        ("JoHn SmItH", "Mixed case"),
        
        # Partial matches
        ("John", "First name only"),
        ("Smith", "Last name only"),
        ("Johnson", "Common last name"),
        
        # Near matches
        ("Jon", "Similar to John"),
        ("Johnny", "Longer variant"),
        ("Smiths", "Plural form"),
        
        # Company vs Person
        ("Johnson", "Should it match the company or people?"),
    ]
    
    print("\n" + "="*80)
    print("EXACT MATCH AND TYPO ROBUSTNESS TEST")
    print("="*80)
    
    # Track exact match failures
    exact_match_failures = []
    typo_successes = []
    
    for query, description in test_cases:
        print(f"\nQuery: '{query}' - {description}")
        results, search_time, match_type = disambiguator.search(
            query, entities, entity_embeddings, threshold=0.3
        )
        
        print(f"  Found {len(results)} matches:")
        
        if len(results) == 0:
            print("  ❌ NO MATCHES FOUND")
            if "exact" in description.lower() or "typo" in description.lower():
                exact_match_failures.append((query, description))
        else:
            for i, result in enumerate(results[:3]):
                print(f"    {i+1}. {result['descriptor']} (similarity: {result['similarity']:.3f})")
            
            # Check if typo was handled well
            if "typo" in description.lower():
                # Check if the intended match is in top results
                if any("John Smith" in r['descriptor'] for r in results[:2]):
                    typo_successes.append((query, description))
                elif any("Michael Johnson" in r['descriptor'] for r in results[:2]):
                    typo_successes.append((query, description))
    
    # Additional test: How similar are exact matches vs typos?
    print("\n" + "="*80)
    print("SIMILARITY SCORE COMPARISON")
    print("="*80)
    
    comparison_queries = [
        ("John Smith", "Exact match"),
        ("Jhon Smith", "One letter typo"),
        ("Jon Smith", "Missing letter"),
        ("John Smth", "Missing letter in surname"),
        ("Jahn Smith", "Different typo"),
    ]
    
    target_entity = entities[0]  # John Smith - Software Engineer
    target_idx = 0
    
    print(f"Target: {target_entity['descriptor']}")
    print("\nQuery similarities:")
    
    for query, desc in comparison_queries:
        query_embedding = disambiguator.model.encode([query]).reshape(1, -1)
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(query_embedding, entity_embeddings[target_idx].reshape(1, -1))[0][0]
        print(f"  '{query}' ({desc}): {similarity:.4f}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total test cases: {len(test_cases)}")
    print(f"Exact match failures: {len(exact_match_failures)}")
    print(f"Typo handling successes: {len(typo_successes)}")
    
    if exact_match_failures:
        print("\n❌ Failed exact matches:")
        for query, desc in exact_match_failures[:5]:
            print(f"  - '{query}': {desc}")
    
    if typo_successes:
        print("\n✅ Successfully handled typos:")
        for query, desc in typo_successes[:5]:
            print(f"  - '{query}': {desc}")


if __name__ == "__main__":
    main()