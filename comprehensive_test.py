import time
import numpy as np
from entity_disambiguation import EntityDisambiguator


def main():
    # Initialize disambiguator
    print("Initializing POTION multilingual model...")
    disambiguator = EntityDisambiguator()
    
    # Larger entity database for more realistic testing
    entities = [
        # Johns
        {"id": "1", "descriptor": "John Smith - Software Engineer at Google specializing in machine learning"},
        {"id": "2", "descriptor": "John Smith - Professor of Physics at MIT"},
        {"id": "3", "descriptor": "John Doe - Data Scientist at Microsoft working on Azure"},
        {"id": "4", "descriptor": "John Williams - Famous film composer known for Star Wars"},
        {"id": "5", "descriptor": "John Legend - R&B singer and songwriter, winner of multiple Grammys"},
        {"id": "6", "descriptor": "John McCarthy - Computer scientist who coined the term AI"},
        
        # Michaels
        {"id": "7", "descriptor": "Michael Johnson - Olympic gold medalist sprinter"},
        {"id": "8", "descriptor": "Michael Jordan - Basketball legend and Chicago Bulls player"},
        {"id": "9", "descriptor": "Michael Jackson - King of Pop, singer and dancer"},
        
        # Others
        {"id": "10", "descriptor": "Jane Smith - Product Manager at Apple working on iOS"},
        {"id": "11", "descriptor": "Sarah Johnson - CEO and founder of AI startup TechVision"},
        {"id": "12", "descriptor": "Robert Williams - Investment banker at Goldman Sachs"},
        {"id": "13", "descriptor": "Emily Davis - Machine Learning researcher at DeepMind"},
        {"id": "14", "descriptor": "David Brown - Professor of Computer Science at Stanford"},
        {"id": "15", "descriptor": "Alice Johnson - Senior Software Developer at Netflix"},
    ]
    
    # Create embeddings
    print(f"\nCreating embeddings for {len(entities)} entities...")
    start = time.time()
    entity_embeddings = disambiguator.create_entity_embeddings(entities)
    print(f"Embedding creation time: {(time.time() - start)*1000:.2f} ms")
    
    # Test scenarios
    test_scenarios = [
        # Exact match scenarios
        ("Star Wars composer", "Should find John Williams"),
        ("Basketball Chicago Bulls", "Should find Michael Jordan"),
        ("King of Pop", "Should find Michael Jackson"),
        ("iOS Product Manager", "Should find Jane Smith"),
        
        # Ambiguous scenarios
        ("John Smith", "Should find both John Smiths"),
        ("Software Engineer", "Should find multiple engineers"),
        ("CEO startup", "Should find Sarah Johnson"),
        
        # Semantic similarity tests
        ("music creator", "Should find composers/singers"),
        ("athlete runner", "Should find Michael Johnson"),
        ("artificial intelligence", "Should find AI-related people"),
        ("tech company employee", "Should find tech workers"),
    ]
    
    print("\n" + "="*80)
    print("ENTITY DISAMBIGUATION TESTS")
    print("="*80)
    
    total_search_time = 0
    search_count = 0
    
    for query, description in test_scenarios:
        print(f"\nQuery: '{query}' - {description}")
        results, search_time, match_type = disambiguator.search(query, entities, entity_embeddings, threshold=0.4)
        total_search_time += search_time
        search_count += 1
        
        print(f"  Match type: {match_type}")
        print(f"  Search time: {search_time*1000:.2f} ms")
        print(f"  Found {len(results)} matches:")
        
        for i, result in enumerate(results[:5]):
            print(f"    {i+1}. {result['descriptor']}")
            print(f"       Similarity: {result['similarity']:.3f}")
    
    # Performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"Model: minishlab/potion-multilingual-128M")
    print(f"Model load time: {disambiguator.load_time:.2f} seconds")
    print(f"Number of entities: {len(entities)}")
    print(f"Average search time: {(total_search_time/search_count)*1000:.2f} ms")
    print(f"Total searches performed: {search_count}")
    
    # Batch search performance test
    print("\n" + "="*80)
    print("BATCH SEARCH PERFORMANCE TEST")
    print("="*80)
    
    batch_queries = ["John", "Michael", "Software", "CEO", "Professor", "Singer", "Engineer", "Scientist"]
    print(f"Running {len(batch_queries)} searches in sequence...")
    
    start_time = time.time()
    for query in batch_queries:
        _, _, _ = disambiguator.search(query, entities, entity_embeddings)
    batch_time = time.time() - start_time
    
    print(f"Total time for {len(batch_queries)} searches: {batch_time*1000:.2f} ms")
    print(f"Average time per search: {(batch_time/len(batch_queries))*1000:.2f} ms")
    print(f"Searches per second: {len(batch_queries)/batch_time:.1f}")


if __name__ == "__main__":
    main()