import time
from typing import List, Dict, Tuple
from model2vec import StaticModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EntityDisambiguator:
    def __init__(self, model_name: str = "minishlab/potion-multilingual-128M"):
        print(f"Loading model: {model_name}")
        start_time = time.time()
        self.model = StaticModel.from_pretrained(model_name)
        self.load_time = time.time() - start_time
        print(f"Model loaded in {self.load_time:.2f} seconds")
        
    def create_entity_embeddings(self, entities: List[Dict[str, str]]) -> np.ndarray:
        """Create embeddings for entity descriptors"""
        descriptors = [entity["descriptor"] for entity in entities]
        return self.model.encode(descriptors)
    
    def search(self, query: str, entities: List[Dict[str, str]], 
              entity_embeddings: np.ndarray, threshold: float = 0.5) -> List[Dict[str, str]]:
        """Search for matching entities based on query"""
        start_time = time.time()
        
        # Encode the query
        query_embedding = self.model.encode([query]).reshape(1, -1)
        
        # Calculate cosine similarities using sklearn
        similarities = cosine_similarity(query_embedding, entity_embeddings)[0]
        
        # Find matches above threshold
        matches = []
        for idx, (entity, similarity) in enumerate(zip(entities, similarities)):
            if similarity >= threshold:
                matches.append({
                    **entity,
                    "similarity": float(similarity),
                    "rank": idx + 1
                })
        
        # Sort by similarity
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        search_time = time.time() - start_time
        
        # Check if we have a single high-confidence match
        if len(matches) == 1 and matches[0]["similarity"] >= 0.85:
            return matches, search_time, "exact"
        elif len(matches) > 0 and matches[0]["similarity"] >= 0.85 and (len(matches) == 1 or matches[0]["similarity"] - matches[1]["similarity"] > 0.1):
            # Single dominant match
            return [matches[0]], search_time, "exact"
        
        # If no semantic matches, fall back to substring matching
        if not matches:
            query_lower = query.lower()
            substring_matches = [
                entity for entity in entities 
                if query_lower in entity["descriptor"].lower()
            ]
            
            # Calculate similarities for substring matches
            for match in substring_matches:
                idx = entities.index(match)
                match["similarity"] = float(similarities[idx])
            
            substring_matches.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            
            return substring_matches, search_time, "ambiguous"
        
        return matches, search_time, "ambiguous"


def evaluate_performance(disambiguator: EntityDisambiguator, test_cases: List[Dict]):
    """Evaluate performance metrics"""
    total_queries = len(test_cases)
    correct_exact = 0
    correct_ambiguous = 0
    total_search_time = 0
    
    for test_case in test_cases:
        query = test_case["query"]
        expected_type = test_case["expected_type"]
        expected_ids = test_case["expected_ids"]
        
        start = time.time()
        results, search_time, result_type = disambiguator.search(
            query, test_case["entities"], test_case["entity_embeddings"]
        )
        total_search_time += search_time
        
        result_ids = [r["id"] for r in results]
        
        if result_type == expected_type:
            if expected_type == "exact" and result_ids == expected_ids:
                correct_exact += 1
            elif expected_type == "ambiguous" and set(result_ids) == set(expected_ids):
                correct_ambiguous += 1
    
    precision = (correct_exact + correct_ambiguous) / total_queries
    avg_search_time = total_search_time / total_queries
    
    return {
        "precision": precision,
        "correct_exact": correct_exact,
        "correct_ambiguous": correct_ambiguous,
        "total_queries": total_queries,
        "avg_search_time_ms": avg_search_time * 1000
    }


if __name__ == "__main__":
    # Initialize disambiguator
    disambiguator = EntityDisambiguator()
    
    # Sample entities database
    entities = [
        {"id": "1", "descriptor": "John Smith - Software Engineer at Google"},
        {"id": "2", "descriptor": "John Doe - Data Scientist at Microsoft"},
        {"id": "3", "descriptor": "John Williams - Composer and Conductor"},
        {"id": "4", "descriptor": "Jane Smith - Product Manager at Apple"},
        {"id": "5", "descriptor": "Michael Johnson - Olympic Athlete"},
        {"id": "6", "descriptor": "John Legend - Singer and Songwriter"},
        {"id": "7", "descriptor": "Sarah Johnson - CEO of Tech Startup"},
        {"id": "8", "descriptor": "David Brown - Professor of Computer Science"},
        {"id": "9", "descriptor": "Emily Davis - Machine Learning Researcher"},
        {"id": "10", "descriptor": "Robert Williams - Investment Banker"}
    ]
    
    # Create embeddings for all entities
    print("\nCreating entity embeddings...")
    entity_embeddings = disambiguator.create_entity_embeddings(entities)
    print(f"Created embeddings for {len(entities)} entities")
    
    # Test queries
    test_queries = [
        "John",  # Ambiguous - should return all Johns
        "Software Engineer at Google",  # More specific - should match John Smith
        "Olympic Athlete",  # Should match Michael Johnson
        "Singer Songwriter",  # Should match John Legend
        "Composer",  # Should match John Williams
        "CEO Tech",  # Should match Sarah Johnson
    ]
    
    print("\n" + "="*60)
    print("TESTING ENTITY DISAMBIGUATION")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results, search_time, match_type = disambiguator.search(query, entities, entity_embeddings)
        print(f"Match type: {match_type}")
        print(f"Search time: {search_time*1000:.2f} ms")
        print(f"Found {len(results)} matches:")
        
        for result in results[:5]:  # Show top 5
            print(f"  - ID: {result['id']}, {result['descriptor']} (similarity: {result['similarity']:.3f})")
    
    # Performance evaluation
    print("\n" + "="*60)
    print("PERFORMANCE EVALUATION")
    print("="*60)
    
    # Create test cases for evaluation
    test_cases = [
        {
            "query": "Software Engineer Google",
            "expected_type": "exact",
            "expected_ids": ["1"],
            "entities": entities,
            "entity_embeddings": entity_embeddings
        },
        {
            "query": "Olympic Athlete",
            "expected_type": "exact",
            "expected_ids": ["5"],
            "entities": entities,
            "entity_embeddings": entity_embeddings
        },
        {
            "query": "Singer",
            "expected_type": "exact",
            "expected_ids": ["6"],
            "entities": entities,
            "entity_embeddings": entity_embeddings
        }
    ]
    
    metrics = evaluate_performance(disambiguator, test_cases)
    
    print(f"\nModel load time: {disambiguator.load_time:.2f} seconds")
    print(f"Average search time: {metrics['avg_search_time_ms']:.2f} ms")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Correct exact matches: {metrics['correct_exact']}/{metrics['total_queries']}")
    print(f"Correct ambiguous matches: {metrics['correct_ambiguous']}/{metrics['total_queries']}")