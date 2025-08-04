import time
from typing import List, Dict, Tuple
from model2vec import StaticModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from difflib import SequenceMatcher


class HybridEntityDisambiguator:
    def __init__(self, model_name: str = "minishlab/potion-multilingual-128M"):
        print(f"Loading model: {model_name}")
        start_time = time.time()
        self.model = StaticModel.from_pretrained(model_name)
        self.load_time = time.time() - start_time
        print(f"Model loaded in {self.load_time:.2f} seconds")
        
    def preprocess_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Convert to lowercase for consistency
        normalized = text.lower()
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        return normalized
    
    def extract_name(self, descriptor: str) -> str:
        """Extract the name part from a descriptor"""
        # Split by common separators
        parts = re.split(r' - | at | of ', descriptor)
        return parts[0].strip()
    
    def fuzzy_match_score(self, s1: str, s2: str) -> float:
        """Calculate fuzzy string matching score"""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def create_entity_embeddings(self, entities: List[Dict[str, str]]) -> Dict[str, np.ndarray]:
        """Create embeddings for entity descriptors with preprocessing"""
        # Original descriptors
        descriptors = [entity["descriptor"] for entity in entities]
        
        # Normalized descriptors
        normalized_descriptors = [self.preprocess_text(desc) for desc in descriptors]
        
        # Extract just names
        names = [self.extract_name(desc) for desc in descriptors]
        normalized_names = [self.preprocess_text(name) for name in names]
        
        # Create embeddings for all versions
        embeddings = {
            "original": self.model.encode(descriptors),
            "normalized": self.model.encode(normalized_descriptors),
            "names": self.model.encode(names),
            "normalized_names": self.model.encode(normalized_names)
        }
        
        return embeddings
    
    def search(self, query: str, entities: List[Dict[str, str]], 
              entity_embeddings: Dict[str, np.ndarray], threshold: float = 0.5) -> Tuple[List[Dict[str, str]], float, str]:
        """Hybrid search combining exact, fuzzy, and semantic matching"""
        start_time = time.time()
        
        # Preprocess query
        query_normalized = self.preprocess_text(query)
        
        # First, check for exact matches (case-insensitive)
        exact_matches = []
        for entity in entities:
            if query.lower() == entity["descriptor"].lower():
                exact_matches.append({
                    **entity,
                    "similarity": 1.0,
                    "match_type": "exact"
                })
            elif query.lower() == self.extract_name(entity["descriptor"]).lower():
                exact_matches.append({
                    **entity,
                    "similarity": 0.95,
                    "match_type": "exact_name"
                })
        
        if exact_matches:
            search_time = time.time() - start_time
            return exact_matches, search_time, "exact"
        
        # Check fuzzy matches for typos
        fuzzy_matches = []
        for entity in entities:
            name = self.extract_name(entity["descriptor"])
            fuzzy_score = self.fuzzy_match_score(query, name)
            if fuzzy_score >= 0.85:  # High threshold for fuzzy matching
                fuzzy_matches.append({
                    **entity,
                    "similarity": fuzzy_score,
                    "match_type": "fuzzy",
                    "fuzzy_score": fuzzy_score
                })
        
        # Semantic search with multiple embedding types
        query_embeddings = {
            "original": self.model.encode([query]).reshape(1, -1),
            "normalized": self.model.encode([query_normalized]).reshape(1, -1)
        }
        
        # Calculate similarities for different embedding types
        all_similarities = {}
        
        # Original query vs original descriptors
        all_similarities["orig_orig"] = cosine_similarity(
            query_embeddings["original"], 
            entity_embeddings["original"]
        )[0]
        
        # Normalized query vs normalized descriptors
        all_similarities["norm_norm"] = cosine_similarity(
            query_embeddings["normalized"], 
            entity_embeddings["normalized"]
        )[0]
        
        # Original query vs names only
        all_similarities["orig_names"] = cosine_similarity(
            query_embeddings["original"], 
            entity_embeddings["names"]
        )[0]
        
        # Normalized query vs normalized names
        all_similarities["norm_names"] = cosine_similarity(
            query_embeddings["normalized"], 
            entity_embeddings["normalized_names"]
        )[0]
        
        # Combine scores with weights
        combined_scores = []
        for idx, entity in enumerate(entities):
            # Get max similarity across different matching strategies
            semantic_score = max(
                all_similarities["orig_orig"][idx],
                all_similarities["norm_norm"][idx],
                all_similarities["orig_names"][idx],
                all_similarities["norm_names"][idx]
            )
            
            # Check if entity is in fuzzy matches
            fuzzy_score = 0
            for fm in fuzzy_matches:
                if fm["id"] == entity["id"]:
                    fuzzy_score = fm["fuzzy_score"]
                    break
            
            # Combine scores: prioritize fuzzy matches for typos
            if fuzzy_score > 0:
                final_score = 0.7 * fuzzy_score + 0.3 * semantic_score
            else:
                final_score = semantic_score
            
            if final_score >= threshold:
                combined_scores.append({
                    **entity,
                    "similarity": float(final_score),
                    "semantic_score": float(semantic_score),
                    "fuzzy_score": float(fuzzy_score)
                })
        
        # Sort by similarity
        combined_scores.sort(key=lambda x: x["similarity"], reverse=True)
        
        search_time = time.time() - start_time
        
        # Determine match type
        if len(combined_scores) == 1 and combined_scores[0]["similarity"] >= 0.85:
            return combined_scores, search_time, "exact"
        elif len(combined_scores) > 0 and combined_scores[0]["similarity"] >= 0.85 and \
             (len(combined_scores) == 1 or combined_scores[0]["similarity"] - combined_scores[1]["similarity"] > 0.1):
            return [combined_scores[0]], search_time, "exact"
        
        return combined_scores, search_time, "ambiguous"


def evaluate_performance(disambiguator: HybridEntityDisambiguator, test_cases: List[Dict]):
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
    # Initialize hybrid disambiguator
    disambiguator = HybridEntityDisambiguator()
    
    # EXACT SAME entities as before
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
    
    # EXACT SAME test queries as before
    test_queries = [
        "John",  # Ambiguous - should return all Johns
        "Software Engineer at Google",  # More specific - should match John Smith
        "Olympic Athlete",  # Should match Michael Johnson
        "Singer Songwriter",  # Should match John Legend
        "Composer",  # Should match John Williams
        "CEO Tech",  # Should match Sarah Johnson
    ]
    
    print("\n" + "="*60)
    print("TESTING HYBRID ENTITY DISAMBIGUATION")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results, search_time, match_type = disambiguator.search(query, entities, entity_embeddings)
        print(f"Match type: {match_type}")
        print(f"Search time: {search_time*1000:.2f} ms")
        print(f"Found {len(results)} matches:")
        
        for result in results[:5]:  # Show top 5
            details = f"  - ID: {result['id']}, {result['descriptor']} (similarity: {result['similarity']:.3f}"
            if 'fuzzy_score' in result and result['fuzzy_score'] > 0:
                details += f", fuzzy: {result['fuzzy_score']:.3f}"
            if 'semantic_score' in result:
                details += f", semantic: {result['semantic_score']:.3f}"
            details += ")"
            print(details)
    
    # Test exact match and typo cases
    print("\n" + "="*60)
    print("EXACT MATCH AND TYPO TESTS")
    print("="*60)
    
    typo_queries = [
        "John Smith",  # Exact match
        "Jhon Smith",  # Common typo
        "john smith",  # Lowercase
        "JOHN SMITH",  # Uppercase
        "Michael Johnson",  # Exact match
        "Micheal Johnson",  # Common typo
    ]
    
    for query in typo_queries:
        print(f"\nQuery: '{query}'")
        results, search_time, match_type = disambiguator.search(query, entities, entity_embeddings, threshold=0.3)
        print(f"Match type: {match_type}")
        print(f"Found {len(results)} matches:")
        
        for result in results[:3]:
            details = f"  - {result['descriptor']} (similarity: {result['similarity']:.3f}"
            if 'match_type' in result:
                details += f", type: {result['match_type']}"
            if 'fuzzy_score' in result and result['fuzzy_score'] > 0:
                details += f", fuzzy: {result['fuzzy_score']:.3f}"
            details += ")"
            print(details)