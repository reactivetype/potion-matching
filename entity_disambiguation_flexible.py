import time
from typing import List, Dict, Tuple, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Try to import both model types
try:
    from model2vec import StaticModel
    STATICMODEL_AVAILABLE = True
except ImportError:
    STATICMODEL_AVAILABLE = False
    print("Warning: model2vec not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCETRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCETRANSFORMER_AVAILABLE = False
    print("Warning: sentence-transformers not available")


class FlexibleEntityDisambiguator:
    """Entity disambiguator that can use either StaticModel or SentenceTransformer"""
    
    def __init__(self, model_name: str = "minishlab/potion-multilingual-128M", 
                 model_type: str = "auto"):
        """
        Initialize with specified model.
        
        Args:
            model_name: Model identifier
            model_type: "static", "sentence-transformer", or "auto"
        """
        self.model_name = model_name
        self.model_type = model_type
        
        print(f"Loading model: {model_name}")
        start_time = time.time()
        
        if model_type == "auto":
            # Auto-detect based on model name
            if "sentence-transformers" in model_name or "MiniLM" in model_name:
                model_type = "sentence-transformer"
            else:
                model_type = "static"
        
        if model_type == "static" and STATICMODEL_AVAILABLE:
            self.model = StaticModel.from_pretrained(model_name)
            self.model_type_loaded = "static"
        elif model_type == "sentence-transformer" and SENTENCETRANSFORMER_AVAILABLE:
            self.model = SentenceTransformer(model_name)
            self.model_type_loaded = "sentence-transformer"
        else:
            raise ValueError(f"Model type '{model_type}' not available or not installed")
        
        self.load_time = time.time() - start_time
        print(f"Model loaded in {self.load_time:.2f} seconds (type: {self.model_type_loaded})")
    
    def encode(self, texts: Union[List[str], str]) -> np.ndarray:
        """Encode texts using the loaded model"""
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model_type_loaded == "static":
            return self.model.encode(texts)
        else:  # sentence-transformer
            return self.model.encode(texts, convert_to_numpy=True)
    
    def create_entity_embeddings(self, entities: List[Dict[str, str]]) -> np.ndarray:
        """Create embeddings for entity descriptors"""
        descriptors = [entity["descriptor"] for entity in entities]
        return self.encode(descriptors)
    
    def search(self, query: str, entities: List[Dict[str, str]], 
              entity_embeddings: np.ndarray, threshold: float = 0.5) -> Tuple[List[Dict[str, str]], float, str]:
        """Search for matching entities based on query (baseline semantic only)"""
        start_time = time.time()
        
        # Encode the query
        query_embedding = self.encode([query]).reshape(1, -1)
        
        # Calculate cosine similarities
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
        
        # Determine match type
        if len(matches) == 1 and matches[0]["similarity"] >= 0.85:
            return matches, search_time, "exact"
        elif len(matches) > 0 and matches[0]["similarity"] >= 0.85 and \
             (len(matches) == 1 or matches[0]["similarity"] - matches[1]["similarity"] > 0.1):
            return [matches[0]], search_time, "exact"
        
        return matches, search_time, "ambiguous"


if __name__ == "__main__":
    # Test with both models
    print("="*80)
    print("TESTING FLEXIBLE ENTITY DISAMBIGUATOR")
    print("="*80)
    
    entities = [
        {"id": "1", "descriptor": "John Smith - Software Engineer at Google"},
        {"id": "2", "descriptor": "John Michael Smith - Professor at MIT"},
        {"id": "3", "descriptor": "John Doe - Data Scientist at Microsoft"},
        {"id": "4", "descriptor": "Jane Smith - Product Manager at Apple"},
        {"id": "5", "descriptor": "Michael Johnson - Olympic Athlete"},
    ]
    
    test_queries = ["John", "John Smith", "Software Engineer", "Michael"]
    
    # Test POTION model
    print("\n1. Testing with POTION static model:")
    print("-"*50)
    try:
        potion_disambiguator = FlexibleEntityDisambiguator(
            model_name="minishlab/potion-multilingual-128M",
            model_type="static"
        )
        potion_embeddings = potion_disambiguator.create_entity_embeddings(entities)
        
        for query in test_queries:
            results, search_time, match_type = potion_disambiguator.search(
                query, entities, potion_embeddings, threshold=0.4
            )
            print(f"\nQuery: '{query}' - Found {len(results)} matches in {search_time*1000:.2f}ms")
    except Exception as e:
        print(f"Error with POTION model: {e}")
    
    # Test MiniLM model
    print("\n\n2. Testing with MiniLM sentence transformer:")
    print("-"*50)
    try:
        minilm_disambiguator = FlexibleEntityDisambiguator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_type="sentence-transformer"
        )
        minilm_embeddings = minilm_disambiguator.create_entity_embeddings(entities)
        
        for query in test_queries:
            results, search_time, match_type = minilm_disambiguator.search(
                query, entities, minilm_embeddings, threshold=0.4
            )
            print(f"\nQuery: '{query}' - Found {len(results)} matches in {search_time*1000:.2f}ms")
    except Exception as e:
        print(f"Error with MiniLM model: {e}")