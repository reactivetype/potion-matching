import time
from typing import List, Dict, Tuple
from model2vec import StaticModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from difflib import SequenceMatcher


class ImprovedEntityDisambiguator:
    def __init__(self, model_name: str = "minishlab/potion-multilingual-128M"):
        print(f"Loading model: {model_name}")
        start_time = time.time()
        self.model = StaticModel.from_pretrained(model_name)
        self.load_time = time.time() - start_time
        print(f"Model loaded in {self.load_time:.2f} seconds")
        
    def preprocess_text(self, text: str) -> str:
        """Normalize text for better matching"""
        normalized = text.lower()
        normalized = ' '.join(normalized.split())
        return normalized
    
    def extract_name(self, descriptor: str) -> str:
        """Extract the name part from a descriptor"""
        parts = re.split(r' - | at | of ', descriptor)
        return parts[0].strip()
    
    def extract_name_parts(self, descriptor: str) -> Dict[str, str]:
        """Extract first, last, and full name from descriptor"""
        full_name = self.extract_name(descriptor)
        parts = full_name.split()
        
        # Extract middle names/initials
        middle_parts = parts[1:-1] if len(parts) > 2 else []
        
        return {
            "full": full_name,
            "first": parts[0] if parts else "",
            "last": parts[-1] if len(parts) > 1 else "",
            "middle": middle_parts,
            "parts": parts,
            "initials": [p[0].upper() for p in parts if p]  # First letter of each part
        }
    
    def fuzzy_match_score(self, s1: str, s2: str) -> float:
        """Calculate fuzzy string matching score"""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def check_name_match_with_initials(self, query_parts: Dict[str, str], entity_parts: Dict[str, str]) -> Tuple[bool, float, str]:
        """
        Check if names match considering middle names and initials.
        Returns (is_match, score, match_type)
        """
        query_tokens = query_parts["parts"]
        entity_tokens = entity_parts["parts"]
        
        # Exact full name match
        if query_parts["full"].lower() == entity_parts["full"].lower():
            return True, 1.0, "exact_full_name"
        
        # If query has 2+ parts, check for name matching with/without middle names
        if len(query_tokens) >= 2:
            query_first = query_tokens[0].lower()
            query_last = query_tokens[-1].lower()
            entity_first = entity_parts["first"].lower()
            entity_last = entity_parts["last"].lower()
            
            # Check first and last name match
            if query_first == entity_first and query_last == entity_last:
                if len(query_tokens) == 2 and len(entity_tokens) > 2:
                    # Query is "John Smith", entity is "John Michael Smith"
                    return True, 0.95, "name_without_middle"
                elif len(query_tokens) > 2:
                    # Check middle names/initials
                    query_middle = [m.lower() for m in query_tokens[1:-1]]
                    entity_middle = [m.lower() for m in entity_parts["middle"]]
                    
                    # Check if middle names match exactly
                    if query_middle == entity_middle:
                        return True, 0.98, "name_with_middle"
                    
                    # Check if query has initials that match entity middle names
                    if all(len(qm) == 2 and qm.endswith('.') for qm in query_middle):
                        # Query has initials like "M."
                        query_initials = [qm[0].upper() for qm in query_middle]
                        entity_middle_initials = [em[0].upper() for em in entity_middle]
                        if query_initials == entity_middle_initials:
                            return True, 0.96, "name_with_middle_initial"
                    
                    # Check if entity has initials that match query middle names
                    if all(len(em) == 2 and em.endswith('.') for em in entity_middle):
                        entity_initials = [em[0].upper() for em in entity_middle]
                        query_middle_initials = [qm[0].upper() for qm in query_middle]
                        if entity_initials == query_middle_initials:
                            return True, 0.96, "name_matches_middle_initial"
        
        return False, 0.0, ""
    
    def detect_query_type(self, query: str) -> str:
        """Detect if query is likely a first name, last name, full name, or semantic"""
        words = query.split()
        
        if len(words) == 1:
            # Single word - likely first or last name
            return "partial_name"
        elif len(words) == 2 and all(word[0].isupper() or word.islower() for word in words):
            # Two words, both capitalized or lowercase - likely full name
            return "full_name"
        else:
            # Multiple words or contains non-name words
            return "semantic"
    
    def create_entity_embeddings(self, entities: List[Dict[str, str]]) -> Dict[str, any]:
        """Create embeddings for entity descriptors with name parts"""
        # Extract all name components
        entity_names = []
        for entity in entities:
            name_parts = self.extract_name_parts(entity["descriptor"])
            entity_names.append(name_parts)
        
        # Original descriptors
        descriptors = [entity["descriptor"] for entity in entities]
        normalized_descriptors = [self.preprocess_text(desc) for desc in descriptors]
        
        # Full names
        full_names = [name["full"] for name in entity_names]
        normalized_names = [self.preprocess_text(name) for name in full_names]
        
        # First names only
        first_names = [name["first"] for name in entity_names]
        
        # Last names only
        last_names = [name["last"] if name["last"] else name["first"] for name in entity_names]
        
        # Create embeddings
        embeddings = {
            "original": self.model.encode(descriptors),
            "normalized": self.model.encode(normalized_descriptors),
            "names": self.model.encode(full_names),
            "normalized_names": self.model.encode(normalized_names),
            "first_names": self.model.encode(first_names),
            "last_names": self.model.encode(last_names),
            "name_parts": entity_names
        }
        
        return embeddings
    
    def search(self, query: str, entities: List[Dict[str, str]], 
              entity_embeddings: Dict[str, any], threshold: float = 0.5) -> Tuple[List[Dict[str, str]], float, str]:
        """Improved search with better partial name handling"""
        start_time = time.time()
        
        # Detect query type
        query_type = self.detect_query_type(query)
        query_normalized = self.preprocess_text(query)
        
        # For partial name queries (single word), use special handling
        if query_type == "partial_name":
            matches = []
            
            for idx, entity in enumerate(entities):
                name_parts = entity_embeddings["name_parts"][idx]
                score = 0.0
                match_type = ""
                
                # Check if query is a single letter (potential initial)
                if len(query) == 1:
                    query_upper = query.upper()
                    # Check against all initials in the name
                    for i, initial in enumerate(name_parts["initials"]):
                        if query_upper == initial:
                            if i == 0:
                                score = 0.85
                                match_type = "first_initial"
                            elif i == len(name_parts["initials"]) - 1:
                                score = 0.85
                                match_type = "last_initial"
                            else:
                                score = 0.80
                                match_type = "middle_initial"
                            break
                # Check exact first name match
                elif query.lower() == name_parts["first"].lower():
                    score = 0.95
                    match_type = "exact_first_name"
                # Check exact last name match
                elif query.lower() == name_parts["last"].lower():
                    score = 0.95
                    match_type = "exact_last_name"
                # Check if query matches any name part exactly (for middle names)
                elif any(query.lower() == part.lower() for part in name_parts["parts"]):
                    score = 0.90
                    match_type = "exact_name_part"
                else:
                    # Fall back to semantic similarity with first/last names
                    query_emb = self.model.encode([query]).reshape(1, -1)
                    
                    # Check similarity with first name
                    first_sim = cosine_similarity(query_emb, entity_embeddings["first_names"][idx].reshape(1, -1))[0][0]
                    # Check similarity with last name
                    last_sim = cosine_similarity(query_emb, entity_embeddings["last_names"][idx].reshape(1, -1))[0][0]
                    
                    # Take max but apply a stronger penalty for semantic matching on partial names
                    # This reduces false positives like "John" matching "Johnson" semantically
                    score = max(first_sim, last_sim) * 0.7
                    match_type = "semantic_name"
                
                if score >= threshold:
                    matches.append({
                        **entity,
                        "similarity": float(score),
                        "match_type": match_type
                    })
            
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            search_time = time.time() - start_time
            
            return matches, search_time, "partial" if len(matches) > 1 else "exact"
        
        # For full names and semantic queries, use hybrid approach
        else:
            # Parse query name parts
            query_parts = self.extract_name_parts(query)
            
            # Check for exact matches first, including middle name handling
            exact_matches = []
            for idx, entity in enumerate(entities):
                entity_parts = entity_embeddings["name_parts"][idx]
                
                # Check exact descriptor match
                if query.lower() == entity["descriptor"].lower():
                    exact_matches.append({
                        **entity,
                        "similarity": 1.0,
                        "match_type": "exact_descriptor"
                    })
                    continue
                
                # Check name matching with middle names/initials
                is_match, score, match_type = self.check_name_match_with_initials(query_parts, entity_parts)
                if is_match:
                    exact_matches.append({
                        **entity,
                        "similarity": score,
                        "match_type": match_type
                    })
            
            if exact_matches:
                # Sort by score to prioritize exact matches
                exact_matches.sort(key=lambda x: x["similarity"], reverse=True)
                search_time = time.time() - start_time
                return exact_matches, search_time, "exact"
            
            # Check fuzzy matches for typos
            fuzzy_matches = []
            for idx, entity in enumerate(entities):
                name_parts = entity_embeddings["name_parts"][idx]
                fuzzy_score = self.fuzzy_match_score(query, name_parts["full"])
                
                if fuzzy_score >= 0.85:
                    fuzzy_matches.append({
                        **entity,
                        "similarity": fuzzy_score * 0.95,  # Slightly reduce to prioritize exact
                        "match_type": "fuzzy",
                        "fuzzy_score": fuzzy_score
                    })
            
            # Semantic search
            query_embeddings = {
                "original": self.model.encode([query]).reshape(1, -1),
                "normalized": self.model.encode([query_normalized]).reshape(1, -1)
            }
            
            # Calculate similarities
            all_similarities = {}
            all_similarities["orig_orig"] = cosine_similarity(
                query_embeddings["original"], 
                entity_embeddings["original"]
            )[0]
            all_similarities["norm_norm"] = cosine_similarity(
                query_embeddings["normalized"], 
                entity_embeddings["normalized"]
            )[0]
            all_similarities["orig_names"] = cosine_similarity(
                query_embeddings["original"], 
                entity_embeddings["names"]
            )[0]
            
            # Combine scores
            combined_scores = []
            for idx, entity in enumerate(entities):
                # Get max semantic similarity
                semantic_score = max(
                    all_similarities["orig_orig"][idx],
                    all_similarities["norm_norm"][idx],
                    all_similarities["orig_names"][idx]
                )
                
                # Check if already in fuzzy matches
                fuzzy_score = 0
                for fm in fuzzy_matches:
                    if fm["id"] == entity["id"]:
                        fuzzy_score = fm["similarity"]
                        break
                
                # Use fuzzy score if available, otherwise semantic
                final_score = fuzzy_score if fuzzy_score > 0 else semantic_score
                
                if final_score >= threshold:
                    combined_scores.append({
                        **entity,
                        "similarity": float(final_score),
                        "semantic_score": float(semantic_score)
                    })
            
            combined_scores.sort(key=lambda x: x["similarity"], reverse=True)
            search_time = time.time() - start_time
            
            if len(combined_scores) == 1 and combined_scores[0]["similarity"] >= 0.85:
                return combined_scores, search_time, "exact"
            
            return combined_scores, search_time, "ambiguous"


if __name__ == "__main__":
    # Test the improved disambiguator
    disambiguator = ImprovedEntityDisambiguator()
    
    entities = [
        {"id": "1", "descriptor": "John Smith - Software Engineer at Google"},
        {"id": "2", "descriptor": "John Smith - Professor of Physics at MIT"},
        {"id": "3", "descriptor": "John Doe - Data Scientist at Microsoft"},
        {"id": "4", "descriptor": "Jane Smith - Product Manager at Apple"},
        {"id": "5", "descriptor": "Michael Johnson - Olympic Athlete"},
        {"id": "6", "descriptor": "John Williams - Composer"},
        {"id": "7", "descriptor": "Sarah Johnson - CEO of Tech Startup"},
    ]
    
    entity_embeddings = disambiguator.create_entity_embeddings(entities)
    
    # Test queries
    test_queries = [
        ("John", "Should match all Johns"),
        ("Smith", "Should match all Smiths"),
        ("Johnson", "Should match all Johnsons"),
        ("John Smith", "Should match both John Smiths"),
        ("Jhon Smith", "Typo - should match John Smiths"),
        ("Michael", "Should match Michael Johnson"),
        ("Software Engineer", "Semantic - should match John Smith at Google"),
    ]
    
    print("\n" + "="*80)
    print("IMPROVED ENTITY DISAMBIGUATION TEST")
    print("="*80)
    
    for test in test_queries:
        query = test[0]
        expected = test[1]
        # Use higher threshold for partial name queries to reduce false positives
        threshold = 0.5 if len(query.split()) == 1 else 0.4
        
        print(f"\nQuery: '{query}' - {expected}")
        results, search_time, match_type = disambiguator.search(query, entities, entity_embeddings, threshold=threshold)
        print(f"Match type: {match_type}, Search time: {search_time*1000:.2f} ms, Threshold: {threshold}")
        print(f"Found {len(results)} matches:")
        
        for i, result in enumerate(results[:5]):
            match_info = f"  {i+1}. {result['descriptor']} (similarity: {result['similarity']:.3f}"
            if 'match_type' in result:
                match_info += f", type: {result['match_type']}"
            match_info += ")"
            print(match_info)