from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import time
import re
from difflib import SequenceMatcher
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EntityHandler(ABC):
    """Abstract base class for entity-specific logic"""
    
    @abstractmethod
    def extract_parts(self, entity_descriptor: str) -> Dict:
        """Extract entity-specific components"""
        pass
    
    @abstractmethod
    def get_exact_match_variations(self, query: str) -> List[str]:
        """Generate variations for exact matching"""
        pass
    
    @abstractmethod
    def calculate_custom_score(self, query: str, entity: str, base_score: float) -> float:
        """Apply entity-specific scoring adjustments"""
        pass
    
    @abstractmethod
    def get_query_type(self, query: str) -> str:
        """Detect query type for this entity type"""
        pass
    
    @abstractmethod
    def normalize_query(self, query: str) -> str:
        """Normalize query for this entity type"""
        pass


class PersonNameHandler(EntityHandler):
    """Handler for person names (existing implementation)"""
    
    def extract_parts(self, entity_descriptor: str) -> Dict:
        # Extract name from descriptor
        parts = re.split(r' - | at | of ', entity_descriptor)
        full_name = parts[0].strip()
        name_parts = full_name.split()
        
        middle_parts = name_parts[1:-1] if len(name_parts) > 2 else []
        
        return {
            "full": full_name,
            "first": name_parts[0] if name_parts else "",
            "last": name_parts[-1] if len(name_parts) > 1 else "",
            "middle": middle_parts,
            "parts": name_parts,
            "initials": [p[0].upper() for p in name_parts if p]
        }
    
    def get_exact_match_variations(self, query: str) -> List[str]:
        """Generate name variations"""
        variations = [query, query.lower(), query.upper()]
        
        # Add variations without middle names
        parts = query.split()
        if len(parts) > 2:
            variations.append(f"{parts[0]} {parts[-1]}")
        
        return variations
    
    def calculate_custom_score(self, query: str, entity: str, base_score: float) -> float:
        """Apply name-specific scoring"""
        # Already handled in main implementation
        return base_score
    
    def get_query_type(self, query: str) -> str:
        words = query.split()
        
        if len(words) == 1:
            return "partial_name"
        elif len(words) == 2 and all(word[0].isupper() or word.islower() for word in words):
            return "full_name"
        else:
            return "semantic"
    
    def normalize_query(self, query: str) -> str:
        return ' '.join(query.lower().split())


class LocationHandler(EntityHandler):
    """Handler for location entities"""
    
    def __init__(self):
        self.abbreviations = {
            "nyc": "new york city",
            "la": "los angeles",
            "sf": "san francisco",
            "dc": "washington dc",
            "uk": "united kingdom",
            "us": "united states",
            "usa": "united states of america"
        }
        
        self.suffixes = ["city", "town", "village", "county", "state", "province"]
        
    def extract_parts(self, entity_descriptor: str) -> Dict:
        # Parse location components
        # Handle formats: "City, State", "City - Description"
        parts = re.split(r' - | at ', entity_descriptor)
        location = parts[0].strip()
        
        # Split by comma for city, state format
        components = [c.strip() for c in location.split(',')]
        
        return {
            "full": location,
            "primary": components[0] if components else "",
            "secondary": components[1] if len(components) > 1 else "",
            "components": components,
            "normalized": self.normalize_location(location)
        }
    
    def normalize_location(self, location: str) -> str:
        """Normalize location name"""
        normalized = location.lower()
        
        # Expand abbreviations
        for abbr, full in self.abbreviations.items():
            normalized = normalized.replace(abbr, full)
        
        # Remove common suffixes
        for suffix in self.suffixes:
            normalized = re.sub(f"\\b{suffix}\\b", "", normalized)
        
        # Standardize spacing
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def get_exact_match_variations(self, query: str) -> List[str]:
        variations = [query, query.lower()]
        
        # Add expanded version
        normalized = self.normalize_location(query)
        if normalized != query.lower():
            variations.append(normalized)
        
        # Add abbreviated version
        for full, abbr in [(v, k) for k, v in self.abbreviations.items()]:
            if full in query.lower():
                variations.append(query.lower().replace(full, abbr))
        
        return list(set(variations))
    
    def calculate_custom_score(self, query: str, entity: str, base_score: float) -> float:
        """Adjust score based on location hierarchy"""
        query_parts = self.extract_parts(query)
        entity_parts = self.extract_parts(entity)
        
        # Boost if query is contained in entity (e.g., "California" in "San Francisco, California")
        if query_parts["normalized"] in entity_parts["normalized"]:
            return min(base_score * 1.2, 1.0)
        
        return base_score
    
    def get_query_type(self, query: str) -> str:
        # Single word locations are partial
        if len(query.split()) == 1:
            return "partial_location"
        
        # Check if it contains location indicators
        if any(sep in query for sep in [',', ' - ']):
            return "full_location"
        
        return "semantic"
    
    def normalize_query(self, query: str) -> str:
        return self.normalize_location(query)


class WorkRoleHandler(EntityHandler):
    """Handler for work roles and job titles"""
    
    def __init__(self):
        self.synonyms = {
            "developer": ["developer", "engineer", "programmer"],
            "manager": ["manager", "lead", "head"],
            "analyst": ["analyst", "specialist"],
            "designer": ["designer", "ux", "ui"],
        }
        
        self.levels = ["junior", "senior", "lead", "principal", "staff", "associate"]
        self.domains = ["software", "data", "product", "marketing", "sales", "hr"]
        
    def extract_parts(self, entity_descriptor: str) -> Dict:
        # Extract role from descriptor
        parts = re.split(r' - | at ', entity_descriptor)
        role = parts[0].strip()
        
        # Identify components
        role_lower = role.lower()
        found_level = None
        found_domain = None
        
        for level in self.levels:
            if level in role_lower:
                found_level = level
                break
        
        for domain in self.domains:
            if domain in role_lower:
                found_domain = domain
                break
        
        # Extract core role (remove level and domain)
        core_role = role
        if found_level:
            core_role = core_role.replace(found_level, "").replace(found_level.title(), "")
        if found_domain:
            core_role = core_role.replace(found_domain, "").replace(found_domain.title(), "")
        
        core_role = ' '.join(core_role.split())
        
        return {
            "full": role,
            "level": found_level,
            "domain": found_domain,
            "core": core_role,
            "normalized": role_lower
        }
    
    def expand_synonyms(self, role: str) -> List[str]:
        """Expand role to include synonyms"""
        expanded = [role]
        role_lower = role.lower()
        
        for base, synonyms in self.synonyms.items():
            if base in role_lower:
                for syn in synonyms:
                    if syn != base:
                        expanded.append(role_lower.replace(base, syn))
        
        return expanded
    
    def get_exact_match_variations(self, query: str) -> List[str]:
        variations = [query, query.lower()]
        
        # Add synonym variations
        variations.extend(self.expand_synonyms(query))
        
        # Add common abbreviations
        abbreviations = {
            "software engineer": ["swe", "software eng"],
            "product manager": ["pm", "prod manager"],
            "user experience": ["ux"],
            "user interface": ["ui"],
        }
        
        query_lower = query.lower()
        for full, abbrevs in abbreviations.items():
            if full in query_lower:
                for abbr in abbrevs:
                    variations.append(query_lower.replace(full, abbr))
            for abbr in abbrevs:
                if abbr in query_lower:
                    variations.append(query_lower.replace(abbr, full))
        
        return list(set(variations))
    
    def calculate_custom_score(self, query: str, entity: str, base_score: float) -> float:
        query_parts = self.extract_parts(query)
        entity_parts = self.extract_parts(entity)
        
        # Penalty for level mismatch
        if query_parts["level"] and entity_parts["level"]:
            if query_parts["level"] != entity_parts["level"]:
                base_score *= 0.8
        
        # Bonus for exact core role match
        if query_parts["core"].lower() == entity_parts["core"].lower():
            base_score = min(base_score * 1.1, 1.0)
        
        return base_score
    
    def get_query_type(self, query: str) -> str:
        words = query.split()
        
        if len(words) == 1:
            return "partial_role"
        
        # Check if it's a common role pattern
        query_lower = query.lower()
        if any(level in query_lower for level in self.levels):
            return "full_role"
        
        if any(domain in query_lower for domain in self.domains):
            return "full_role"
        
        return "semantic"
    
    def normalize_query(self, query: str) -> str:
        # Expand common abbreviations
        normalized = query.lower()
        replacements = {
            "swe": "software engineer",
            "pm": "product manager",
            "eng": "engineer",
        }
        
        for abbr, full in replacements.items():
            normalized = re.sub(f"\\b{abbr}\\b", full, normalized)
        
        return ' '.join(normalized.split())


class GeneralizedEntityDisambiguator:
    """Main disambiguator that works with any entity type"""
    
    def __init__(self, model_name: str, entity_type: str, model_loader):
        self.model_name = model_name
        self.entity_type = entity_type
        self.model = model_loader(model_name)
        self.handler = self.create_handler(entity_type)
        
    def create_handler(self, entity_type: str) -> EntityHandler:
        handlers = {
            "person": PersonNameHandler(),
            "location": LocationHandler(),
            "role": WorkRoleHandler(),
        }
        
        if entity_type not in handlers:
            raise ValueError(f"Unknown entity type: {entity_type}")
        
        return handlers[entity_type]
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        if hasattr(self.model, 'encode'):
            # Sentence transformer style
            return self.model.encode(texts, convert_to_numpy=True)
        else:
            # Model2vec style
            return self.model.encode(texts)
    
    def create_entity_embeddings(self, entities: List[Dict[str, str]]) -> Dict:
        """Create embeddings with entity-specific handling"""
        # Extract parts for each entity
        entity_parts = []
        for entity in entities:
            parts = self.handler.extract_parts(entity["descriptor"])
            entity_parts.append(parts)
        
        # Get descriptors and normalized versions
        descriptors = [entity["descriptor"] for entity in entities]
        normalized = [self.handler.normalize_query(desc) for desc in descriptors]
        
        # Create embeddings
        embeddings = {
            "original": self.encode(descriptors),
            "normalized": self.encode(normalized),
            "parts": entity_parts
        }
        
        return embeddings
    
    def search(self, query: str, entities: List[Dict[str, str]], 
              entity_embeddings: Dict, threshold: float = 0.5) -> Tuple[List[Dict], float, str]:
        """Search with entity-specific logic"""
        start_time = time.time()
        
        # Detect query type
        query_type = self.handler.get_query_type(query)
        
        # Get query variations
        query_variations = self.handler.get_exact_match_variations(query)
        
        # Check exact matches first
        exact_matches = []
        for idx, entity in enumerate(entities):
            entity_parts = entity_embeddings["parts"][idx]
            
            # Check all variations
            for variation in query_variations:
                if variation in entity_parts.get("normalized", "").lower():
                    exact_matches.append({
                        **entity,
                        "similarity": 1.0,
                        "match_type": "exact"
                    })
                    break
        
        if exact_matches:
            search_time = time.time() - start_time
            return exact_matches, search_time, "exact"
        
        # Fuzzy matching
        fuzzy_matches = []
        for idx, entity in enumerate(entities):
            normalized_entity = self.handler.normalize_query(entity["descriptor"])
            normalized_query = self.handler.normalize_query(query)
            
            fuzzy_score = SequenceMatcher(None, normalized_query, normalized_entity).ratio()
            
            if fuzzy_score >= 0.85:
                fuzzy_matches.append({
                    **entity,
                    "similarity": fuzzy_score * 0.95,
                    "match_type": "fuzzy"
                })
        
        # Semantic search
        query_emb = self.encode([query]).reshape(1, -1)
        similarities = cosine_similarity(query_emb, entity_embeddings["original"])[0]
        
        # Combine results
        all_matches = []
        for idx, entity in enumerate(entities):
            # Check if already in fuzzy matches
            is_fuzzy = any(m["id"] == entity["id"] for m in fuzzy_matches)
            
            if is_fuzzy:
                continue
            
            # Apply custom scoring
            base_score = float(similarities[idx])
            final_score = self.handler.calculate_custom_score(
                query, entity["descriptor"], base_score
            )
            
            if final_score >= threshold:
                all_matches.append({
                    **entity,
                    "similarity": final_score,
                    "match_type": "semantic"
                })
        
        # Combine and sort
        all_matches.extend(fuzzy_matches)
        all_matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        search_time = time.time() - start_time
        match_type = "exact" if len(all_matches) == 1 else "ambiguous"
        
        return all_matches, search_time, match_type


# Example usage
if __name__ == "__main__":
    print("GENERALIZED ENTITY DISAMBIGUATION FRAMEWORK")
    print("=" * 80)
    
    # Test with different entity types
    test_cases = [
        {
            "entity_type": "location",
            "entities": [
                {"id": "1", "descriptor": "New York City - Financial District"},
                {"id": "2", "descriptor": "San Francisco, California - Tech Hub"},
                {"id": "3", "descriptor": "Los Angeles, CA - Entertainment Capital"},
                {"id": "4", "descriptor": "NYC Metro Area - Greater New York"},
            ],
            "queries": ["NYC", "California", "new york", "LA"]
        },
        {
            "entity_type": "role",
            "entities": [
                {"id": "1", "descriptor": "Senior Software Engineer - Backend Systems"},
                {"id": "2", "descriptor": "Software Developer - Full Stack"},
                {"id": "3", "descriptor": "Product Manager - Mobile Apps"},
                {"id": "4", "descriptor": "Junior Developer - Frontend"},
            ],
            "queries": ["SWE", "Developer", "PM", "Senior Engineer"]
        }
    ]
    
    # Mock model loader
    class MockModel:
        def encode(self, texts):
            # Simple mock encoding
            return np.random.randn(len(texts), 128)
    
    def mock_model_loader(model_name):
        return MockModel()
    
    for test in test_cases:
        print(f"\n\nTesting {test['entity_type'].upper()} entities:")
        print("-" * 60)
        
        disambiguator = GeneralizedEntityDisambiguator(
            model_name="mock-model",
            entity_type=test["entity_type"],
            model_loader=mock_model_loader
        )
        
        embeddings = disambiguator.create_entity_embeddings(test["entities"])
        
        for query in test["queries"]:
            results, time_taken, match_type = disambiguator.search(
                query, test["entities"], embeddings, threshold=0.4
            )
            
            print(f"\nQuery: '{query}' ({match_type})")
            for r in results[:3]:
                print(f"  - {r['descriptor']} (score: {r['similarity']:.3f})")