# Entity Disambiguation: Generalization and Re-ranking Analysis

## Part 1: Generalizing to Other Entity Types

### Current Architecture Overview

Our current implementation for person names has these key components:

1. **Query Type Detection** - Identifies partial vs full vs semantic queries
2. **Entity Part Extraction** - Breaks down entities into components (first/last/middle names)
3. **Multiple Matching Strategies** - Exact, fuzzy, partial, semantic
4. **Dynamic Thresholds** - Different thresholds for different query types
5. **Hierarchical Scoring** - Prioritizes exact matches over fuzzy over semantic

### What Can Be Reused (Core Framework)

```python
class BaseEntityDisambiguator:
    """Base class for all entity types"""
    
    def __init__(self, model_name, entity_type):
        self.model = self.load_model(model_name)
        self.entity_type = entity_type
        self.entity_handler = self.get_entity_handler(entity_type)
    
    # Reusable methods:
    - encode()
    - cosine_similarity()
    - fuzzy_match_score()
    - create_entity_embeddings()
    - search() # with entity-specific customization hooks
```

### Entity-Specific Requirements

#### 1. **Locations**

**Unique Challenges:**
- Hierarchical relationships (neighborhood → city → state → country)
- Multiple naming conventions (NYC, New York City, New York, NY)
- Abbreviations and variations (St. vs Saint, Mt. vs Mount)
- Geocoding considerations

**Custom Components Needed:**
```python
class LocationEntityHandler:
    def extract_parts(self, location: str) -> Dict:
        # Extract city, state, country, postal code
        # Handle "San Francisco, CA, USA" format
        # Parse addresses vs place names
        
    def normalize_location(self, location: str) -> str:
        # Expand abbreviations: NYC → New York City
        # Standardize: St. → Saint, Mt. → Mount
        # Handle international variations
        
    def calculate_proximity_bonus(self, query_loc, entity_loc) -> float:
        # Bonus for nearby locations
        # Hierarchical containment bonus
```

**Example Matching Logic:**
- "NYC" → matches "New York City", "New York, NY", "Manhattan, NYC"
- "California" → matches all CA cities with lower scores
- "St. Louis" → matches "Saint Louis"

#### 2. **Work Roles/Job Titles**

**Unique Challenges:**
- Synonym groups (Developer ↔ Engineer ↔ Programmer)
- Hierarchical roles (Senior/Junior/Lead variations)
- Domain specializations (Frontend/Backend/Full-stack)
- Industry-specific terminology

**Custom Components Needed:**
```python
class RoleEntityHandler:
    def __init__(self):
        self.synonym_groups = self.load_role_synonyms()
        self.hierarchy_map = self.load_role_hierarchy()
        
    def extract_parts(self, role: str) -> Dict:
        # Extract: level (Senior), core_role (Engineer), 
        # specialization (Frontend), domain (Software)
        
    def expand_synonyms(self, role: str) -> List[str]:
        # Developer → [Developer, Engineer, Programmer]
        # PM → [Product Manager, Program Manager]
        
    def calculate_hierarchy_score(self, query_role, entity_role) -> float:
        # "Developer" matches "Senior Developer" with penalty
        # "Senior Developer" doesn't match "Junior Developer"
```

**Example Matching Logic:**
- "SWE" → "Software Engineer", "Software Developer"
- "Developer" → matches all developer roles with specialization penalties
- "Senior Engineer" → matches "Senior Software Engineer" but not "Junior Engineer"

#### 3. **Organizations/Companies**

**Unique Challenges:**
- Legal entity variations (Inc., LLC, Corporation)
- Brand names vs legal names
- Acronyms and abbreviations
- Mergers and acquisitions history
- Subsidiaries and parent companies

**Custom Components Needed:**
```python
class OrganizationEntityHandler:
    def __init__(self):
        self.legal_suffixes = ["Inc", "LLC", "Corp", "Ltd", "GmbH"]
        self.known_aliases = self.load_company_aliases()
        
    def extract_parts(self, org: str) -> Dict:
        # Extract: base_name, legal_suffix, division
        # Handle "Google LLC", "Alphabet Inc."
        
    def normalize_organization(self, org: str) -> str:
        # Remove legal suffixes for matching
        # Expand known acronyms: IBM → International Business Machines
        
    def check_ownership_relationship(self, org1, org2) -> float:
        # YouTube → Google (subsidiary relationship)
```

#### 4. **Departments**

**Unique Challenges:**
- Hierarchical structure (Engineering → Platform Engineering)
- Common abbreviations (HR, IT, QA)
- Alternative naming (People Ops vs Human Resources)

**Custom Components Needed:**
```python
class DepartmentEntityHandler:
    def __init__(self):
        self.dept_hierarchy = self.load_dept_structure()
        self.abbreviations = self.load_dept_abbreviations()
        
    def extract_hierarchy_path(self, dept: str) -> List[str]:
        # "Platform Engineering" → ["Engineering", "Platform Engineering"]
        
    def expand_abbreviations(self, dept: str) -> List[str]:
        # "HR" → ["HR", "Human Resources", "People Operations"]
```

#### 5. **Process Statuses (HR/Finance)**

**Unique Challenges:**
- Temporal relationships (pending → approved → completed)
- Status codes vs descriptions
- Workflow dependencies
- Industry-specific terminology

**Custom Components Needed:**
```python
class StatusEntityHandler:
    def __init__(self):
        self.status_workflows = self.load_workflows()
        self.code_mappings = self.load_status_codes()
        
    def get_workflow_distance(self, status1, status2) -> int:
        # Distance between "pending" and "approved" in workflow
        
    def normalize_status(self, status: str) -> str:
        # Map codes: "STAT_001" → "Active"
        # Standardize: "In Progress" → "in_progress"
```

### Proposed Generalized Architecture

```python
from abc import ABC, abstractmethod

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
    def calculate_custom_similarity(self, query: str, entity: str) -> float:
        """Entity-specific similarity calculation"""
        pass
    
    @abstractmethod
    def get_query_type(self, query: str) -> str:
        """Detect query type for this entity type"""
        pass

class GeneralizedEntityDisambiguator:
    """Main disambiguator that works with any entity type"""
    
    def __init__(self, model_name: str, entity_type: str):
        self.model = self.load_model(model_name)
        self.handler = self.create_handler(entity_type)
        
    def create_handler(self, entity_type: str) -> EntityHandler:
        handlers = {
            "person": PersonNameHandler(),
            "location": LocationHandler(),
            "role": WorkRoleHandler(),
            "organization": OrganizationHandler(),
            "department": DepartmentHandler(),
            "status": StatusHandler()
        }
        return handlers[entity_type]
```

### Configuration-Driven Approach

```yaml
# entity_configs/location.yaml
entity_type: location
exact_match_variations:
  - abbreviations:
      NYC: "New York City"
      LA: "Los Angeles"
      SF: "San Francisco"
  - suffixes_to_ignore: ["City", "Town", "Village"]
  
hierarchy:
  levels: ["neighborhood", "city", "state", "country"]
  
scoring:
  exact_match: 1.0
  hierarchy_match: 0.9
  proximity_bonus: 0.1
  
thresholds:
  partial: 0.6
  full: 0.5
  semantic: 0.4
```

## Part 2: Re-ranking Analysis

### Current Ranking Approach

Currently, we use simple score-based sorting:
```python
matches.sort(key=lambda x: x["similarity"], reverse=True)
```

**Limitations:**
1. No cross-attention between query and entity
2. No learning from user feedback
3. No consideration of entity popularity/recency
4. No query-entity interaction features

### Benefits of Adding Re-ranking

#### 1. **Better Semantic Understanding**
```python
# Current: "Software Engineer at Google" 
# Scores similarly for queries "Engineer" and "Google Engineer"

# With re-ranker: Can understand query intent better
# "Google Engineer" → prioritizes Google-related results
```

#### 2. **Learning from Interactions**
- Click-through rate data
- Dwell time on selected results
- User corrections/feedback

#### 3. **Additional Features**
```python
class ReRankerFeatures:
    - query_entity_overlap
    - entity_popularity
    - entity_recency
    - query_length_ratio
    - exact_match_positions
    - semantic_similarity_distribution
```

### Re-ranking Architecture

```python
class TwoStageSearch:
    def __init__(self, retriever, reranker):
        self.retriever = retriever  # Our current system
        self.reranker = reranker    # mxbai-rerank-v2 or similar
        
    def search(self, query, entities, top_k=5):
        # Stage 1: Fast retrieval (current system)
        candidates = self.retriever.search(
            query, entities, top_k=20  # Get more candidates
        )
        
        # Stage 2: Re-ranking
        if len(candidates) > top_k:
            reranked = self.reranker.rerank(
                query=query,
                documents=[c["descriptor"] for c in candidates],
                top_k=top_k
            )
            return reranked
        return candidates
```

### Latency Analysis

#### Current System:
- Model loading: ~3 seconds (one-time)
- Per search: 0.01-0.2ms (avg ~0.15ms)
- Throughput: ~6,700 searches/second

#### With Re-ranking:
```
Stage 1 (Retrieval): 0.15ms (get top-20)
Stage 2 (Re-rank):   10-15ms (re-rank to top-5)
Total:               10-15ms per search
Throughput:          65-100 searches/second
```

**Latency Breakdown:**
1. **mxbai-rerank-v2 inference**: ~8-12ms for 20 documents
2. **Feature extraction**: ~1-2ms
3. **Sorting and formatting**: ~0.5ms

### Optimization Strategies

#### 1. **Selective Re-ranking**
```python
def should_rerank(results, query_type):
    # Only re-rank when beneficial
    if query_type == "exact" and results[0]["similarity"] > 0.95:
        return False  # Already have clear winner
    
    if len(results) <= 3:
        return False  # Too few results
    
    # Check score distribution
    score_variance = np.var([r["similarity"] for r in results[:5]])
    if score_variance < 0.01:
        return True  # Scores too similar, need re-ranking
    
    return False
```

#### 2. **Caching Re-ranking Results**
```python
class CachedReRanker:
    def __init__(self, reranker, cache_size=10000):
        self.reranker = reranker
        self.cache = LRUCache(cache_size)
        
    def rerank(self, query, documents):
        cache_key = hash((query, tuple(documents)))
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        results = self.reranker.rerank(query, documents)
        self.cache[cache_key] = results
        return results
```

#### 3. **Async Re-ranking for Non-Critical Paths**
```python
async def search_with_progressive_results(query, entities):
    # Return initial results immediately
    initial_results = retriever.search(query, entities, top_k=5)
    yield initial_results
    
    # Re-rank in background if needed
    if should_rerank(initial_results):
        reranked = await reranker.rerank_async(query, initial_results)
        yield reranked
```

### Implementation Roadmap

#### Phase 1: Infrastructure (Week 1-2)
1. Create generalized entity handler framework
2. Implement handlers for 2-3 entity types
3. Add configuration system
4. Update evaluation framework

#### Phase 2: Re-ranking Integration (Week 3-4)
1. Integrate mxbai-rerank-v2 or similar model
2. Implement two-stage search
3. Add selective re-ranking logic
4. Build caching layer

#### Phase 3: Optimization (Week 5-6)
1. Profile and optimize latency
2. Implement progressive loading
3. Add A/B testing framework
4. Collect user interaction data

### Recommended Next Steps

1. **Start with High-Value Entity Types**
   - Work roles (high business impact)
   - Organizations (frequently searched)
   
2. **Re-ranking Pilot**
   - Test on queries with ambiguous results
   - Measure latency impact
   - A/B test with/without re-ranking

3. **Build Feedback Loop**
   - Log searches and selections
   - Train custom re-ranker on your data
   - Iterate based on metrics

## Conclusion

### For Generalization:
- **70% of code is reusable** (base framework, evaluation, embeddings)
- **30% needs customization** (entity handlers, scoring rules)
- Configuration-driven approach recommended for maintainability

### For Re-ranking:
- **10-15ms latency addition** is acceptable for many use cases
- **Selective re-ranking** can minimize impact
- **Significant quality improvements** expected for ambiguous queries
- Start with pilot on subset of queries

The key is to build a flexible framework that can accommodate both entity-specific logic and optional re-ranking while maintaining the speed advantages of the current approach.