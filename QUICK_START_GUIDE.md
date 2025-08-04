# Entity Disambiguation - Quick Start Guide

## Installation

```bash
# Install dependencies
uv pip install -r pyproject.toml

# Required packages:
# - model2vec>=0.6.0 (for POTION static embeddings)
# - sentence-transformers>=2.2.0 (for MiniLM dense embeddings)
# - scikit-learn>=1.3.0
# - numpy>=1.24.0
```

## Quick Examples

### 1. Basic Usage - Person Name Disambiguation

```python
from entity_disambiguation_improved import ImprovedEntityDisambiguator

# Initialize
disambiguator = ImprovedEntityDisambiguator()

# Define entities
entities = [
    {"id": "1", "descriptor": "John Smith - Software Engineer at Google"},
    {"id": "2", "descriptor": "John Michael Smith - Professor at MIT"},
    {"id": "3", "descriptor": "Jane Smith - Product Manager at Apple"},
]

# Create embeddings (one-time operation)
embeddings = disambiguator.create_entity_embeddings(entities)

# Search examples
results, time, match_type = disambiguator.search("John Smith", entities, embeddings)
# Returns both John Smiths (including John Michael Smith)

results, time, match_type = disambiguator.search("John", entities, embeddings)
# Returns all people named John

results, time, match_type = disambiguator.search("Software Engineer", entities, embeddings)
# Returns John Smith at Google (semantic search)
```

### 2. Using Different Models

```python
from entity_disambiguation_improved_flexible import ImprovedFlexibleEntityDisambiguator

# Use POTION (fast, lightweight)
potion_disambiguator = ImprovedFlexibleEntityDisambiguator(
    model_name="minishlab/potion-multilingual-128M",
    model_type="static"
)

# Use MiniLM (better semantic understanding)
minilm_disambiguator = ImprovedFlexibleEntityDisambiguator(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_type="sentence-transformer"
)
```

### 3. Handling Different Query Types

```python
# Partial name (first or last name only)
disambiguator.search("Smith", entities, embeddings, threshold=0.5)

# Name with typo
disambiguator.search("Jhon Smith", entities, embeddings)  # Matches John Smith

# Case insensitive
disambiguator.search("JOHN SMITH", entities, embeddings)

# Middle name handling
disambiguator.search("John M. Smith", entities, embeddings)  # Matches John Michael Smith

# Initial search
disambiguator.search("J", entities, embeddings)  # Returns all J names
```

## Running Evaluations

```bash
# Test basic functionality
uv run python entity_disambiguation_improved.py

# Run full evaluation
uv run python evaluate_with_middle_names.py

# Compare models
uv run python flexible_model_comparison.py

# Analyze search speed
uv run python analyze_search_speed.py
```

## Key Features

### 1. Query Type Detection
- **Partial names**: "John" → searches first/last names
- **Full names**: "John Smith" → exact and fuzzy matching
- **Semantic**: "Software Engineer Google" → embedding similarity

### 2. Scoring Hierarchy
```
1.000 - Exact match
0.950 - Name without middle (John Smith → John Michael Smith)
0.900 - Fuzzy match (typos)
0.800 - Semantic similarity
```

### 3. Performance
- **Search speed**: 0.01-0.2ms per query
- **Model loading**: ~3 seconds
- **F1 Score**: 0.835 (41.7% improvement over baseline)

## Extending to Other Entity Types

```python
from generalized_entity_framework import GeneralizedEntityDisambiguator

# For locations
location_disambiguator = GeneralizedEntityDisambiguator(
    model_name="minishlab/potion-multilingual-128M",
    entity_type="location"
)

# For work roles
role_disambiguator = GeneralizedEntityDisambiguator(
    model_name="minishlab/potion-multilingual-128M",
    entity_type="role"
)
```

## Common Use Cases

### HR/Recruiting
```python
candidates = [
    {"id": "1", "descriptor": "John Smith - Senior Software Engineer - 10 years experience"},
    {"id": "2", "descriptor": "John Smith - Junior Developer - Recent graduate"},
]

# Search by name
results = disambiguator.search("John Smith", candidates, embeddings)

# Search by role
results = disambiguator.search("Senior Engineer", candidates, embeddings)
```

### Customer Support
```python
customers = [
    {"id": "1", "descriptor": "John Smith - john.smith@email.com - Account #12345"},
    {"id": "2", "descriptor": "Jane Smith - jane.s@email.com - Account #67890"},
]

# Handle typos in customer names
results = disambiguator.search("Jon Smith", customers, embeddings)
```

### Academic/Research
```python
authors = [
    {"id": "1", "descriptor": "John Smith - MIT - Machine Learning"},
    {"id": "2", "descriptor": "J. Smith - Stanford - Computer Vision"},
]

# Search by partial name or initial
results = disambiguator.search("J Smith", authors, embeddings)
```

## Best Practices

1. **Threshold Selection**
   - Use 0.5 for general searches
   - Use 0.4 for partial names (higher recall)
   - Use 0.6 for exact matches only (higher precision)

2. **Entity Descriptor Format**
   - Include full name first
   - Add context after dash: "Name - Role/Company/Description"
   - Be consistent across entities

3. **Performance Optimization**
   - Create embeddings once and reuse
   - Use POTION for speed-critical applications
   - Use MiniLM for better semantic understanding

## Troubleshooting

### Issue: Too many false positives
```python
# Increase threshold
results = disambiguator.search(query, entities, embeddings, threshold=0.7)
```

### Issue: Missing expected matches
```python
# Lower threshold or check query type
results = disambiguator.search(query, entities, embeddings, threshold=0.3)
```

### Issue: Slow performance
```python
# Use POTION instead of MiniLM
# Pre-compute and cache embeddings
# Consider batch processing for multiple queries
```

## Next Steps

1. See `PROJECT_SUMMARY.md` for detailed documentation
2. Check `implementation_roadmap.md` for extending to new entity types
3. Review `reranking_integration.py` for quality improvements
4. Explore `entity_type_examples.py` for custom entity handlers