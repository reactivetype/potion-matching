"""
Concrete examples of how to implement different entity types
Shows the 70/30 split between reusable and custom code
"""

from typing import List, Dict, Tuple
import re
from abc import ABC, abstractmethod


# ============================================================================
# REUSABLE BASE (70% of code)
# ============================================================================

class BaseEntityHandler(ABC):
    """Base functionality shared by ALL entity types"""
    
    def __init__(self):
        self.fuzzy_threshold = 0.85
        self.exact_match_boost = 1.0
        self.fuzzy_match_penalty = 0.05
    
    # These methods are IDENTICAL for all entity types
    def fuzzy_match_score(self, s1: str, s2: str) -> float:
        """Calculate fuzzy string matching score - REUSABLE"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def normalize_text(self, text: str) -> str:
        """Basic text normalization - REUSABLE"""
        return ' '.join(text.lower().split())
    
    def check_exact_match(self, query: str, entity: str) -> bool:
        """Check for exact string match - REUSABLE"""
        return self.normalize_text(query) == self.normalize_text(entity)
    
    def calculate_base_scores(self, query: str, entity: str) -> Dict[str, float]:
        """Calculate standard scores - REUSABLE"""
        return {
            "exact": 1.0 if self.check_exact_match(query, entity) else 0.0,
            "fuzzy": self.fuzzy_match_score(query, entity),
            "contains": 1.0 if query.lower() in entity.lower() else 0.0
        }


# ============================================================================
# CUSTOM IMPLEMENTATIONS (30% of code per entity type)
# ============================================================================

class StatusEntityHandler(BaseEntityHandler):
    """Handler for HR/Finance process statuses"""
    
    def __init__(self):
        super().__init__()
        # Status-specific configuration
        self.status_groups = {
            "active": ["active", "in progress", "ongoing", "current"],
            "completed": ["completed", "done", "finished", "closed"],
            "pending": ["pending", "waiting", "on hold", "queued"],
            "failed": ["failed", "error", "rejected", "denied"]
        }
        
        self.status_codes = {
            "STAT_001": "active",
            "STAT_002": "completed",
            "STAT_003": "pending",
            "STAT_004": "failed"
        }
    
    def extract_parts(self, entity_descriptor: str) -> Dict:
        """Extract status-specific parts"""
        # Handle formats like "STAT_001 - Active Process"
        parts = entity_descriptor.split(' - ')
        code = None
        description = entity_descriptor
        
        if len(parts) > 1 and parts[0] in self.status_codes:
            code = parts[0]
            description = parts[1]
        
        # Identify status group
        status_group = None
        desc_lower = description.lower()
        for group, keywords in self.status_groups.items():
            if any(keyword in desc_lower for keyword in keywords):
                status_group = group
                break
        
        return {
            "full": entity_descriptor,
            "code": code,
            "description": description,
            "group": status_group,
            "normalized": self.normalize_text(description)
        }
    
    def get_exact_match_variations(self, query: str) -> List[str]:
        """Generate status-specific variations"""
        variations = [query, query.lower(), query.upper()]
        
        # Add code mappings
        if query.upper() in self.status_codes:
            variations.append(self.status_codes[query.upper()])
        
        # Add group variations
        query_lower = query.lower()
        for group, keywords in self.status_groups.items():
            if query_lower in keywords:
                variations.extend(keywords)
        
        return list(set(variations))
    
    def calculate_custom_score(self, query: str, entity: str, base_score: float) -> float:
        """Apply status-specific scoring"""
        query_parts = self.extract_parts(query)
        entity_parts = self.extract_parts(entity)
        
        # Boost for same status group
        if query_parts["group"] and query_parts["group"] == entity_parts["group"]:
            base_score = min(base_score * 1.2, 1.0)
        
        # Perfect match for status codes
        if query.upper() in self.status_codes:
            if entity_parts["code"] == query.upper():
                return 1.0
        
        return base_score


class DepartmentEntityHandler(BaseEntityHandler):
    """Handler for organizational departments"""
    
    def __init__(self):
        super().__init__()
        self.abbreviations = {
            "hr": "human resources",
            "it": "information technology",
            "qa": "quality assurance",
            "r&d": "research and development",
            "bd": "business development",
            "pr": "public relations"
        }
        
        self.hierarchy = {
            "engineering": ["frontend", "backend", "devops", "qa", "mobile"],
            "sales": ["inside sales", "enterprise", "smb", "partnerships"],
            "marketing": ["digital", "content", "brand", "growth"],
        }
    
    def extract_parts(self, entity_descriptor: str) -> Dict:
        """Extract department hierarchy"""
        parts = re.split(r' - | › ', entity_descriptor)
        dept_name = parts[0].strip()
        
        # Find parent department
        parent = None
        dept_lower = dept_name.lower()
        for parent_dept, sub_depts in self.hierarchy.items():
            if parent_dept in dept_lower:
                parent = parent_dept
            for sub in sub_depts:
                if sub in dept_lower:
                    parent = parent_dept
                    break
        
        return {
            "full": dept_name,
            "parent": parent,
            "normalized": self.expand_abbreviations(dept_name),
            "level": "sub" if parent else "top"
        }
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand department abbreviations"""
        result = text.lower()
        for abbr, full in self.abbreviations.items():
            result = re.sub(f"\\b{abbr}\\b", full, result)
        return result
    
    def get_exact_match_variations(self, query: str) -> List[str]:
        """Generate department variations"""
        variations = [query, query.lower()]
        
        # Add expanded form
        expanded = self.expand_abbreviations(query)
        if expanded != query.lower():
            variations.append(expanded)
        
        # Add abbreviated form
        query_lower = query.lower()
        for abbr, full in self.abbreviations.items():
            if full in query_lower:
                variations.append(query_lower.replace(full, abbr))
        
        return list(set(variations))


class ProductEntityHandler(BaseEntityHandler):
    """Handler for product names and SKUs"""
    
    def __init__(self):
        super().__init__()
        self.sku_pattern = re.compile(r'^[A-Z0-9]{3,}-[A-Z0-9]{3,}')
        self.version_pattern = re.compile(r'v?\d+\.?\d*')
    
    def extract_parts(self, entity_descriptor: str) -> Dict:
        """Extract product components"""
        # Handle "SKU-123 - Product Name v2.0"
        parts = entity_descriptor.split(' - ')
        
        sku = None
        name = entity_descriptor
        version = None
        
        if len(parts) > 1 and self.sku_pattern.match(parts[0]):
            sku = parts[0]
            name = parts[1]
        
        # Extract version
        version_match = self.version_pattern.search(name)
        if version_match:
            version = version_match.group()
            name_without_version = name.replace(version, '').strip()
        else:
            name_without_version = name
        
        return {
            "full": entity_descriptor,
            "sku": sku,
            "name": name,
            "name_without_version": name_without_version,
            "version": version,
            "normalized": self.normalize_text(name_without_version)
        }
    
    def calculate_custom_score(self, query: str, entity: str, base_score: float) -> float:
        """Product-specific scoring"""
        query_parts = self.extract_parts(query)
        entity_parts = self.extract_parts(entity)
        
        # Perfect match for SKU
        if query_parts["sku"] and query_parts["sku"] == entity_parts["sku"]:
            return 1.0
        
        # Boost for matching product name without version
        if query_parts["normalized"] == entity_parts["normalized"]:
            return 0.95
        
        return base_score


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def demonstrate_entity_handlers():
    """Show how different entity types work with the same base code"""
    
    print("ENTITY TYPE IMPLEMENTATION EXAMPLES")
    print("=" * 80)
    
    # Example 1: Status Handler
    print("\n1. STATUS ENTITIES (HR/Finance Processes)")
    print("-" * 60)
    
    status_handler = StatusEntityHandler()
    status_entities = [
        "STAT_001 - Active Employee Onboarding",
        "STAT_002 - Completed Background Check",
        "Pending Approval - Manager Review",
        "In Progress - Payroll Processing"
    ]
    
    print("Query: 'active'")
    for entity in status_entities:
        parts = status_handler.extract_parts(entity)
        variations = status_handler.get_exact_match_variations("active")
        print(f"  {entity}")
        print(f"    → Group: {parts['group']}, Variations: {len(variations)}")
    
    # Example 2: Department Handler
    print("\n\n2. DEPARTMENT ENTITIES")
    print("-" * 60)
    
    dept_handler = DepartmentEntityHandler()
    dept_entities = [
        "HR - Benefits Administration",
        "Engineering › Frontend Development",
        "IT Support - Help Desk",
        "Research and Development - AI Lab"
    ]
    
    print("Query: 'HR'")
    for entity in dept_entities:
        parts = dept_handler.extract_parts(entity)
        expanded = dept_handler.expand_abbreviations("HR")
        print(f"  {entity}")
        print(f"    → Expanded: '{expanded}', Parent: {parts['parent']}")
    
    # Example 3: Product Handler
    print("\n\n3. PRODUCT ENTITIES")
    print("-" * 60)
    
    product_handler = ProductEntityHandler()
    product_entities = [
        "PRD-001 - Analytics Dashboard v2.3",
        "PRD-002 - Mobile App v1.0",
        "Enterprise CRM Suite",
        "API-GW-01 - API Gateway Service"
    ]
    
    print("Query: 'Analytics Dashboard' (without version)")
    for entity in product_entities:
        parts = product_handler.extract_parts(entity)
        print(f"  {entity}")
        print(f"    → SKU: {parts['sku']}, Version: {parts['version']}, ")
        print(f"      Name without version: '{parts['name_without_version']}'")
    
    # Show code reuse
    print("\n\n4. CODE REUSE ANALYSIS")
    print("=" * 80)
    
    base_methods = [
        "fuzzy_match_score",
        "normalize_text", 
        "check_exact_match",
        "calculate_base_scores",
        "encode",
        "cosine_similarity",
        "search",
        "create_embeddings"
    ]
    
    custom_methods = [
        "extract_parts",
        "get_exact_match_variations",
        "calculate_custom_score",
        "entity-specific __init__"
    ]
    
    print(f"REUSABLE methods (70%): {len(base_methods)}")
    for method in base_methods:
        print(f"  ✓ {method}")
    
    print(f"\nCUSTOM methods per entity type (30%): {len(custom_methods)}")
    for method in custom_methods:
        print(f"  • {method}")
    
    print("\n\nCONCLUSION:")
    print("-" * 60)
    print("Adding a new entity type requires implementing only 3-4 methods!")
    print("The heavy lifting (embeddings, search, scoring) is all reusable.")


if __name__ == "__main__":
    demonstrate_entity_handlers()