from entity_disambiguation_improved import ImprovedEntityDisambiguator


def test_middle_names():
    print("TESTING MIDDLE NAME HANDLING")
    print("="*80)
    
    # Initialize
    disambiguator = ImprovedEntityDisambiguator()
    
    # Test entities with middle names
    entities = [
        {"id": "1", "descriptor": "John Michael Smith - Software Engineer"},
        {"id": "2", "descriptor": "John Smith - Professor"},
        {"id": "3", "descriptor": "Michael John Davis - Data Scientist"},
        {"id": "4", "descriptor": "Sarah Jane Smith - Product Manager"},
        {"id": "5", "descriptor": "Jane Smith - Designer"},
        {"id": "6", "descriptor": "Robert Michael Johnson - CEO"},
        {"id": "7", "descriptor": "Michael - Intern"},  # Just first name
        {"id": "8", "descriptor": "John Paul Jones - Musician"},
        {"id": "9", "descriptor": "Mary Jane Watson - Journalist"},
    ]
    
    # Create embeddings
    entity_embeddings = disambiguator.create_entity_embeddings(entities)
    
    # Let's check what name parts are extracted
    print("\nEXTRACTED NAME PARTS:")
    print("-"*80)
    for i, entity in enumerate(entities):
        name_parts = entity_embeddings["name_parts"][i]
        print(f"Entity: {entity['descriptor']}")
        print(f"  Full name: '{name_parts['full']}'")
        print(f"  First: '{name_parts['first']}'")
        print(f"  Last: '{name_parts['last']}'")
        print(f"  All parts: {name_parts['parts']}")
        print()
    
    # Test queries
    test_queries = [
        ("Michael", "Should match all people with Michael as first OR middle name"),
        ("Jane", "Should match all people with Jane as first OR middle name"),
        ("John", "Should match all Johns"),
        ("Paul", "Should match John Paul Jones if middle names work"),
        ("John Michael Smith", "Exact match with middle name"),
        ("John Smith", "Should match both John Smiths"),
        ("Michael Smith", "Should NOT match John Michael Smith (Michael is middle)"),
        ("Sarah Jane", "First + middle name"),
        ("Jane Watson", "Middle + last name"),
    ]
    
    print("\n" + "="*80)
    print("QUERY RESULTS")
    print("="*80)
    
    for query, expected in test_queries:
        print(f"\nQuery: '{query}' - {expected}")
        results, _, match_type = disambiguator.search(query, entities, entity_embeddings, threshold=0.4)
        print(f"Found {len(results)} matches:")
        
        for i, result in enumerate(results[:5]):
            match_info = f"  {i+1}. {result['descriptor']} (score: {result['similarity']:.3f}"
            if 'match_type' in result:
                match_info += f", type: {result['match_type']}"
            match_info += ")"
            print(match_info)
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    print("\nCurrent implementation status for middle names:")
    print("1. Name extraction: Splits full name into parts")
    print("2. First name matching: ✓ Works (uses first element)")
    print("3. Last name matching: ✓ Works (uses last element)")
    print("4. Middle name matching: Let's see from the results...")


if __name__ == "__main__":
    test_middle_names()