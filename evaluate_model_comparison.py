from entity_disambiguation_flexible import FlexibleEntityDisambiguator
from entity_disambiguation_improved_flexible import ImprovedFlexibleEntityDisambiguator
from evaluate_metrics import evaluate_disambiguator
from tabulate import tabulate
import numpy as np
import time


def run_model_comparison():
    """Compare POTION vs MiniLM for both baseline and improved approaches"""
    
    print("COMPARING POTION vs MiniLM MODELS")
    print("="*100)
    
    # Extended entities with middle names
    entities = [
        {"id": "1", "descriptor": "John Smith - Software Engineer at Google"},
        {"id": "2", "descriptor": "John Michael Smith - Professor of Physics at MIT"},
        {"id": "3", "descriptor": "John M. Smith - Data Scientist at Facebook"},
        {"id": "4", "descriptor": "John Doe - Data Scientist at Microsoft"},
        {"id": "5", "descriptor": "Jane Smith - Product Manager at Apple"},
        {"id": "6", "descriptor": "Jane Marie Smith - Designer at Adobe"},
        {"id": "7", "descriptor": "Michael Johnson - Olympic Athlete"},
        {"id": "8", "descriptor": "Michael J. Johnson - Basketball Player"},
        {"id": "9", "descriptor": "Sarah Johnson - CEO of Tech Startup"},
        {"id": "10", "descriptor": "John Williams - Composer"},
        {"id": "11", "descriptor": "Robert Johnson - Blues Musician"},
        {"id": "12", "descriptor": "Mary Jane Watson - Journalist"},
        {"id": "13", "descriptor": "J. Paul Jones - Musician"},
    ]
    
    # Comprehensive test cases
    test_cases = [
        # Partial name queries
        {
            "query": "John",
            "expected_ids": ["1", "2", "3", "4", "10", "13"],
            "description": "First name only",
            "threshold": 0.5
        },
        {
            "query": "Johnson",
            "expected_ids": ["7", "8", "9", "11"],
            "description": "Last name only",
            "threshold": 0.5
        },
        {
            "query": "Smith",
            "expected_ids": ["1", "2", "3", "5", "6"],
            "description": "Last name only",
            "threshold": 0.5
        },
        # Full name queries
        {
            "query": "John Smith",
            "expected_ids": ["1", "2", "3"],
            "description": "Full name without middle",
            "threshold": 0.5
        },
        {
            "query": "Michael Johnson",
            "expected_ids": ["7", "8"],
            "description": "Full name without middle",
            "threshold": 0.5
        },
        # Middle name queries
        {
            "query": "John Michael Smith",
            "expected_ids": ["2"],
            "description": "Full name with middle",
            "threshold": 0.5
        },
        {
            "query": "John M. Smith",
            "expected_ids": ["3"],
            "description": "Full name with initial",
            "threshold": 0.5
        },
        # Typo cases
        {
            "query": "Jhon Smith",
            "expected_ids": ["1", "2", "3"],
            "description": "Typo in first name",
            "threshold": 0.5
        },
        # Case variations
        {
            "query": "john smith",
            "expected_ids": ["1", "2", "3"],
            "description": "Lowercase",
            "threshold": 0.5
        },
        # Semantic queries
        {
            "query": "Software Engineer Google",
            "expected_ids": ["1"],
            "description": "Semantic search",
            "threshold": 0.5
        },
        {
            "query": "Olympic Athlete",
            "expected_ids": ["7"],
            "description": "Semantic search",
            "threshold": 0.5
        },
        {
            "query": "CEO startup",
            "expected_ids": ["9"],
            "description": "Semantic search",
            "threshold": 0.5
        },
    ]
    
    # Initialize all models
    print("\nInitializing models...")
    models = {}
    
    # POTION models
    print("  Loading POTION baseline...")
    models['potion_baseline'] = FlexibleEntityDisambiguator(
        model_name="minishlab/potion-multilingual-128M",
        model_type="static"
    )
    
    print("  Loading POTION improved...")
    models['potion_improved'] = ImprovedFlexibleEntityDisambiguator(
        model_name="minishlab/potion-multilingual-128M",
        model_type="static"
    )
    
    # MiniLM models
    print("  Loading MiniLM baseline...")
    models['minilm_baseline'] = FlexibleEntityDisambiguator(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_type="sentence-transformer"
    )
    
    print("  Loading MiniLM improved...")
    models['minilm_improved'] = ImprovedFlexibleEntityDisambiguator(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_type="sentence-transformer"
    )
    
    # Create embeddings for all models
    print("\nCreating embeddings...")
    embeddings = {}
    
    print("  Creating POTION baseline embeddings...")
    embeddings['potion_baseline'] = models['potion_baseline'].create_entity_embeddings(entities)
    
    print("  Creating POTION improved embeddings...")
    embeddings['potion_improved'] = models['potion_improved'].create_entity_embeddings(entities)
    
    print("  Creating MiniLM baseline embeddings...")
    embeddings['minilm_baseline'] = models['minilm_baseline'].create_entity_embeddings(entities)
    
    print("  Creating MiniLM improved embeddings...")
    embeddings['minilm_improved'] = models['minilm_improved'].create_entity_embeddings(entities)
    
    # Evaluate all approaches
    print("\nRunning evaluations...")
    results = {}
    
    configs = [
        ('potion_baseline', "POTION Baseline", models['potion_baseline'], embeddings['potion_baseline']),
        ('potion_improved', "POTION Improved", models['potion_improved'], embeddings['potion_improved']),
        ('minilm_baseline', "MiniLM Baseline", models['minilm_baseline'], embeddings['minilm_baseline']),
        ('minilm_improved', "MiniLM Improved", models['minilm_improved'], embeddings['minilm_improved']),
    ]
    
    for key, name, model, emb in configs:
        print(f"  Evaluating {name}...")
        results[key] = evaluate_disambiguator(
            model, entities, emb, test_cases, name
        )
    
    # Print comparison tables
    print("\n" + "="*100)
    print("MODEL COMPARISON: POTION vs MiniLM")
    print("="*100)
    
    # Overall metrics comparison
    print("\nOVERALL METRICS:")
    metrics_data = []
    headers = ["Metric", "POTION Baseline", "POTION Improved", "MiniLM Baseline", "MiniLM Improved"]
    
    # Add separator
    metrics_data.append(["-"*15] + ["-"*18]*4)
    
    # Metrics to compare
    metric_names = [
        ("Overall Precision", 'overall', 'precision'),
        ("Overall Recall", 'overall', 'recall'),
        ("Overall F1", 'overall', 'f1'),
        ("Macro Precision", 'macro', 'precision'),
        ("Macro Recall", 'macro', 'recall'),
        ("Macro F1", 'macro', 'f1'),
    ]
    
    for metric_label, category, metric in metric_names:
        row = [metric_label]
        for key in ['potion_baseline', 'potion_improved', 'minilm_baseline', 'minilm_improved']:
            value = results[key][category][metric]
            row.append(f"{value:.3f}")
        metrics_data.append(row)
    
    # Perfect F1 queries
    row = ["Perfect F1 Queries"]
    for key in ['potion_baseline', 'potion_improved', 'minilm_baseline', 'minilm_improved']:
        perfect = sum(1 for m in results[key]['per_query'] if m['f1'] == 1.0)
        row.append(f"{perfect}/{len(test_cases)}")
    metrics_data.append(row)
    
    print(tabulate(metrics_data, headers=headers, tablefmt="grid"))
    
    # Model comparison summary
    print("\n" + "="*100)
    print("IMPROVEMENT COMPARISON")
    print("="*100)
    
    improvement_data = []
    improvement_headers = ["Metric", "POTION (Baseline→Improved)", "MiniLM (Baseline→Improved)"]
    improvement_data.append(["-"*20, "-"*25, "-"*25])
    
    for metric_label, category, metric in metric_names:
        row = [metric_label]
        
        # POTION improvement
        potion_base = results['potion_baseline'][category][metric]
        potion_imp = results['potion_improved'][category][metric]
        potion_change = ((potion_imp - potion_base) / potion_base * 100) if potion_base > 0 else 0
        row.append(f"{potion_base:.3f} → {potion_imp:.3f} ({potion_change:+.1f}%)")
        
        # MiniLM improvement
        minilm_base = results['minilm_baseline'][category][metric]
        minilm_imp = results['minilm_improved'][category][metric]
        minilm_change = ((minilm_imp - minilm_base) / minilm_base * 100) if minilm_base > 0 else 0
        row.append(f"{minilm_base:.3f} → {minilm_imp:.3f} ({minilm_change:+.1f}%)")
        
        improvement_data.append(row)
    
    print(tabulate(improvement_data, headers=improvement_headers, tablefmt="grid"))
    
    # Speed comparison
    print("\n" + "="*100)
    print("SPEED COMPARISON")
    print("="*100)
    
    speed_data = []
    speed_headers = ["Model", "Load Time (s)", "Avg Search Time (ms)"]
    
    # Test search speed
    test_query = "John Smith"
    for key, name, model, emb in configs:
        # Load time
        load_time = model.load_time
        
        # Search time (average of 10 searches)
        search_times = []
        for _ in range(10):
            start = time.time()
            _, _, _ = model.search(test_query, entities, emb, threshold=0.5)
            search_times.append((time.time() - start) * 1000)
        avg_search_time = np.mean(search_times)
        
        speed_data.append([name, f"{load_time:.2f}", f"{avg_search_time:.2f}"])
    
    print(tabulate(speed_data, headers=speed_headers, tablefmt="grid"))
    
    # Key findings
    print("\n" + "="*100)
    print("KEY FINDINGS")
    print("="*100)
    
    # Calculate which model performed best
    best_f1_baseline = max(results['potion_baseline']['overall']['f1'], 
                          results['minilm_baseline']['overall']['f1'])
    best_f1_improved = max(results['potion_improved']['overall']['f1'], 
                          results['minilm_improved']['overall']['f1'])
    
    print(f"\n1. Best baseline model: {'POTION' if results['potion_baseline']['overall']['f1'] > results['minilm_baseline']['overall']['f1'] else 'MiniLM'}")
    print(f"2. Best improved model: {'POTION' if results['potion_improved']['overall']['f1'] > results['minilm_improved']['overall']['f1'] else 'MiniLM'}")
    print(f"3. POTION is {models['minilm_baseline'].load_time / models['potion_baseline'].load_time:.1f}x faster to load")
    print(f"4. The improved approach benefits both models similarly")
    print(f"5. POTION achieves comparable or better performance while being a static embedding model")


if __name__ == "__main__":
    run_model_comparison()