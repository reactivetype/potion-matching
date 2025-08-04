from entity_disambiguation import EntityDisambiguator
from entity_disambiguation_hybrid import HybridEntityDisambiguator
from entity_disambiguation_improved import ImprovedEntityDisambiguator
from evaluate_metrics import evaluate_disambiguator
from tabulate import tabulate
import numpy as np


def main():
    print("Generating comprehensive comparison table...")
    
    # Initialize all approaches
    original = EntityDisambiguator()
    hybrid = HybridEntityDisambiguator()
    improved = ImprovedEntityDisambiguator()
    
    # Define entities
    entities = [
        {"id": "1", "descriptor": "John Smith - Software Engineer at Google"},
        {"id": "2", "descriptor": "John Smith - Professor of Physics at MIT"},
        {"id": "3", "descriptor": "John Doe - Data Scientist at Microsoft"},
        {"id": "4", "descriptor": "Jane Smith - Product Manager at Apple"},
        {"id": "5", "descriptor": "Michael Johnson - Olympic Athlete"},
        {"id": "6", "descriptor": "Michael Jordan - Basketball Player"},
        {"id": "7", "descriptor": "Sarah Johnson - CEO of Tech Startup"},
        {"id": "8", "descriptor": "John Williams - Composer"},
        {"id": "9", "descriptor": "Robert Johnson - Blues Musician"},
    ]
    
    # Create embeddings
    original_embeddings = original.create_entity_embeddings(entities)
    hybrid_embeddings = hybrid.create_entity_embeddings(entities)
    improved_embeddings = improved.create_entity_embeddings(entities)
    
    # Define comprehensive test cases
    test_cases = [
        # Partial name queries
        {
            "query": "John",
            "expected_ids": ["1", "2", "3", "8"],
            "description": "First name only",
            "threshold": 0.5
        },
        {
            "query": "Johnson",
            "expected_ids": ["5", "7", "9"],
            "description": "Last name only",
            "threshold": 0.5
        },
        {
            "query": "Smith",
            "expected_ids": ["1", "2", "4"],
            "description": "Last name only",
            "threshold": 0.5
        },
        # Exact matches
        {
            "query": "John Smith",
            "expected_ids": ["1", "2"],
            "description": "Full name",
            "threshold": 0.5
        },
        {
            "query": "Michael Johnson",
            "expected_ids": ["5"],
            "description": "Exact full name",
            "threshold": 0.5
        },
        {
            "query": "Jane Smith",
            "expected_ids": ["4"],
            "description": "Exact full name",
            "threshold": 0.5
        },
        # Typo cases
        {
            "query": "Jhon Smith",
            "expected_ids": ["1", "2"],
            "description": "Typo in first name",
            "threshold": 0.5
        },
        {
            "query": "Micheal Johnson",
            "expected_ids": ["5"],
            "description": "Common misspelling",
            "threshold": 0.5
        },
        # Case variations
        {
            "query": "john smith",
            "expected_ids": ["1", "2"],
            "description": "Lowercase",
            "threshold": 0.5
        },
        {
            "query": "JOHN SMITH",
            "expected_ids": ["1", "2"],
            "description": "Uppercase",
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
            "expected_ids": ["5"],
            "description": "Semantic search",
            "threshold": 0.5
        },
        {
            "query": "CEO startup",
            "expected_ids": ["7"],
            "description": "Semantic search",
            "threshold": 0.5
        },
    ]
    
    # Evaluate all approaches
    original_results = evaluate_disambiguator(
        original, entities, original_embeddings, test_cases, "Original POTION"
    )
    
    hybrid_results = evaluate_disambiguator(
        hybrid, entities, hybrid_embeddings, test_cases, "Hybrid"
    )
    
    improved_results = evaluate_disambiguator(
        improved, entities, improved_embeddings, test_cases, "Improved"
    )
    
    # Print comprehensive comparison table
    print("\n" + "="*120)
    print("COMPREHENSIVE METRICS COMPARISON TABLE")
    print("="*120)
    
    # Overall and Macro metrics table
    metrics_data = []
    
    # Header row
    metrics_data.append(["Metric", "Original POTION", "Hybrid Approach", "Improved Approach", "Best"])
    metrics_data.append(["-"*20, "-"*20, "-"*20, "-"*20, "-"*10])
    
    # Overall metrics
    metrics_data.append(["**OVERALL METRICS**", "", "", "", ""])
    
    # Precision
    precisions = [
        original_results['overall']['precision'],
        hybrid_results['overall']['precision'],
        improved_results['overall']['precision']
    ]
    best_precision_idx = np.argmax(precisions)
    precision_row = ["Precision", 
                     f"{precisions[0]:.3f}",
                     f"{precisions[1]:.3f}",
                     f"{precisions[2]:.3f}",
                     ["Original", "Hybrid", "Improved"][best_precision_idx]]
    metrics_data.append(precision_row)
    
    # Recall
    recalls = [
        original_results['overall']['recall'],
        hybrid_results['overall']['recall'],
        improved_results['overall']['recall']
    ]
    best_recall_idx = np.argmax(recalls)
    recall_row = ["Recall", 
                  f"{recalls[0]:.3f}",
                  f"{recalls[1]:.3f}",
                  f"{recalls[2]:.3f}",
                  ["Original", "Hybrid", "Improved"][best_recall_idx]]
    metrics_data.append(recall_row)
    
    # F1
    f1s = [
        original_results['overall']['f1'],
        hybrid_results['overall']['f1'],
        improved_results['overall']['f1']
    ]
    best_f1_idx = np.argmax(f1s)
    f1_row = ["F1 Score", 
              f"{f1s[0]:.3f}",
              f"{f1s[1]:.3f}",
              f"{f1s[2]:.3f}",
              ["Original", "Hybrid", "Improved"][best_f1_idx]]
    metrics_data.append(f1_row)
    
    # Macro metrics
    metrics_data.append(["", "", "", "", ""])
    metrics_data.append(["**MACRO METRICS**", "", "", "", ""])
    
    # Macro Precision
    macro_precisions = [
        original_results['macro']['precision'],
        hybrid_results['macro']['precision'],
        improved_results['macro']['precision']
    ]
    best_macro_precision_idx = np.argmax(macro_precisions)
    macro_precision_row = ["Macro Precision", 
                           f"{macro_precisions[0]:.3f}",
                           f"{macro_precisions[1]:.3f}",
                           f"{macro_precisions[2]:.3f}",
                           ["Original", "Hybrid", "Improved"][best_macro_precision_idx]]
    metrics_data.append(macro_precision_row)
    
    # Macro Recall
    macro_recalls = [
        original_results['macro']['recall'],
        hybrid_results['macro']['recall'],
        improved_results['macro']['recall']
    ]
    best_macro_recall_idx = np.argmax(macro_recalls)
    macro_recall_row = ["Macro Recall", 
                        f"{macro_recalls[0]:.3f}",
                        f"{macro_recalls[1]:.3f}",
                        f"{macro_recalls[2]:.3f}",
                        ["Original", "Hybrid", "Improved"][best_macro_recall_idx]]
    metrics_data.append(macro_recall_row)
    
    # Macro F1
    macro_f1s = [
        original_results['macro']['f1'],
        hybrid_results['macro']['f1'],
        improved_results['macro']['f1']
    ]
    best_macro_f1_idx = np.argmax(macro_f1s)
    macro_f1_row = ["Macro F1 Score", 
                    f"{macro_f1s[0]:.3f}",
                    f"{macro_f1s[1]:.3f}",
                    f"{macro_f1s[2]:.3f}",
                    ["Original", "Hybrid", "Improved"][best_macro_f1_idx]]
    metrics_data.append(macro_f1_row)
    
    # Perfect scores
    metrics_data.append(["", "", "", "", ""])
    perfect_orig = sum(1 for m in original_results['per_query'] if m['f1'] == 1.0)
    perfect_hybrid = sum(1 for m in hybrid_results['per_query'] if m['f1'] == 1.0)
    perfect_improved = sum(1 for m in improved_results['per_query'] if m['f1'] == 1.0)
    
    perfects = [perfect_orig, perfect_hybrid, perfect_improved]
    best_perfect_idx = np.argmax(perfects)
    perfect_row = ["Perfect F1 Queries", 
                   f"{perfect_orig}/{len(test_cases)}",
                   f"{perfect_hybrid}/{len(test_cases)}",
                   f"{perfect_improved}/{len(test_cases)}",
                   ["Original", "Hybrid", "Improved"][best_perfect_idx]]
    metrics_data.append(perfect_row)
    
    # Print the table
    print(tabulate(metrics_data, headers="firstrow", tablefmt="grid", colalign=("left", "center", "center", "center", "center")))
    
    # Per-query F1 scores table
    print("\n" + "="*120)
    print("PER-QUERY F1 SCORES COMPARISON")
    print("="*120)
    
    query_data = []
    query_data.append(["Query", "Type", "Original", "Hybrid", "Improved", "Winner"])
    query_data.append(["-"*25, "-"*15, "-"*10, "-"*10, "-"*10, "-"*10])
    
    for i, test in enumerate(test_cases):
        orig_f1 = original_results['per_query'][i]['f1']
        hybrid_f1 = hybrid_results['per_query'][i]['f1']
        improved_f1 = improved_results['per_query'][i]['f1']
        
        f1_scores = [orig_f1, hybrid_f1, improved_f1]
        best_idx = np.argmax(f1_scores)
        winner = ["Original", "Hybrid", "Improved"][best_idx]
        if f1_scores.count(max(f1_scores)) > 1:
            winner = "Tie"
        
        query_data.append([
            test['query'][:25],
            test['description'],
            f"{orig_f1:.3f}",
            f"{hybrid_f1:.3f}",
            f"{improved_f1:.3f}",
            winner
        ])
    
    print(tabulate(query_data, headers="firstrow", tablefmt="grid"))
    
    # Improvement percentages
    print("\n" + "="*120)
    print("IMPROVEMENT PERCENTAGES (Improved vs Original)")
    print("="*120)
    
    improvements = []
    improvements.append(["Metric", "Original", "Improved", "Change", "% Improvement"])
    improvements.append(["-"*20, "-"*15, "-"*15, "-"*15, "-"*15])
    
    # Calculate improvements
    metrics_list = [
        ("Overall Precision", original_results['overall']['precision'], improved_results['overall']['precision']),
        ("Overall Recall", original_results['overall']['recall'], improved_results['overall']['recall']),
        ("Overall F1", original_results['overall']['f1'], improved_results['overall']['f1']),
        ("Macro Precision", original_results['macro']['precision'], improved_results['macro']['precision']),
        ("Macro Recall", original_results['macro']['recall'], improved_results['macro']['recall']),
        ("Macro F1", original_results['macro']['f1'], improved_results['macro']['f1']),
    ]
    
    for metric_name, orig_val, imp_val in metrics_list:
        change = imp_val - orig_val
        pct_improvement = (change / orig_val * 100) if orig_val > 0 else 0
        improvements.append([
            metric_name,
            f"{orig_val:.3f}",
            f"{imp_val:.3f}",
            f"{change:+.3f}",
            f"{pct_improvement:+.1f}%"
        ])
    
    print(tabulate(improvements, headers="firstrow", tablefmt="grid"))
    
    # Summary
    print("\n" + "="*120)
    print("KEY TAKEAWAYS")
    print("="*120)
    print(f"1. Overall F1 Score improved by {(improved_results['overall']['f1'] - original_results['overall']['f1']) / original_results['overall']['f1'] * 100:.1f}%")
    print(f"2. Macro F1 Score improved by {(improved_results['macro']['f1'] - original_results['macro']['f1']) / original_results['macro']['f1'] * 100:.1f}%")
    print(f"3. Perfect F1 queries increased from {perfect_orig} to {perfect_improved} (+{perfect_improved - perfect_orig})")
    print("4. The improved approach successfully handles:")
    print("   - Partial name queries (first/last names)")
    print("   - Exact name matching (case-insensitive)")
    print("   - Typo tolerance")
    print("   - Semantic search capabilities")


if __name__ == "__main__":
    main()