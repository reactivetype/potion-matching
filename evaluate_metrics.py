import numpy as np
from entity_disambiguation import EntityDisambiguator
from entity_disambiguation_hybrid import HybridEntityDisambiguator
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from collections import defaultdict


def evaluate_disambiguator(disambiguator, entities, entity_embeddings, test_cases, approach_name):
    """Evaluate a disambiguator with precision, recall, and F1 scores"""
    
    # Collect predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    
    # Per-query metrics
    query_metrics = []
    
    for test in test_cases:
        query = test["query"]
        expected_ids = test["expected_ids"]
        threshold = test.get("threshold", 0.5)
        
        # Get predictions
        results, search_time, match_type = disambiguator.search(
            query, entities, entity_embeddings, threshold=threshold
        )
        
        predicted_ids = [r["id"] for r in results]
        
        # Calculate per-query metrics
        # True Positives: predicted IDs that are in expected IDs
        tp = len(set(predicted_ids) & set(expected_ids))
        # False Positives: predicted IDs that are not in expected IDs
        fp = len(set(predicted_ids) - set(expected_ids))
        # False Negatives: expected IDs that were not predicted
        fn = len(set(expected_ids) - set(predicted_ids))
        
        # Per-query precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        query_metrics.append({
            "query": query,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "predicted": len(predicted_ids),
            "expected": len(expected_ids)
        })
        
        # For overall metrics, create binary labels for each entity
        for entity in entities:
            entity_id = entity["id"]
            # Ground truth: 1 if entity should be returned, 0 otherwise
            all_ground_truth.append(1 if entity_id in expected_ids else 0)
            # Prediction: 1 if entity was returned, 0 otherwise
            all_predictions.append(1 if entity_id in predicted_ids else 0)
    
    # Calculate overall metrics
    overall_precision = precision_score(all_ground_truth, all_predictions, zero_division=0)
    overall_recall = recall_score(all_ground_truth, all_predictions, zero_division=0)
    overall_f1 = f1_score(all_ground_truth, all_predictions, zero_division=0)
    
    # Calculate macro-averaged metrics
    macro_precision = np.mean([m["precision"] for m in query_metrics])
    macro_recall = np.mean([m["recall"] for m in query_metrics])
    macro_f1 = np.mean([m["f1"] for m in query_metrics])
    
    return {
        "approach": approach_name,
        "overall": {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1
        },
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1
        },
        "per_query": query_metrics
    }


def main():
    print("Loading models for evaluation...")
    original = EntityDisambiguator()
    hybrid = HybridEntityDisambiguator()
    
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
    
    # Define test cases with ground truth
    test_cases = [
        # Exact match cases
        {
            "query": "John Smith",
            "expected_ids": ["1", "2"],  # Both John Smiths
            "description": "Exact name - should return both John Smiths",
            "threshold": 0.5
        },
        {
            "query": "Michael Johnson",
            "expected_ids": ["5"],  # Only the Olympic athlete
            "description": "Exact full name match",
            "threshold": 0.5
        },
        {
            "query": "Jane Smith",
            "expected_ids": ["4"],
            "description": "Exact match - Jane Smith",
            "threshold": 0.5
        },
        
        # Typo cases
        {
            "query": "Jhon Smith",
            "expected_ids": ["1", "2"],  # Should still match both John Smiths
            "description": "Typo - should handle and return John Smiths",
            "threshold": 0.5
        },
        {
            "query": "Micheal Johnson",
            "expected_ids": ["5"],  # Should match Michael Johnson
            "description": "Common misspelling",
            "threshold": 0.5
        },
        
        # Case variations
        {
            "query": "john smith",
            "expected_ids": ["1", "2"],
            "description": "Lowercase - should be case insensitive",
            "threshold": 0.5
        },
        {
            "query": "MICHAEL JOHNSON",
            "expected_ids": ["5"],
            "description": "Uppercase - should be case insensitive",
            "threshold": 0.5
        },
        
        # Partial matches
        {
            "query": "John",
            "expected_ids": ["1", "2", "3", "8"],  # All Johns
            "description": "First name only - should match all Johns",
            "threshold": 0.4
        },
        {
            "query": "Johnson",
            "expected_ids": ["5", "7", "9"],  # All Johnsons
            "description": "Last name only - should match all Johnsons",
            "threshold": 0.4
        },
        {
            "query": "Smith",
            "expected_ids": ["1", "2", "4"],  # All Smiths
            "description": "Last name only - should match all Smiths",
            "threshold": 0.4
        },
        
        # Semantic matches
        {
            "query": "Software Engineer Google",
            "expected_ids": ["1"],  # John Smith at Google
            "description": "Semantic search for role and company",
            "threshold": 0.5
        },
        {
            "query": "Olympic Athlete",
            "expected_ids": ["5"],  # Michael Johnson
            "description": "Semantic search for role",
            "threshold": 0.5
        },
        {
            "query": "CEO startup",
            "expected_ids": ["7"],  # Sarah Johnson
            "description": "Semantic search for role",
            "threshold": 0.5
        },
    ]
    
    # Evaluate both approaches
    original_results = evaluate_disambiguator(
        original, entities, original_embeddings, test_cases, "Original POTION"
    )
    
    hybrid_results = evaluate_disambiguator(
        hybrid, entities, hybrid_embeddings, test_cases, "Hybrid Approach"
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION METRICS COMPARISON")
    print("="*80)
    
    # Overall metrics
    print("\nOVERALL METRICS (Binary classification across all entity-query pairs):")
    print(f"{'Approach':<20} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}")
    print("-"*65)
    
    for results in [original_results, hybrid_results]:
        print(f"{results['approach']:<20} "
              f"{results['overall']['precision']:<15.3f} "
              f"{results['overall']['recall']:<15.3f} "
              f"{results['overall']['f1']:<15.3f}")
    
    # Macro-averaged metrics
    print("\nMACRO-AVERAGED METRICS (Average across queries):")
    print(f"{'Approach':<20} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}")
    print("-"*65)
    
    for results in [original_results, hybrid_results]:
        print(f"{results['approach']:<20} "
              f"{results['macro']['precision']:<15.3f} "
              f"{results['macro']['recall']:<15.3f} "
              f"{results['macro']['f1']:<15.3f}")
    
    # Per-query comparison for interesting cases
    print("\n" + "="*80)
    print("PER-QUERY METRICS COMPARISON")
    print("="*80)
    
    print(f"\n{'Query':<25} {'Metric':<10} {'Original':<15} {'Hybrid':<15} {'Improvement':<15}")
    print("-"*80)
    
    for i, test in enumerate(test_cases[:8]):  # Show first 8 cases
        orig_metrics = original_results['per_query'][i]
        hybrid_metrics = hybrid_results['per_query'][i]
        
        # F1 Score comparison
        f1_improvement = ((hybrid_metrics['f1'] - orig_metrics['f1']) / orig_metrics['f1'] * 100) if orig_metrics['f1'] > 0 else float('inf')
        
        print(f"{test['query']:<25} {'F1':<10} {orig_metrics['f1']:<15.3f} {hybrid_metrics['f1']:<15.3f} "
              f"{'+' + str(round(f1_improvement, 1)) + '%' if f1_improvement != float('inf') else '+∞':<15}")
        
        # Show precision and recall details for cases with differences
        if abs(orig_metrics['f1'] - hybrid_metrics['f1']) > 0.1:
            print(f"{'':<25} {'Precision':<10} {orig_metrics['precision']:<15.3f} {hybrid_metrics['precision']:<15.3f}")
            print(f"{'':<25} {'Recall':<10} {orig_metrics['recall']:<15.3f} {hybrid_metrics['recall']:<15.3f}")
            orig_tp_fp_fn = f"{orig_metrics['tp']}/{orig_metrics['fp']}/{orig_metrics['fn']}"
            hybrid_tp_fp_fn = f"{hybrid_metrics['tp']}/{hybrid_metrics['fp']}/{hybrid_metrics['fn']}"
            print(f"{'':<25} {'TP/FP/FN':<10} {orig_tp_fp_fn:<15} {hybrid_tp_fp_fn:<15}")
            print()
    
    # Summary improvements
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)
    
    overall_f1_improvement = ((hybrid_results['overall']['f1'] - original_results['overall']['f1']) / 
                             original_results['overall']['f1'] * 100)
    macro_f1_improvement = ((hybrid_results['macro']['f1'] - original_results['macro']['f1']) / 
                           original_results['macro']['f1'] * 100)
    
    print(f"\nOverall F1 Score Improvement: +{overall_f1_improvement:.1f}%")
    print(f"Macro F1 Score Improvement: +{macro_f1_improvement:.1f}%")
    
    # Count perfect scores
    perfect_original = sum(1 for m in original_results['per_query'] if m['f1'] == 1.0)
    perfect_hybrid = sum(1 for m in hybrid_results['per_query'] if m['f1'] == 1.0)
    
    print(f"\nPerfect F1 scores (1.0):")
    print(f"  Original: {perfect_original}/{len(test_cases)} queries")
    print(f"  Hybrid: {perfect_hybrid}/{len(test_cases)} queries")


if __name__ == "__main__":
    main()