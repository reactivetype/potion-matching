import numpy as np

def explain_metrics_difference():
    """Explain the difference between overall and macro metrics with examples"""
    
    print("="*80)
    print("UNDERSTANDING OVERALL vs MACRO METRICS")
    print("="*80)
    
    # Simple example with 3 queries
    print("\nExample: 3 queries searching for entities")
    print("-"*80)
    
    queries = [
        {
            "name": "Query 1: 'John'",
            "expected": ["E1", "E2", "E3", "E4"],  # 4 entities should match
            "predicted": ["E1", "E2", "E3", "E4", "E5"],  # 5 entities returned
            "tp": 4, "fp": 1, "fn": 0
        },
        {
            "name": "Query 2: 'Smith'", 
            "expected": ["E6", "E7"],  # 2 entities should match
            "predicted": ["E6"],  # Only 1 returned
            "tp": 1, "fp": 0, "fn": 1
        },
        {
            "name": "Query 3: 'Michael'",
            "expected": ["E8"],  # 1 entity should match
            "predicted": ["E8"],  # Exactly 1 returned
            "tp": 1, "fp": 0, "fn": 0
        }
    ]
    
    # Calculate per-query metrics
    print("\nPER-QUERY METRICS:")
    per_query_recalls = []
    per_query_precisions = []
    
    for q in queries:
        recall = q["tp"] / (q["tp"] + q["fn"]) if (q["tp"] + q["fn"]) > 0 else 0
        precision = q["tp"] / (q["tp"] + q["fp"]) if (q["tp"] + q["fp"]) > 0 else 0
        per_query_recalls.append(recall)
        per_query_precisions.append(precision)
        
        print(f"\n{q['name']}:")
        print(f"  Expected: {q['expected']} (count: {len(q['expected'])})")
        print(f"  Predicted: {q['predicted']} (count: {len(q['predicted'])})")
        print(f"  TP={q['tp']}, FP={q['fp']}, FN={q['fn']}")
        print(f"  Recall: {recall:.2f}, Precision: {precision:.2f}")
    
    # Calculate MACRO metrics
    macro_recall = np.mean(per_query_recalls)
    macro_precision = np.mean(per_query_precisions)
    
    print(f"\n" + "="*80)
    print("MACRO METRICS (Average of per-query metrics):")
    print(f"  Macro Recall = ({' + '.join([f'{r:.2f}' for r in per_query_recalls])}) / 3 = {macro_recall:.3f}")
    print(f"  Macro Precision = ({' + '.join([f'{p:.2f}' for p in per_query_precisions])}) / 3 = {macro_precision:.3f}")
    
    # Calculate OVERALL metrics
    total_tp = sum(q["tp"] for q in queries)
    total_fp = sum(q["fp"] for q in queries)
    total_fn = sum(q["fn"] for q in queries)
    
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    
    print(f"\n" + "="*80)
    print("OVERALL METRICS (Calculated from all predictions together):")
    print(f"  Total TP = {total_tp}, Total FP = {total_fp}, Total FN = {total_fn}")
    print(f"  Overall Recall = {total_tp} / ({total_tp} + {total_fn}) = {overall_recall:.3f}")
    print(f"  Overall Precision = {total_tp} / ({total_tp} + {total_fp}) = {overall_precision:.3f}")
    
    print(f"\n" + "="*80)
    print("KEY DIFFERENCES:")
    print("="*80)
    
    print("""
1. MACRO METRICS:
   - Calculate metric for EACH query separately
   - Then take the AVERAGE across all queries
   - Treats each query equally regardless of how many entities it should return
   - Good for: Understanding average performance per query

2. OVERALL METRICS:
   - Pool ALL predictions and ground truth labels together
   - Calculate metric on the entire pool at once
   - Queries that return more entities have more weight
   - Good for: Understanding total system performance

3. WHY THEY DIFFER:
   - Macro gives equal weight to each query (1/3 each in our example)
   - Overall gives more weight to queries with more expected results
   - Query 1 expects 4 entities, Query 3 expects 1 entity
   - In overall metrics, Query 1's performance affects the score 4x more than Query 3

4. IN OUR ENTITY DISAMBIGUATION CONTEXT:
   - Query "John" expects 4 matches → contributes 4 data points to overall
   - Query "Michael Johnson" expects 1 match → contributes 1 data point to overall
   - But in macro metrics, both queries count equally (1/13 of the total)
""")
    
    print("\nWHICH TO USE?")
    print("-"*80)
    print("- MACRO: When you want to know 'On average, how well does each query perform?'")
    print("- OVERALL: When you want to know 'Of all the entities that should be returned, what % do we get?'")
    print("- For entity disambiguation: MACRO is often more relevant as users care about per-query accuracy")


if __name__ == "__main__":
    explain_metrics_difference()