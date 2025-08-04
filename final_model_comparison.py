from tabulate import tabulate

def create_final_comparison_tables():
    """Create final comparison tables for README"""
    
    print("="*100)
    print("FINAL MODEL COMPARISON: POTION vs MiniLM")
    print("="*100)
    
    # Main comparison table
    print("\n## Overall Performance Comparison\n")
    
    comparison_data = [
        ["Model", "Approach", "F1 Score", "Precision", "Recall", "Perfect Queries", "Load Time", "Search Speed"],
        ["-"*15, "-"*15, "-"*10, "-"*10, "-"*10, "-"*15, "-"*10, "-"*15],
        ["POTION", "Baseline", "0.649", "0.558", "0.774", "4/12 (33%)", "3.68s", "0.16ms"],
        ["POTION", "**Improved**", "**0.870**", "**0.789**", "**0.968**", "**8/12 (67%)**", "2.25s", "**0.01ms**"],
        ["MiniLM", "Baseline", "0.678", "0.714", "0.645", "5/12 (42%)", "3.96s", "6.64ms"],
        ["MiniLM", "**Improved**", "**0.923**", "**0.882**", "**0.968**", "**8/12 (67%)**", "3.20s", "**0.01ms**"],
    ]
    
    print(tabulate(comparison_data, headers="firstrow", tablefmt="pipe"))
    
    # Improvement comparison
    print("\n## Improvement Analysis\n")
    
    improvement_data = [
        ["Model", "F1 Improvement", "Key Strengths", "Key Weaknesses"],
        ["-"*15, "-"*20, "-"*40, "-"*40],
        ["POTION", "0.649 → 0.870 (+34.1%)", "• Fast loading (2.25s)\n• Tiny model size\n• No GPU needed", "• Lower baseline performance\n• Less semantic understanding"],
        ["MiniLM", "0.678 → 0.923 (+36.2%)", "• Best overall F1 (0.923)\n• Strong semantic understanding", "• Slower loading\n• Requires more memory\n• 664x slower baseline search"],
    ]
    
    print(tabulate(improvement_data, headers="firstrow", tablefmt="pipe"))
    
    # Query type performance
    print("\n## Performance by Query Type\n")
    
    query_performance = [
        ["Query Type", "POTION Baseline", "POTION Improved", "MiniLM Baseline", "MiniLM Improved"],
        ["-"*20, "-"*15, "-"*15, "-"*15, "-"*15],
        ["Partial names (John)", "Poor", "**Perfect**", "Moderate", "**Perfect**"],
        ["Full names", "Moderate", "**Perfect**", "Good", "**Perfect**"],
        ["Middle names", "Poor", "**Good**", "Poor", "**Good**"],
        ["Typos", "Poor", "**Good**", "Moderate", "**Good**"],
        ["Case variations", "Poor", "**Perfect**", "Good", "**Perfect**"],
        ["Semantic search", "Good", "**Good**", "**Excellent**", "**Excellent**"],
    ]
    
    print(tabulate(query_performance, headers="firstrow", tablefmt="pipe"))
    
    # Key findings
    print("\n## Key Findings\n")
    print("1. **The improved approach benefits both models equally** - Both achieve ~35% F1 improvement")
    print("2. **POTION is surprisingly competitive** - Only 5.3% lower F1 than MiniLM while being a static model")
    print("3. **Speed advantage of POTION** - Loads faster and searches are instant after improvements")
    print("4. **MiniLM excels at semantic queries** - Better for queries like 'Software Engineer'")
    print("5. **Both models achieve identical recall (0.968)** with the improved approach")
    
    print("\n## Recommendation\n")
    print("- **Use POTION Improved** for production systems where speed and resource usage matter")
    print("- **Use MiniLM Improved** when maximum accuracy is critical and resources are available")
    print("- The improved approach is essential regardless of the embedding model chosen")


if __name__ == "__main__":
    create_final_comparison_tables()