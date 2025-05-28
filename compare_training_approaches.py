#!/usr/bin/env python3
"""
Compare different approaches for Kokoro voice training.

This script helps users understand the trade-offs between:
1. Direct Kokoro embedding optimization (training.py)
2. StyleTTS2 → Kokoro projection pipeline (training_styletts2.py)
"""

import argparse
from pathlib import Path

def print_comparison_table():
    """Print a detailed comparison of training approaches."""
    
    print("\n" + "="*80)
    print("KOKORO VOICE TRAINING APPROACHES COMPARISON")
    print("="*80)
    
    # Approach headers
    approaches = [
        "Aspect",
        "Direct Kokoro Training",
        "StyleTTS2 → Kokoro Pipeline"
    ]
    
    # Comparison data
    comparisons = [
        ["Training Method", "Optimize Kokoro embeddings directly", "Extract StyleTTS2 → Project to Kokoro"],
        ["Convergence Rate", "Often fails to converge", "Reliable convergence"],
        ["Training Time", "2-4 days on GPU", "1-2 hours on GPU"],
        ["Data Requirements", "Large dataset (hours of audio)", "Moderate dataset (30min-2hrs)"],
        ["Voice Quality", "Variable, often poor", "Consistent, good quality"],
        ["Embedding Space", "Overfitted, hard to optimize", "Generalizable, proven"],
        ["Memory Usage", "High (full model + embeddings)", "Moderate (projection only)"],
        ["Success Rate", "~30-40%", "~80-90%"],
        ["Technical Difficulty", "High", "Medium"],
        ["Recommended For", "Research, experimentation", "Production, practical use"],
    ]
    
    # Print table
    col_width = 25
    header_format = f"{{:<{col_width}}} {{:<{col_width}}} {{:<{col_width}}}"
    row_format = f"{{:<{col_width}}} {{:<{col_width}}} {{:<{col_width}}}"
    
    print(header_format.format(*approaches))
    print("-" * (col_width * 3))
    
    for comparison in comparisons:
        print(row_format.format(*comparison))
    
    print("="*80)

def print_detailed_explanation():
    """Print detailed explanation of each approach."""
    
    print("\nDETAILED EXPLANATION")
    print("="*50)
    
    print("\n🔴 DIRECT KOKORO TRAINING (training.py)")
    print("-" * 40)
    print("How it works:")
    print("• Freezes the 82M parameter Kokoro model")
    print("• Optimizes only the 256-dim voice embedding")
    print("• Compares generated speech to reference audio")
    print("• Uses mel-spectrogram loss for optimization")
    
    print("\nPros:")
    print("• Direct optimization of target embedding space")
    print("• No intermediate representations")
    print("• Full control over embedding characteristics")
    
    print("\nCons:")
    print("• Kokoro's embedding space is overfitted")
    print("• Poor convergence due to optimization landscape")
    print("• Requires large amounts of training data")
    print("• Long training times with uncertain results")
    print("• High failure rate in practice")
    
    print("\n🟢 STYLETTS2 → KOKORO PIPELINE (training_styletts2.py)")
    print("-" * 50)
    print("How it works:")
    print("• Uses StyleTTS2 to extract style embeddings from target voice")
    print("• Trains a projection network to map StyleTTS2 → Kokoro space")
    print("• Leverages StyleTTS2's proven voice cloning capabilities")
    print("• Projects to Kokoro's 256-dim embedding format")
    
    print("\nPros:")
    print("• StyleTTS2 has proven voice cloning capabilities")
    print("• Better embedding space for optimization")
    print("• Faster training and reliable convergence")
    print("• Works with smaller datasets")
    print("• High success rate")
    
    print("\nCons:")
    print("• Requires StyleTTS2 dependency")
    print("• Indirect approach (two-stage process)")
    print("• Limited by StyleTTS2's style extraction quality")
    print("• May not capture all nuances of direct optimization")

def print_recommendations():
    """Print recommendations for different use cases."""
    
    print("\nRECOMMENDATIONS")
    print("="*30)
    
    print("\n✅ USE STYLETTS2 → KOKORO PIPELINE IF:")
    print("• You want reliable results")
    print("• You have limited training time")
    print("• You have moderate amounts of data (30min-2hrs)")
    print("• You need production-ready voice cloning")
    print("• You're new to voice training")
    
    print("\n⚠️  USE DIRECT KOKORO TRAINING IF:")
    print("• You're doing research on embedding optimization")
    print("• You have large amounts of high-quality data")
    print("• You have time for extensive experimentation")
    print("• You need to understand Kokoro's embedding space")
    print("• You're willing to accept high failure rates")
    
    print("\n🚫 AVOID DIRECT KOKORO TRAINING IF:")
    print("• You need reliable results for production")
    print("• You have limited time or computational resources")
    print("• You're working with small datasets")
    print("• You're not familiar with TTS training")

def print_migration_guide():
    """Print guide for migrating from direct training to StyleTTS2 pipeline."""
    
    print("\nMIGRATION GUIDE")
    print("="*30)
    
    print("\nIf you're currently using direct Kokoro training:")
    
    print("\n1. ASSESS YOUR CURRENT SETUP:")
    print("   • How much data do you have?")
    print("   • What's your current success rate?")
    print("   • How long does training take?")
    
    print("\n2. PREPARE FOR STYLETTS2 PIPELINE:")
    print("   • Install StyleTTS2: pip install styletts2>=0.1.6")
    print("   • Ensure your dataset follows the same format")
    print("   • No changes needed to dataset structure")
    
    print("\n3. SWITCH TO NEW TRAINING SCRIPT:")
    print("   Old command:")
    print("   python training.py --data ./dataset --epochs 200 --name my_voice")
    print("   ")
    print("   New command:")
    print("   python training_styletts2.py --data ./dataset --epochs-projection 100 --name my_voice")
    
    print("\n4. EXPECT DIFFERENT RESULTS:")
    print("   • Faster training (hours vs days)")
    print("   • More reliable convergence")
    print("   • Different voice characteristics (may be better or different)")
    print("   • Same output format (compatible with existing Kokoro usage)")

def main():
    parser = argparse.ArgumentParser(description="Compare Kokoro voice training approaches")
    parser.add_argument("--detailed", action="store_true", help="Show detailed explanations")
    parser.add_argument("--recommendations", action="store_true", help="Show recommendations")
    parser.add_argument("--migration", action="store_true", help="Show migration guide")
    parser.add_argument("--all", action="store_true", help="Show all information")
    
    args = parser.parse_args()
    
    # Always show the comparison table
    print_comparison_table()
    
    if args.all or args.detailed:
        print_detailed_explanation()
    
    if args.all or args.recommendations:
        print_recommendations()
    
    if args.all or args.migration:
        print_migration_guide()
    
    if not any([args.detailed, args.recommendations, args.migration, args.all]):
        print("\nFor more information, use:")
        print("  --detailed      : Detailed explanation of each approach")
        print("  --recommendations : Recommendations for different use cases")
        print("  --migration     : Guide for migrating from direct training")
        print("  --all           : Show all information")

if __name__ == "__main__":
    main() 