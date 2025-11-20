#!/usr/bin/env python3
"""
Check quality of precomputed SAM2 instances.

Reports purity distribution, class balance, and recommendations.
"""
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np


def analyze_manifest(manifest_path: Path):
    """Analyze instance quality from manifest file."""
    print(f"\nAnalyzing: {manifest_path}")
    print("=" * 80)
    
    instances = []
    with manifest_path.open('r') as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    
    if not instances:
        print("❌ No instances found!")
        return
    
    # Extract metrics
    purities = np.array([inst['purity'] for inst in instances])
    areas = np.array([inst['area'] for inst in instances])
    labels = [inst['label'] for inst in instances]
    
    # Purity analysis
    print("\n📊 Purity Distribution:")
    print(f"  Mean:   {purities.mean():.3f}")
    print(f"  Median: {np.median(purities):.3f}")
    print(f"  Std:    {purities.std():.3f}")
    print(f"\n  >0.95 (excellent): {(purities > 0.95).sum():6d} ({(purities > 0.95).mean()*100:5.1f}%)")
    print(f"  >0.90 (very good): {(purities > 0.90).sum():6d} ({(purities > 0.90).mean()*100:5.1f}%)")
    print(f"  >0.80 (good):      {(purities > 0.80).sum():6d} ({(purities > 0.80).mean()*100:5.1f}%)")
    print(f"  >0.70 (okay):      {(purities > 0.70).sum():6d} ({(purities > 0.70).mean()*100:5.1f}%)")
    print(f"  <0.70 (noisy):     {(purities < 0.70).sum():6d} ({(purities < 0.70).mean()*100:5.1f}%)")
    
    # Area analysis
    print("\n📏 Area Distribution:")
    print(f"  Mean:   {areas.mean():.0f} pixels")
    print(f"  Median: {np.median(areas):.0f} pixels")
    print(f"  Min:    {areas.min():.0f} pixels")
    print(f"  Max:    {areas.max():.0f} pixels")
    
    # Class distribution
    label_counts = Counter(labels)
    print(f"\n🏷️  Class Distribution ({len(label_counts)} classes):")
    
    # Sort by frequency
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n  Top 10 most frequent classes:")
    for label, count in sorted_labels[:10]:
        print(f"    Class {label:3d}: {count:6d} ({count/len(labels)*100:5.1f}%)")
    
    print(f"\n  Bottom 10 least frequent classes:")
    for label, count in sorted_labels[-10:]:
        print(f"    Class {label:3d}: {count:6d} ({count/len(labels)*100:5.1f}%)")
    
    # Imbalance ratio
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"\n  Imbalance ratio: {imbalance_ratio:.1f}x (max/min)")
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("-" * 80)
    
    if purities.mean() > 0.85:
        print("✅ Excellent label quality! Use --min-purity 0.7 or 0.8")
    elif purities.mean() > 0.75:
        print("✅ Good label quality. Use --min-purity 0.7")
    elif purities.mean() > 0.65:
        print("⚠️  Moderate label quality. Start with --min-purity 0.6")
    else:
        print("❌ Low label quality. Check SAM2 masks and ground truth alignment.")
    
    if imbalance_ratio > 100:
        print("⚠️  Severe class imbalance. Consider class-weighted loss or resampling.")
    elif imbalance_ratio > 50:
        print("⚠️  High class imbalance. May need class weighting.")
    else:
        print("✅ Reasonable class balance.")
    
    if areas.mean() < 200:
        print("⚠️  Small instances. Consider using larger --image-size (e.g., 384x384)")
    
    print("\n📝 Suggested training command:")
    print("-" * 80)
    
    # Determine min_purity
    if purities.mean() > 0.8:
        min_purity = 0.7
    elif purities.mean() > 0.7:
        min_purity = 0.6
    else:
        min_purity = 0.5
    
    # Determine min_area
    min_area = int(np.percentile(areas, 5))  # 5th percentile
    
    manifest_name = manifest_path.name
    output_root = manifest_path.parent.parent
    
    print(f"""
python train_instance_classifier.py \\
  --train-manifest {manifest_path} \\
  --val-manifest {output_root}/val/{manifest_name} \\
  --output-root {output_root} \\
  --num-classes {len(label_counts)} \\
  --encoder resnet50 \\
  --batch-size 32 \\
  --lr 1e-4 \\
  --epochs 50 \\
  --min-purity {min_purity} \\
  --min-area {min_area} \\
  --output-dir outputs/instance_classifier
""")


def main():
    parser = argparse.ArgumentParser(description='Check instance quality')
    parser.add_argument('manifest', type=str, help='Path to instances_manifest.jsonl')
    args = parser.parse_args()
    
    manifest_path = Path(args.manifest)
    
    if not manifest_path.exists():
        print(f"❌ File not found: {manifest_path}")
        return
    
    analyze_manifest(manifest_path)


if __name__ == '__main__':
    main()

