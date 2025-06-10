#!/usr/bin/env python3
import json
import glob
import argparse
from typing import List, Dict

def is_story_mismatch(actual: str, predicted: str) -> bool:
    """Return True if story_class prediction is the opposite of actual."""
    return (actual == "Story" and predicted != "Story") or \
           (actual != "Story" and predicted == "Story")

def has_large_perception_diff(entry: Dict, threshold: int) -> bool:
    """Return True if any perception differs by >= threshold."""
    for dim in ("suspense", "curiosity", "surprise"):
        actual = entry.get(dim, 0)
        pred   = entry.get(f"predicted_{dim}", 0)
        if abs(actual - pred) >= threshold:
            return True
    return False

def find_bad_ids(files: List[str], threshold: int) -> List[int]:
    bad_ids = []
    for fn in files:
        with open(fn, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for entry in data:
            if is_story_mismatch(entry["story_class"], entry["predicted_story_class"]) \
               and has_large_perception_diff(entry, threshold):
                bad_ids.append(entry["id"])
    return bad_ids

def main():
    parser = argparse.ArgumentParser(
        description="List IDs where story_class is mispredicted AND perceptions differ by a lot"
    )
    parser.add_argument(
        "--pattern", "-p",
        default="few_shot_temp0.6_cotYes_structuredYes_fold*_predictions.json",
        help="Glob pattern to match prediction JSON files"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=2,
        help="Min absolute difference in any perception to flag (default: 2)"
    )
    args = parser.parse_args()

    files = glob.glob(args.pattern)
    if not files:
        print(f"No files matched pattern {args.pattern!r}")
        return

    bad_ids = find_bad_ids(files, args.threshold)
    if bad_ids:
        print("Posts with mispredicted story_class AND large perception diff (IDs):")
        for pid in sorted(set(bad_ids)):
            print(pid)
    else:
        print("No posts found that meet both criteria.")

if __name__ == "__main__":
    main()
