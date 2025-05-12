#!/usr/bin/env python3

# Name: create_golden_standard.py
# Author: Martijn Straatsburg
# Description: This script generates a golden-standard JSON file from multiple annotator data.
# It calculates the majority vote for story classification and the median for suspense, curiosity, and surprise ratings.
# It also handles the input and output file paths through command-line arguments.
# Usage: python create_golden_standard.py <input_json> <output_json>
# Example: python create_golden_standard.py anno.json golden-standard.json

import json
import argparse
from collections import Counter
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate golden-standard annotations from multiple annotator data")
    parser.add_argument(
        "input", help="Path to the input JSON file with annotations")
    parser.add_argument(
        "output", help="Path to write the golden-standard JSON output")
    return parser.parse_args()


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        entry["suspense_rating"] = [r["rating"]
                                    for r in entry.get("suspense", [])]
        entry["curiosity_rating"] = [r["rating"]
                                     for r in entry.get("curiosity", [])]
        entry["surprise_rating"] = [r["rating"]
                                    for r in entry.get("surprise", [])]
    return pd.DataFrame(data)


def majority_vote(series):
    counts = Counter(series)
    return counts.most_common(1)[0][0]


def median_round(values):
    if not values:
        return None
    median = pd.Series(values).median()
    return int(round(median))


def generate_gold_standard(df):
    static_cols = ["name", "id", "author", "created_utc",
                   "body", "parent_id", "persuasion_success"]

    def agg_func(group):
        return {
            "story_class": majority_vote(group["story_class"]),
            "suspense": median_round(sum(group["suspense_rating"], [])),
            "curiosity": median_round(sum(group["curiosity_rating"], [])),
            "surprise": median_round(sum(group["surprise_rating"], [])),
        }

    grouped = df.groupby(static_cols).apply(lambda g: pd.Series(agg_func(g)))
    result = grouped.reset_index()
    return result


def write_output(gold_df, path):
    records = []
    for row in gold_df.to_dict(orient="records"):
        rec = {
            "name": row["name"],
            "id": row["id"],
            "author": row["author"],
            "created_utc": row["created_utc"],
            "body": row["body"],
            "parent_id": row["parent_id"],
            "persuasion_success": row["persuasion_success"],
            "story_class": row["story_class"],
            "suspense": row["suspense"],
            "curiosity": row["curiosity"],
            "surprise": row["surprise"],
        }
        records.append(rec)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    df = load_data(args.input)
    gold = generate_gold_standard(df)
    write_output(gold, args.output)
    print(f"Golden-standard annotations written to {args.output}")
    # Split done by other group member and results in gs-train.json and gs-test.json


if __name__ == "__main__":
    main()
