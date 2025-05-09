import json
import os
import csv
from collections import defaultdict

def evaluate_config(metrics):
    """Score a config based on the desired metrics."""
    sc = metrics["story_classification"]
    rm = metrics["rating_metrics"]
    
    # story: prefer high macro_f1 and mcc
    story_score = sc["macro_f1"] + sc["mcc"]
    
    # readers: prefer high kappa, low mae
    reader_score = 0
    for dim in ["suspense", "curiosity", "surprise"]:
        reader_score += rm[dim]["quadratic_weighted_kappa"] - rm[dim]["mae"]
    
    return story_score + reader_score

def extract_shot_type(filename):
    """Infer shot type from filename, ignoring any 'fold' variants."""
    fn = filename.lower()
    for shot in ["zero", "one", "few", "multi"]:
        if shot in fn and "fold" not in fn:
            return shot
    return None

def load_average_metrics(data):
    """
    Given either a dict with 'average_metrics' or a list thereof,
    yield each average_metrics block.
    """
    if isinstance(data, dict) and "average_metrics" in data:
        yield data["average_metrics"]
    elif isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict) and "average_metrics" in entry:
                yield entry["average_metrics"]

def flatten_metrics(avg):
    """Pull out just the numbers we care about into a uniform dict."""
    sc = avg["story_classification"]
    rm = avg["rating_metrics"]
    return {
        "story_classification": {
            "macro_f1": sc["macro_f1"]["mean"],
            "mcc":     sc["mcc"]["mean"]
        },
        "rating_metrics": {
            dim: {
                "mae":                         rm[dim]["mae"]["mean"],
                "quadratic_weighted_kappa":   rm[dim]["quadratic_weighted_kappa"]["mean"]
            }
            for dim in ["suspense", "curiosity", "surprise"]
        }
    }

def find_top_configs(folder):
    """
    Scan all .json files in `folder`, compute a single score per file
    (averaged across any list entries), and select the top-2 per shot type.
    """
    scores_by_shot = defaultdict(list)

    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        shot = extract_shot_type(fname)
        if shot is None:
            continue

        path = os.path.join(folder, fname)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # accumulate scores for each average_metrics block
        block_scores = []
        for avg in load_average_metrics(raw):
            try:
                flat = flatten_metrics(avg)
                block_scores.append(evaluate_config(flat))
            except KeyError:
                # skip malformed blocks
                continue

        if not block_scores:
            continue

        # average the block scores (in case file had multiple entries)
        mean_score = sum(block_scores) / len(block_scores)
        scores_by_shot[shot].append((mean_score, fname))

    # pick top-2 per shot type
    best = {}
    for shot, lst in scores_by_shot.items():
        lst.sort(key=lambda x: x[0], reverse=True)
        best[shot] = lst[:2]

    return best

if __name__ == "__main__":
    folder = "8bb-test-cv_results"
    best = find_top_configs(folder)

    for shot, items in best.items():
        print(f"\nTop 2 configs for '{shot}' shot:")
        for score, fname in items:
            print(f"  {fname:40s}  Score: {score:.4f}")

    csv_path = "best_4B_configs_per_shot.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["shot_type", "filename", "score"])
        for shot, items in best.items():
            for score, fname in items:
                writer.writerow([shot, fname, f"{score:.4f}"])
    print(f"\nResults also saved to: {csv_path}")
