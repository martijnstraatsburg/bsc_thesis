import json
import os
import re
import pandas as pd
from typing import Dict, List

def parse_config_name(filename: str) -> Dict:
    """Extract parameters from filename pattern: shotType_tempX_cotY_structuredY"""
    pattern = r"^(zero|one|few|multi)_shot_temp([\d.]+)_cot(Yes|No)_structured(Yes|No)_cv_results\.json$"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
        
    return {
        "shot_type": match.group(1),
        "temperature": float(match.group(2)),
        "chain_of_thought": match.group(3) == "Yes",
        "structured_output": match.group(4) == "Yes"
    }

def load_configs(results_dir: str) -> List[Dict]:
    """Load all config results with parsed parameters"""
    configs = []
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(results_dir, filename)) as f:
                    config_data = json.load(f)
                    params = parse_config_name(filename)
                    configs.append({
                        **params,
                        "filename": filename,
                        "average_metrics": config_data["average_metrics"],
                        "fold_metrics": config_data["fold_metrics"]
                    })
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    return configs

def calculate_composite_score(config: Dict, weights: Dict) -> float:
    """Calculate weighted score based on metrics importance"""
    scores = []
    
    # Story classification
    story = config["average_metrics"]["story_classification"]
    scores.append(weights["story_f1"] * story["f1"]["mean"])
    
    # Rating metrics
    ratings = config["average_metrics"]["rating_metrics"]
    for dimension in ["suspense", "curiosity", "surprise"]:
        dim_metrics = ratings[dimension]
        scores.append(weights[f"{dimension}_pearson"] * dim_metrics["pearson_correlation"]["mean"])
        scores.append(weights[f"{dimension}_tolerance1"] * dim_metrics["tolerance1_accuracy"]["mean"])
    
    return sum(scores)

def compare_configs(configs: List[Dict], metric_weights: Dict) -> pd.DataFrame:
    """Create comparison dataframe with key metrics"""
    rows = []
    for config in configs:
        row = {
            "filename": config["filename"],
            "shot_type": config["shot_type"],
            "temperature": config["temperature"],
            "chain_of_thought": config["chain_of_thought"],
            "structured_output": config["structured_output"],
            "composite_score": calculate_composite_score(config, metric_weights)
        }
        
        # Add key metrics
        story = config["average_metrics"]["story_classification"]
        row.update({f"story_{k}": v["mean"] for k, v in story.items()})
        
        for dimension in ["suspense", "curiosity", "surprise"]:
            dim_metrics = config["average_metrics"]["rating_metrics"][dimension]
            row.update({
                f"{dimension}_pearson": dim_metrics["pearson_correlation"]["mean"],
                f"{dimension}_tolerance1": dim_metrics["tolerance1_accuracy"]["mean"],
                f"{dimension}_rmse": dim_metrics["rmse"]["mean"]
            })
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df.sort_values("composite_score", ascending=False)

# Example weights - modify based on your priorities
METRIC_WEIGHTS = {
    "story_f1": 0.1,
    "suspense_pearson": 0.2,
    "curiosity_pearson": 0.15,
    "surprise_pearson": 0.1,
    "suspense_tolerance1": 0.9,
    "curiosity_tolerance1": 0.9,
    "surprise_tolerance1": 0.9
}

if __name__ == "__main__":
    # Load and compare configs
    configs = load_configs("cv_results")
    comparison_df = compare_configs(configs, METRIC_WEIGHTS)
    
    # Display top configurations
    pd.set_option("display.max_columns", None)
    print("\nTop Configurations:")
    print(comparison_df.head(10))
    
    # Save full results
    comparison_df.to_csv("config_comparison.csv", index=False)
    print("\nFull results saved to config_comparison.csv")

    # Optional: Generate visualizations
    # (Uncomment to plot key metrics)
    # import matplotlib.pyplot as plt
    # comparison_df.plot.bar(x="filename", y="composite_score")
    # plt.title("Composite Scores by Configuration")
    # plt.xticks(rotation=45, ha="right")
    # plt.tight_layout()
    # plt.show()