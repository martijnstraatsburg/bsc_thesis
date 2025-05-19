#!/usr/bin/env python3

# Name: prompt-cv-eval.py
# Author: Martijn Straatsburg
# Description: 

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, Any, List, Union
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_squared_error, mean_absolute_error,
    cohen_kappa_score, matthews_corrcoef
)
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

BASEURL = "http://localhost:8000/v1/"
APIKEY = "EMPTY"
#MODEL = "Qwen/Qwen3-4B"
MODEL = "Qwen/Qwen3-8B"
#MODEL = "Qwen/Qwen2.5-3B-Instruct"
#MODEL = "Qwen/Qwen2.5-7B-Instruct"


class CrossValidationEvaluator:
    """
    Class to perform cross-validation evaluation of the Qwen models
    """
    def __init__(self, client, input_file: str, n_splits: int = 5, output_dir: str = "8B-test-cv_results"):
        """
        Initialize the evaluator with the client, input file, number of splits, and output directory
        """
        self.client = client
        self.n_splits = n_splits
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(input_file, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        def convert_to_level(r):
            """..."""
            if r <= 2:
                return "low"
            elif r == 3:
                return "medium"
            else:
                return "high"

        for item in self.dataset:
            item["suspense_level"] = convert_to_level(item["suspense"])
            item["curiosity_level"] = convert_to_level(item["curiosity"])
            item["surprise_level"] = convert_to_level(item["surprise"])
            item["story_binary"] = 1 if item["story_class"] == "Story" else 0

        y = np.array([
            [
                item["story_binary"],
                {"low": 0, "medium": 1, "high": 2}[item["suspense_level"]],
                {"low": 0, "medium": 1, "high": 2}[item["curiosity_level"]],
                {"low": 0, "medium": 1, "high": 2}[item["surprise_level"]],
            ]
            for item in self.dataset
        ])

        self.kf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.split_indices = list(self.kf.split(self.dataset, y))
        self.configurations = [
            # Standard configs for all Qwen3 models
            {"shot": "zero", "temperature": 0.6, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "one", "temperature": 0.6, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "few", "temperature": 0.6, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "multi", "temperature": 0.6, "chain_of_thought": True, "use_structured_output": True}
            # Standard configs for all Qwen2.5 models
            #{"shot": "zero", "temperature": 0.7, "chain_of_thought": True, "use_structured_output": True},
            #{"shot": "one", "temperature": 0.7, "chain_of_thought": True, "use_structured_output": True},
            #{"shot": "few", "temperature": 0.7, "chain_of_thought": True, "use_structured_output": True},
            #{"shot": "multi", "temperature": 0.7, "chain_of_thought": True, "use_structured_output": True}
        ]

    def run_cross_validation(self, model_name: str = MODEL):
        """
        Run cross-validation for all configurations
        """
        all_results = {}
        for config in self.configurations:
            config_name = f"{config['shot']}_shot_temp{config['temperature']}_cot{'Yes' if config['chain_of_thought'] else 'No'}_structured{'Yes' if config['use_structured_output'] else 'No'}"
            print(f"\n--- Cross-validating with {config_name} ---")
            fold_results = []
            for fold_idx, (train_idx, test_idx) in enumerate(self.split_indices):
                print(f"  Fold {fold_idx+1}/{self.n_splits}")
                train_data = [self.dataset[i] for i in train_idx]
                test_data = [self.dataset[i] for i in test_idx]
                analyser = PostAnalyser(
                    client=self.client,
                    model_name=model_name,
                    shot=config["shot"],
                    temperature=config["temperature"],
                    chain_of_thought=config["chain_of_thought"],
                    use_structured_output=config["use_structured_output"]
                )
                test_predictions = copy.deepcopy(test_data)
                for i, entry in enumerate(tqdm(test_predictions, desc=f"Analysing fold {fold_idx+1}")):
                    text = entry["body"]
                    analysis = analyser.analyse_text(text)
                    if "error" not in analysis:
                        entry["predicted_story_class"] = analysis["story_class"]
                        entry["predicted_suspense"] = analysis["suspense"]
                        entry["predicted_curiosity"] = analysis["curiosity"]
                        entry["predicted_surprise"] = analysis["surprise"]
                    else:
                        print(
                            f"Error analysing entry {i+1}: {analysis['error']}")
                        entry["predicted_story_class"] = "Not Story"
                        entry["predicted_suspense"] = 1
                        entry["predicted_curiosity"] = 1
                        entry["predicted_surprise"] = 1
                fold_metric = self.calculate_metrics(
                    test_data, test_predictions)
                fold_results.append(fold_metric)
                fold_output = os.path.join(
                    self.output_dir, f"{config_name}_fold{fold_idx+1}_predictions.json")
                with open(fold_output, "w", encoding="utf-8") as f:
                    json.dump(test_predictions, f,
                              indent=2, ensure_ascii=False)
            avg_metrics = self.calculate_average_metrics(fold_results)
            all_results[config_name] = {
                "average_metrics": avg_metrics,
                "fold_metrics": fold_results
            }
            config_output = os.path.join(
                self.output_dir, f"{config_name}_cv_results.json")
            with open(config_output, "w", encoding="utf-8") as f:
                json.dump(all_results[config_name], f,
                          indent=2, ensure_ascii=False)
            print(f"\nAverage metrics for {config_name}:")
            print(json.dumps(avg_metrics, indent=2))
        self.create_summary_report(all_results)
        return all_results

    def calculate_metrics(self, true_data: List[Dict], pred_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate metrics for story classification and readers perception ratings
        """
        y_true_story = [1 if item["story_class"] == "Story" else 0 for item in true_data]
        y_pred_story = [1 if item["predicted_story_class"] == "Story" else 0 for item in pred_data]
        story_accuracy = accuracy_score(y_true_story, y_pred_story)
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true_story, y_pred_story, average="micro"
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true_story, y_pred_story, average="macro"
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true_story, y_pred_story, average="weighted"
        )
        story_mcc = matthews_corrcoef(y_true_story, y_pred_story)
        
        rating_metrics = {}
        for rating in ["suspense", "curiosity", "surprise"]:
            y_true_rating = [item[rating] for item in true_data]
            y_pred_rating = [item[f"predicted_{rating}"] for item in pred_data]
            
            perfect_accuracy = accuracy_score(y_true_rating, y_pred_rating)
            off_by_one_accuracy = sum(abs(y_true - y_pred) <= 1 for y_true, y_pred in zip(
                y_true_rating, y_pred_rating)) / len(y_true_rating)
            rmse = np.sqrt(mean_squared_error(y_true_rating, y_pred_rating))
            mae = mean_absolute_error(y_true_rating, y_pred_rating)
            weighted_kappa = cohen_kappa_score(y_true_rating, y_pred_rating, weights="quadratic")
            
            def convert_to_level(rating_value):
                if rating_value <= 2:
                    return "low"
                elif rating_value == 3:
                    return "medium"
                else:
                    return "high"
            
            y_true_levels = [convert_to_level(r) for r in y_true_rating]
            y_pred_levels = [convert_to_level(r) for r in y_pred_rating]
            level_accuracy = accuracy_score(y_true_levels, y_pred_levels)
            
            def convert_to_binary(rating_value):
                return 1 if rating_value > 2.5 else 0
                
            y_true_binary = [convert_to_binary(r) for r in y_true_rating]
            y_pred_binary = [convert_to_binary(r) for r in y_pred_rating]
            binary_accuracy = accuracy_score(y_true_binary, y_pred_binary)
            
            try:
                raw_precision, raw_recall, raw_f1, _ = precision_recall_fscore_support(
                    y_true_rating, y_pred_rating, average=None, labels=[1, 2, 3, 4, 5], zero_division=0
                )
                raw_macro_precision, raw_macro_recall, raw_macro_f1, _ = precision_recall_fscore_support(
                    y_true_rating, y_pred_rating, average="macro", zero_division=0
                )
                raw_weighted_precision, raw_weighted_recall, raw_weighted_f1, _ = precision_recall_fscore_support(
                    y_true_rating, y_pred_rating, average="weighted", zero_division=0
                )
            except:
                raw_precision, raw_recall, raw_f1 = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
                raw_macro_precision, raw_macro_recall, raw_macro_f1 = 0, 0, 0
                raw_weighted_precision, raw_weighted_recall, raw_weighted_f1 = 0, 0, 0
            
            try:
                level_macro_precision, level_macro_recall, level_macro_f1, _ = precision_recall_fscore_support(
                    y_true_levels, y_pred_levels, average="macro", zero_division=0
                )
                level_weighted_precision, level_weighted_recall, level_weighted_f1, _ = precision_recall_fscore_support(
                    y_true_levels, y_pred_levels, average="weighted", zero_division=0
                )
            except:
                level_macro_precision, level_macro_recall, level_macro_f1 = 0, 0, 0
                level_weighted_precision, level_weighted_recall, level_weighted_f1 = 0, 0, 0
            
            try:
                binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
                    y_true_binary, y_pred_binary, average=None, zero_division=0
                )
                binary_macro_precision, binary_macro_recall, binary_macro_f1, _ = precision_recall_fscore_support(
                    y_true_binary, y_pred_binary, average="macro", zero_division=0
                )
                binary_weighted_precision, binary_weighted_recall, binary_weighted_f1, _ = precision_recall_fscore_support(
                    y_true_binary, y_pred_binary, average="weighted", zero_division=0
                )
            except:
                binary_precision, binary_recall, binary_f1 = [0, 0], [0, 0], [0, 0]
                binary_macro_precision, binary_macro_recall, binary_macro_f1 = 0, 0, 0
                binary_weighted_precision, binary_weighted_recall, binary_weighted_f1 = 0, 0, 0
                
            rating_metrics[rating] = {
                "perfect_accuracy": perfect_accuracy,
                "off_by_one_accuracy": off_by_one_accuracy,
                "level_accuracy": level_accuracy,
                "rmse": rmse,
                "mae": mae,
                "quadratic_weighted_kappa": weighted_kappa,
                
                "binary_accuracy": binary_accuracy,
                
                "raw_classification": {
                    "macro_precision": raw_macro_precision,
                    "macro_recall": raw_macro_recall,
                    "macro_f1": raw_macro_f1,
                    "weighted_precision": raw_weighted_precision, 
                    "weighted_recall": raw_weighted_recall,
                    "weighted_f1": raw_weighted_f1
                },
                
                "level_classification": {
                    "macro_precision": level_macro_precision,
                    "macro_recall": level_macro_recall,
                    "macro_f1": level_macro_f1,
                    "weighted_precision": level_weighted_precision,
                    "weighted_recall": level_weighted_recall,
                    "weighted_f1": level_weighted_f1
                },
                
                "binary_classification": {
                    "macro_precision": binary_macro_precision,
                    "macro_recall": binary_macro_recall,
                    "macro_f1": binary_macro_f1,
                    "weighted_precision": binary_weighted_precision,
                    "weighted_recall": binary_weighted_recall,
                    "weighted_f1": binary_weighted_f1
                }
            }
            
        return {
            "story_classification": {
                "accuracy": story_accuracy,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
                "weighted_f1": weighted_f1,
                "mcc": story_mcc
            },
            "rating_metrics": rating_metrics
        }

    def calculate_average_metrics(self, fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate average metrics across all folds
        """
        avg_metrics = {
            "story_classification": {
                "accuracy": {"mean": 0.0, "std": 0.0},
                "micro_precision": {"mean": 0.0, "std": 0.0},
                "micro_recall": {"mean": 0.0, "std": 0.0},
                "micro_f1": {"mean": 0.0, "std": 0.0},
                "macro_precision": {"mean": 0.0, "std": 0.0},
                "macro_recall": {"mean": 0.0, "std": 0.0},
                "macro_f1": {"mean": 0.0, "std": 0.0},
                "weighted_precision": {"mean": 0.0, "std": 0.0},
                "weighted_recall": {"mean": 0.0, "std": 0.0},
                "weighted_f1": {"mean": 0.0, "std": 0.0},
                "mcc": {"mean": 0.0, "std": 0.0}
            },
            "rating_metrics": {
                "suspense": {},
                "curiosity": {},
                "surprise": {}
            }
        }
        
        story_metrics = {
            "accuracy": [],
            "micro_precision": [],
            "micro_recall": [],
            "micro_f1": [],
            "macro_precision": [],
            "macro_recall": [],
            "macro_f1": [],
            "weighted_precision": [],
            "weighted_recall": [],
            "weighted_f1": [],
            "mcc": []
        }
        
        rating_metrics = {
            "suspense": {
                "perfect_accuracy": [], "off_by_one_accuracy": [], "level_accuracy": [],
                "rmse": [], "mae": [], "quadratic_weighted_kappa": [],
                "binary_accuracy": [],
                "raw_classification": {
                    "macro_precision": [], "macro_recall": [], "macro_f1": [],
                    "weighted_precision": [], "weighted_recall": [], "weighted_f1": []
                },
                "level_classification": {
                    "macro_precision": [], "macro_recall": [], "macro_f1": [],
                    "weighted_precision": [], "weighted_recall": [], "weighted_f1": []
                },
                "binary_classification": {
                    "macro_precision": [], "macro_recall": [], "macro_f1": [],
                    "weighted_precision": [], "weighted_recall": [], "weighted_f1": []
                }
            },
            "curiosity": {
                "perfect_accuracy": [], "off_by_one_accuracy": [], "level_accuracy": [],
                "rmse": [], "mae": [], "quadratic_weighted_kappa": [],
                "binary_accuracy": [],
                "raw_classification": {
                    "macro_precision": [], "macro_recall": [], "macro_f1": [],
                    "weighted_precision": [], "weighted_recall": [], "weighted_f1": []
                },
                "level_classification": {
                    "macro_precision": [], "macro_recall": [], "macro_f1": [],
                    "weighted_precision": [], "weighted_recall": [], "weighted_f1": []
                },
                "binary_classification": {
                    "macro_precision": [], "macro_recall": [], "macro_f1": [],
                    "weighted_precision": [], "weighted_recall": [], "weighted_f1": []
                }
            },
            "surprise": {
                "perfect_accuracy": [], "off_by_one_accuracy": [], "level_accuracy": [],
                "rmse": [], "mae": [], "quadratic_weighted_kappa": [],
                "binary_accuracy": [],
                "raw_classification": {
                    "macro_precision": [], "macro_recall": [], "macro_f1": [],
                    "weighted_precision": [], "weighted_recall": [], "weighted_f1": []
                },
                "level_classification": {
                    "macro_precision": [], "macro_recall": [], "macro_f1": [],
                    "weighted_precision": [], "weighted_recall": [], "weighted_f1": []
                },
                "binary_classification": {
                    "macro_precision": [], "macro_recall": [], "macro_f1": [],
                    "weighted_precision": [], "weighted_recall": [], "weighted_f1": []
                }
            }
        }
        
        for fold_metric in fold_metrics:
            for key in story_metrics.keys():
                story_metrics[key].append(fold_metric["story_classification"][key])
            
            for rating in ["suspense", "curiosity", "surprise"]:
                for metric_key in ["perfect_accuracy", "off_by_one_accuracy", "level_accuracy", 
                                "rmse", "mae", "quadratic_weighted_kappa", "binary_accuracy"]:
                    if metric_key in fold_metric["rating_metrics"][rating]:
                        rating_metrics[rating][metric_key].append(fold_metric["rating_metrics"][rating][metric_key])
                
                for class_type in ["raw_classification", "level_classification", "binary_classification"]:
                    if class_type in fold_metric["rating_metrics"][rating]:
                        for metric_key in ["macro_precision", "macro_recall", "macro_f1", 
                                        "weighted_precision", "weighted_recall", "weighted_f1"]:
                            if metric_key in fold_metric["rating_metrics"][rating][class_type]:
                                rating_metrics[rating][class_type][metric_key].append(
                                    fold_metric["rating_metrics"][rating][class_type][metric_key])
        
        for key in story_metrics.keys():
            avg_metrics["story_classification"][key]["mean"] = np.mean(story_metrics[key])
            avg_metrics["story_classification"][key]["std"] = np.std(story_metrics[key])
        
        for rating in ["suspense", "curiosity", "surprise"]:
            avg_metrics["rating_metrics"][rating] = {}
            
            for metric_key in ["perfect_accuracy", "off_by_one_accuracy", "level_accuracy", 
                            "rmse", "mae", "quadratic_weighted_kappa", "binary_accuracy"]:
                if metric_key in rating_metrics[rating] and rating_metrics[rating][metric_key]:
                    avg_metrics["rating_metrics"][rating][metric_key] = {
                        "mean": np.mean(rating_metrics[rating][metric_key]),
                        "std": np.std(rating_metrics[rating][metric_key])
                    }
            
            for class_type in ["raw_classification", "level_classification", "binary_classification"]:
                if class_type in rating_metrics[rating]:
                    avg_metrics["rating_metrics"][rating][class_type] = {}
                    for metric_key in ["macro_precision", "macro_recall", "macro_f1", 
                                    "weighted_precision", "weighted_recall", "weighted_f1"]:
                        if metric_key in rating_metrics[rating][class_type] and rating_metrics[rating][class_type][metric_key]:
                            avg_metrics["rating_metrics"][rating][class_type][metric_key] = {
                                "mean": np.mean(rating_metrics[rating][class_type][metric_key]),
                                "std": np.std(rating_metrics[rating][class_type][metric_key])
                            }
        
        return avg_metrics

    def create_summary_report(self, all_results: Dict[str, Any]):
        """
        Create summary report with metrics for all configs
        """
        story_rows = []
        for config_name, results in all_results.items():
            metrics = results["average_metrics"]["story_classification"]
            row = {
                "Configuration": config_name,
                "Accuracy": f"{metrics['accuracy']['mean']:.4f} ± {metrics['accuracy']['std']:.4f}",
                "Micro Precision": f"{metrics['micro_precision']['mean']:.4f} ± {metrics['micro_precision']['std']:.4f}",
                "Micro Recall": f"{metrics['micro_recall']['mean']:.4f} ± {metrics['micro_recall']['std']:.4f}",
                "Micro F1": f"{metrics['micro_f1']['mean']:.4f} ± {metrics['micro_f1']['std']:.4f}",
                "Macro Precision": f"{metrics['macro_precision']['mean']:.4f} ± {metrics['macro_precision']['std']:.4f}",
                "Macro Recall": f"{metrics['macro_recall']['mean']:.4f} ± {metrics['macro_recall']['std']:.4f}",
                "Macro F1": f"{metrics['macro_f1']['mean']:.4f} ± {metrics['macro_f1']['std']:.4f}",
                "Weighted Precision": f"{metrics['weighted_precision']['mean']:.4f} ± {metrics['weighted_precision']['std']:.4f}",
                "Weighted Recall": f"{metrics['weighted_recall']['mean']:.4f} ± {metrics['weighted_recall']['std']:.4f}",
                "Weighted F1": f"{metrics['weighted_f1']['mean']:.4f} ± {metrics['weighted_f1']['std']:.4f}",
                "MCC": f"{metrics['mcc']['mean']:.4f} ± {metrics['mcc']['std']:.4f}"
            }
            story_rows.append(row)
        story_df = pd.DataFrame(story_rows)
        
        rating_dfs = {}
        
        for rating in ["suspense", "curiosity", "surprise"]:
            main_rows = []
            for config_name, results in all_results.items():
                metrics = results["average_metrics"]["rating_metrics"][rating]
                row = {
                    "Configuration": config_name,
                    "Perfect Accuracy": f"{metrics.get('perfect_accuracy', {}).get('mean', 0):.4f} ± {metrics.get('perfect_accuracy', {}).get('std', 0):.4f}",
                    "Off-by-One Accuracy": f"{metrics.get('off_by_one_accuracy', {}).get('mean', 0):.4f} ± {metrics.get('off_by_one_accuracy', {}).get('std', 0):.4f}",
                    "Level Accuracy (Low/Med/High)": f"{metrics.get('level_accuracy', {}).get('mean', 0):.4f} ± {metrics.get('level_accuracy', {}).get('std', 0):.4f}",
                    "Binary Accuracy (≤2.5/>2.5)": f"{metrics.get('binary_accuracy', {}).get('mean', 0):.4f} ± {metrics.get('binary_accuracy', {}).get('std', 0):.4f}",
                    "RMSE": f"{metrics.get('rmse', {}).get('mean', 0):.4f} ± {metrics.get('rmse', {}).get('std', 0):.4f}",
                    "MAE": f"{metrics.get('mae', {}).get('mean', 0):.4f} ± {metrics.get('mae', {}).get('std', 0):.4f}",
                    "Quadratic Weighted Kappa": f"{metrics.get('quadratic_weighted_kappa', {}).get('mean', 0):.4f} ± {metrics.get('quadratic_weighted_kappa', {}).get('std', 0):.4f}"
                }
                main_rows.append(row)
            rating_dfs[f"{rating}_main"] = pd.DataFrame(main_rows)
            
            raw_rows = []
            for config_name, results in all_results.items():
                if "raw_classification" in results["average_metrics"]["rating_metrics"][rating]:
                    metrics = results["average_metrics"]["rating_metrics"][rating]["raw_classification"]
                    row = {
                        "Configuration": config_name,
                        "Macro Precision (1-5)": f"{metrics.get('macro_precision', {}).get('mean', 0):.4f} ± {metrics.get('macro_precision', {}).get('std', 0):.4f}",
                        "Macro Recall (1-5)": f"{metrics.get('macro_recall', {}).get('mean', 0):.4f} ± {metrics.get('macro_recall', {}).get('std', 0):.4f}",
                        "Macro F1 (1-5)": f"{metrics.get('macro_f1', {}).get('mean', 0):.4f} ± {metrics.get('macro_f1', {}).get('std', 0):.4f}",
                        "Weighted Precision (1-5)": f"{metrics.get('weighted_precision', {}).get('mean', 0):.4f} ± {metrics.get('weighted_precision', {}).get('std', 0):.4f}",
                        "Weighted Recall (1-5)": f"{metrics.get('weighted_recall', {}).get('mean', 0):.4f} ± {metrics.get('weighted_recall', {}).get('std', 0):.4f}",
                        "Weighted F1 (1-5)": f"{metrics.get('weighted_f1', {}).get('mean', 0):.4f} ± {metrics.get('weighted_f1', {}).get('std', 0):.4f}"
                    }
                    raw_rows.append(row)
            rating_dfs[f"{rating}_raw"] = pd.DataFrame(raw_rows)
            
            level_rows = []
            for config_name, results in all_results.items():
                if "level_classification" in results["average_metrics"]["rating_metrics"][rating]:
                    metrics = results["average_metrics"]["rating_metrics"][rating]["level_classification"]
                    row = {
                        "Configuration": config_name,
                        "Macro Precision (Low/Med/High)": f"{metrics.get('macro_precision', {}).get('mean', 0):.4f} ± {metrics.get('macro_precision', {}).get('std', 0):.4f}",
                        "Macro Recall (Low/Med/High)": f"{metrics.get('macro_recall', {}).get('mean', 0):.4f} ± {metrics.get('macro_recall', {}).get('std', 0):.4f}",
                        "Macro F1 (Low/Med/High)": f"{metrics.get('macro_f1', {}).get('mean', 0):.4f} ± {metrics.get('macro_f1', {}).get('std', 0):.4f}",
                        "Weighted Precision (Low/Med/High)": f"{metrics.get('weighted_precision', {}).get('mean', 0):.4f} ± {metrics.get('weighted_precision', {}).get('std', 0):.4f}",
                        "Weighted Recall (Low/Med/High)": f"{metrics.get('weighted_recall', {}).get('mean', 0):.4f} ± {metrics.get('weighted_recall', {}).get('std', 0):.4f}",
                        "Weighted F1 (Low/Med/High)": f"{metrics.get('weighted_f1', {}).get('mean', 0):.4f} ± {metrics.get('weighted_f1', {}).get('std', 0):.4f}"
                    }
                    level_rows.append(row)
            rating_dfs[f"{rating}_level"] = pd.DataFrame(level_rows)
            
            binary_rows = []
            for config_name, results in all_results.items():
                if "binary_classification" in results["average_metrics"]["rating_metrics"][rating]:
                    metrics = results["average_metrics"]["rating_metrics"][rating]["binary_classification"]
                    row = {
                        "Configuration": config_name,
                        "Macro Precision (≤2.5/>2.5)": f"{metrics.get('macro_precision', {}).get('mean', 0):.4f} ± {metrics.get('macro_precision', {}).get('std', 0):.4f}",
                        "Macro Recall (≤2.5/>2.5)": f"{metrics.get('macro_recall', {}).get('mean', 0):.4f} ± {metrics.get('macro_recall', {}).get('std', 0):.4f}",
                        "Macro F1 (≤2.5/>2.5)": f"{metrics.get('macro_f1', {}).get('mean', 0):.4f} ± {metrics.get('macro_f1', {}).get('std', 0):.4f}",
                        "Weighted Precision (≤2.5/>2.5)": f"{metrics.get('weighted_precision', {}).get('mean', 0):.4f} ± {metrics.get('weighted_precision', {}).get('std', 0):.4f}",
                        "Weighted Recall (≤2.5/>2.5)": f"{metrics.get('weighted_recall', {}).get('mean', 0):.4f} ± {metrics.get('weighted_recall', {}).get('std', 0):.4f}",
                        "Weighted F1 (≤2.5/>2.5)": f"{metrics.get('weighted_f1', {}).get('mean', 0):.4f} ± {metrics.get('weighted_f1', {}).get('std', 0):.4f}"
                    }
                    binary_rows.append(row)
            rating_dfs[f"{rating}_binary"] = pd.DataFrame(binary_rows)
        
        story_df.to_csv(os.path.join(self.output_dir, "story_classification_summary.csv"), index=False)
        
        for rating in ["suspense", "curiosity", "surprise"]:
            if f"{rating}_main" in rating_dfs:
                rating_dfs[f"{rating}_main"].to_csv(
                    os.path.join(self.output_dir, f"{rating}_main_metrics_summary.csv"), index=False)
            
            if f"{rating}_raw" in rating_dfs:
                rating_dfs[f"{rating}_raw"].to_csv(
                    os.path.join(self.output_dir, f"{rating}_raw_classification_summary.csv"), index=False)
            
            if f"{rating}_level" in rating_dfs:
                rating_dfs[f"{rating}_level"].to_csv(
                    os.path.join(self.output_dir, f"{rating}_level_classification_summary.csv"), index=False)
            
            if f"{rating}_binary" in rating_dfs:
                rating_dfs[f"{rating}_binary"].to_csv(
                    os.path.join(self.output_dir, f"{rating}_binary_classification_summary.csv"), index=False)
        
        self.create_summary_plots(all_results)
        print("\nEnhanced summary reports created in the output directory.")

    def create_summary_plots(self, all_results: Dict[str, Any]):
        """
        Create simple standard summary plots for the metrics
        """
        for rating in ["suspense", "curiosity", "surprise"]:
            fig, axes = plt.subplots(3, 1, figsize=(12, 18))
            fig.suptitle(f"{rating.capitalize()} Rating Classification Metrics", fontsize=16)
            
            configs = list(all_results.keys())
            
            raw_f1_means = []
            raw_f1_stds = []
            level_f1_means = []
            level_f1_stds = []
            binary_f1_means = []
            binary_f1_stds = []
            
            for config in configs:
                try:
                    raw_f1_mean = all_results[config]["average_metrics"]["rating_metrics"][rating]["raw_classification"]["macro_f1"]["mean"]
                    raw_f1_std = all_results[config]["average_metrics"]["rating_metrics"][rating]["raw_classification"]["macro_f1"]["std"]
                except (KeyError, TypeError):
                    raw_f1_mean = 0
                    raw_f1_std = 0
                raw_f1_means.append(raw_f1_mean)
                raw_f1_stds.append(raw_f1_std)
                
                try:
                    level_f1_mean = all_results[config]["average_metrics"]["rating_metrics"][rating]["level_classification"]["macro_f1"]["mean"]
                    level_f1_std = all_results[config]["average_metrics"]["rating_metrics"][rating]["level_classification"]["macro_f1"]["std"]
                except (KeyError, TypeError):
                    level_f1_mean = 0
                    level_f1_std = 0
                level_f1_means.append(level_f1_mean)
                level_f1_stds.append(level_f1_std)
                
                try:
                    binary_f1_mean = all_results[config]["average_metrics"]["rating_metrics"][rating]["binary_classification"]["macro_f1"]["mean"]
                    binary_f1_std = all_results[config]["average_metrics"]["rating_metrics"][rating]["binary_classification"]["macro_f1"]["std"]
                except (KeyError, TypeError):
                    binary_f1_mean = 0
                    binary_f1_std = 0
                binary_f1_means.append(binary_f1_mean)
                binary_f1_stds.append(binary_f1_std)
            
            x = np.arange(len(configs))
            width = 0.3
            axes[0].bar(x, raw_f1_means, width, yerr=raw_f1_stds, capsize=5, label='Macro F1 Score')
            axes[0].set_xlabel('Configuration')
            axes[0].set_ylabel('Score')
            axes[0].set_title(f'{rating.capitalize()} - Raw (1-5) Classification Metrics')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(configs, rotation=45, ha='right')
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.7)
            
            axes[1].bar(x, level_f1_means, width, yerr=level_f1_stds, capsize=5, label='Macro F1 Score')
            axes[1].set_xlabel('Configuration')
            axes[1].set_ylabel('Score')
            axes[1].set_title(f'{rating.capitalize()} - Level (Low/Med/High) Classification Metrics')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(configs, rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.7)
            
            axes[2].bar(x, binary_f1_means, width, yerr=binary_f1_stds, capsize=5, label='Macro F1 Score')
            axes[2].set_xlabel('Configuration')
            axes[2].set_ylabel('Score')
            axes[2].set_title(f'{rating.capitalize()} - Binary (≤2.5/>2.5) Classification Metrics')
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(configs, rotation=45, ha='right')
            axes[2].legend()
            axes[2].grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(self.output_dir, f"{rating}_classification_metrics.png"))
            plt.close()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(configs))
            width = 0.25
            
            perfect_acc_means = []
            level_acc_means = []
            binary_acc_means = []
            perfect_acc_stds = []
            level_acc_stds = []
            binary_acc_stds = []
            
            for config in configs:
                try:
                    perfect_acc_mean = all_results[config]["average_metrics"]["rating_metrics"][rating]["perfect_accuracy"]["mean"]
                    perfect_acc_std = all_results[config]["average_metrics"]["rating_metrics"][rating]["perfect_accuracy"]["std"]
                except (KeyError, TypeError):
                    perfect_acc_mean = 0
                    perfect_acc_std = 0
                perfect_acc_means.append(perfect_acc_mean)
                perfect_acc_stds.append(perfect_acc_std)
                
                try:
                    level_acc_mean = all_results[config]["average_metrics"]["rating_metrics"][rating]["level_accuracy"]["mean"]
                    level_acc_std = all_results[config]["average_metrics"]["rating_metrics"][rating]["level_accuracy"]["std"]
                except (KeyError, TypeError):
                    level_acc_mean = 0
                    level_acc_std = 0
                level_acc_means.append(level_acc_mean)
                level_acc_stds.append(level_acc_std)
                
                try:
                    binary_acc_mean = all_results[config]["average_metrics"]["rating_metrics"][rating]["binary_accuracy"]["mean"]
                    binary_acc_std = all_results[config]["average_metrics"]["rating_metrics"][rating]["binary_accuracy"]["std"]
                except (KeyError, TypeError):
                    binary_acc_mean = 0
                    binary_acc_std = 0
                binary_acc_means.append(binary_acc_mean)
                binary_acc_stds.append(binary_acc_std)
            
            ax.bar(x - width, perfect_acc_means, width, yerr=perfect_acc_stds, capsize=5, 
                label='Perfect Accuracy (1-5)', color='royalblue')
            ax.bar(x, level_acc_means, width, yerr=level_acc_stds, capsize=5, 
                label='Level Accuracy (Low/Med/High)', color='forestgreen')
            ax.bar(x + width, binary_acc_means, width, yerr=binary_acc_stds, capsize=5, 
                label='Binary Accuracy (≤2.5/>2.5)', color='darkorange')
            
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{rating.capitalize()} - Accuracy Comparison Across Scales')
            ax.set_xticks(x)
            ax.set_xticklabels(configs, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{rating}_accuracy_comparison.png"))
            plt.close()


class PostAnalyser:
    """
    Class to prompt model to analyse and classify posts and comments
    """
    def __init__(self, client, model_name: str = MODEL, shot: str = "multi", temperature: float = 0.6,
                 chain_of_thought: bool = True, use_structured_output: bool = True):
        """..."""
        self.client = client
        self.model = model_name
        self.shot = shot
        self.temperature = temperature
        self.chain_of_thought = chain_of_thought
        self.use_structured_output = use_structured_output
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """
        Create the prompt for the model (based on config)
        """
        base = """You are an empathic, caring, world-curious, eager linguist specialising in narrative analysis and text classification.
        
You will classify ONE input text at a time, in JSON, according to the following guidelines.
Use the guidelines below to support your decisions, but ultimately, follow your best judgment and more than not it is a story.
A text is a story if it describes a sequence of events involving one or more people. Stories must include multiple events.
It's ok if the events are out of order, but there should still be some sequence.
These texts can contain multiple stories, or one small story, or no story at all. Most of the time, the text will contain a story.
So, if you are not sure, classify it as a story.
Stories can be experienced in different ways, and the same story can be experienced differently by different readers.
These experiences are called readers' perceptions.
There are three readers' perceptions you will rate within these texts:\n
- SUSPENSE: "Triggers desire for information about future events and postpones a feeling of resolution."
- CURIOSITY: "Triggers desire for information about past events and leaves me wondering about missing information."
- SURPRISE: "Presents information that I experience as unexpected."

Task:
Decide whether the text CONTAINS a story or not AND rate your readers' perceptions of the text on a scale from 1 to 5.
\n
\n
"""
        if self.chain_of_thought:
            base += """
Let's think step by step. Follow these steps to complete the task:\n
1. Read the text fully.
2. Decide whether the text CONTAINS a story or not. Most of the time, the text will contain a story you might have missed.
3. Rate SUSPENSE (future information) on a scale from 1 to 5.
4. Rate CURIOSITY (past information) on a scale from 1 to 5.
5. Rate SURPRISE (unexpected information) on a scale from 1 to 5.
6. Provide the final output in JSON format. Ensure the output is valid JSON. Ensure the ratings are integers between 1 and 5.
\n
\n
"""
        if self.use_structured_output:
            base += """
You MUST provide the output in a structured JSON format with the following structure:\n
{
  "story_class": "Story" or "Not Story",
  "suspense": integer (1-5),
  "curiosity": integer (1-5),
  "surprise": integer (1-5)
}
\n
"""
        examples = {
            "one": """```json
            [
                {
                    "body": "As I read it, the OP did not consent even when drunk. Those the consent is dubious not just because of the alcohol involved but also because I think there might not have been any consent whatsoever. \n\nFurthermore, on Mandy's part, if she's partying with the secondary intention of banging the OP even after he's made it completely clear while relatively sober that he does not want to have sex with her, she is entirely to blame. The OP's lack of consent was not just implied (through an absence of active consent) but actually explicit. Now it's possible that Mandy was drinking with no intention of banging the OP but that after enough alcohol, she couldn't control herself even though she expected she would be able to. Then perhaps she's not entirely to blame, but any residual blame goes to the alcohol rather than to the OP. \n\nNow, if you were to wake up in someone's bed after you had told that person explicitly while sober, \"I don't want to have sex with you,\" would you still be blaming yourself? Especially given that the other person initiated the act and you were too out of it to really do anything to prevent it?",
    "story_class": "Story",
    "suspense": 4,
    "curiosity": 3,
    "surprise": 4
                }
            ]
            ```""",
            "few": """```json
            [
                {
                    "body": "As I read it, the OP did not consent even when drunk. Those the consent is dubious not just because of the alcohol involved but also because I think there might not have been any consent whatsoever. \n\nFurthermore, on Mandy's part, if she's partying with the secondary intention of banging the OP even after he's made it completely clear while relatively sober that he does not want to have sex with her, she is entirely to blame. The OP's lack of consent was not just implied (through an absence of active consent) but actually explicit. Now it's possible that Mandy was drinking with no intention of banging the OP but that after enough alcohol, she couldn't control herself even though she expected she would be able to. Then perhaps she's not entirely to blame, but any residual blame goes to the alcohol rather than to the OP. \n\nNow, if you were to wake up in someone's bed after you had told that person explicitly while sober, \"I don't want to have sex with you,\" would you still be blaming yourself? Especially given that the other person initiated the act and you were too out of it to really do anything to prevent it?",
    "story_class": "Story",
    "suspense": 4,
    "curiosity": 3,
    "surprise": 4
                },
                {
                    "body": "Rooney extremely overreacted. \n\nIt's one thing to pursue a student for being absent, it's another thing to devote his entire day to it and attempt to break into his home.",
    "story_class": "Not Story",
    "suspense": 3,
    "curiosity": 2,
    "surprise": 3
                },
                {
                    "body": "Students and cafeteria loos, usually disaster. I checked a dozen loo berths and it was all underhang. I even took a picture in disbelief! \n\nNow I think it must have been policy, sick of finding rolls \"spun\" onto the floor.",
    "story_class": "Story",
    "suspense": 1,
    "curiosity": 2,
    "surprise": 2
                }
            ]
            ```""",
            "multi": """```json
            [
                {
                    "body": "As I read it, the OP did not consent even when drunk. Those the consent is dubious not just because of the alcohol involved but also because I think there might not have been any consent whatsoever. \n\nFurthermore, on Mandy's part, if she's partying with the secondary intention of banging the OP even after he's made it completely clear while relatively sober that he does not want to have sex with her, she is entirely to blame. The OP's lack of consent was not just implied (through an absence of active consent) but actually explicit. Now it's possible that Mandy was drinking with no intention of banging the OP but that after enough alcohol, she couldn't control herself even though she expected she would be able to. Then perhaps she's not entirely to blame, but any residual blame goes to the alcohol rather than to the OP. \n\nNow, if you were to wake up in someone's bed after you had told that person explicitly while sober, \"I don't want to have sex with you,\" would you still be blaming yourself? Especially given that the other person initiated the act and you were too out of it to really do anything to prevent it?",
    "story_class": "Story",
    "suspense": 4,
    "curiosity": 3,
    "surprise": 4
                },
                {
                    "body": "Rooney extremely overreacted. \n\nIt's one thing to pursue a student for being absent, it's another thing to devote his entire day to it and attempt to break into his home.",
    "story_class": "Not Story",
    "suspense": 3,
    "curiosity": 2,
    "surprise": 3
                },
                {
                    "body": "Students and cafeteria loos, usually disaster. I checked a dozen loo berths and it was all underhang. I even took a picture in disbelief! \n\nNow I think it must have been policy, sick of finding rolls \"spun\" onto the floor.",
    "story_class": "Story",
    "suspense": 1,
    "curiosity": 2,
    "surprise": 2
                },
                {
                    "body": "The physical evidence shows that he was not turned around and that he was moving toward the police officer. \n\nYou can't argue with physical evidence.",
    "story_class": "Not Story",
    "suspense": 2,
    "curiosity": 2,
    "surprise": 3
                },
                {
                    "body": "About a month ago I posted a CMV about how I think tipping should be made illegal (my view was NOT changed), and today I read [this](http://jayporter.com/dispatches/observations-from-a-tipless-restaurant-part-1-overview/) article about a San Diego restaurant that forbids tips!",
    "story_class": "Story",
    "suspense": 3,
    "curiosity": 2,
    "surprise": 2
                },
                {
                    "body": "Nobody is calling the police questioning him \"police brutality.\"  It's all the physical intimidation and pushing him on the ground.",
    "story_class": "Not Story",
    "suspense": 4,
    "curiosity": 2,
    "surprise": 2
                },
                {
                    "body": "It is not necessary to have a complex character for a character study in feature film. The character only needs to be interesting. I think that Plainview is plenty interesting. There was some ambiguity in his character that keeps you intrigued. There is obviously something to his relationship with H.W. that goes beyond simple exploitation. Saving him from the oil well and the way he cradle him afterwards shows that there is some sort feelings there. Also how he greets H.W. on his return from the school again shows that something has developed between them. Also his interactions with Henry show that he wanted some sort of human relationship otherwise he would of just sent him away after their first meeting.\n\nI didn't have any problem with the structure of the movie or the leaps in time. This is the first I've heard of anybody having issue with that as well. I don't know what else to say about that.\n\nI don't think broader context was really necessary for the film and in fact think that the intense focus on Plainview was one of its strengths. I think broadening the focus would have meant that the film would have lost its tension.\n\nI don't see Plainview killing Eli as puzzling. In fact I think it is really the only ending that would have made sense. What happens to somebody that is driven by success to sociopathic ends after they win that success? They turn on themselves and anybody around them. This is why it also showed Plainview pushing H.W. out of his life for good.",
    "story_class": "Story",
    "suspense": 2,
    "curiosity": 4,
    "surprise": 3
                },
                {
                   "body": "He also had devil traps under all his rugs, a car full of shotguns, saw monsters everywhere and almost killed his neighbor's dog.\n\nComparatively, Sam just had a house, a dog, a bitch, and no crazy.",
    "story_class": "Story",
    "suspense": 1,
    "curiosity": 3,
    "surprise": 4
                }
            ]
            ```"""
        }
        if self.shot == "zero":
            return base + "\n\nDo not include any text outside of this JSON format. Ensure your response is valid JSON."
        else:
            return (
                base
                + "\n\n" + "EXAMPLES:\n\n" +
                examples[self.shot] + "\n\n" +
                "Do not include any text outside of this JSON format. Ensure your response is valid JSON."
            )

    def analyse_text(self, text: str) -> Dict[str, Any]:
        """
        Set up prompt and error handling to save results
        """
        prompt = f"Text to analyse:\n\n{text}"
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                response_format={
                    "type": "json_object"} if self.use_structured_output else None
            )
            response_text = response.choices[0].message.content
            try:
                result = json.loads(response_text)
                if "story_class" not in result:
                    raise ValueError("Missing 'story_class' in response")
                if result["story_class"] not in ["Story", "Not Story"]:
                    result["story_class"] = "Not Story" if result["story_class"].lower(
                    ) in ["not story", "not a story", "no story"] else "Story"
                for key in ["suspense", "curiosity", "surprise"]:
                    if key not in result:
                        result[key] = 1
                    else:
                        try:
                            result[key] = max(1, min(5, int(result[key])))
                        except (ValueError, TypeError):
                            result[key] = 1
                return result
            except json.JSONDecodeError:
                import re
                json_match = re.search(r"({.*})", response_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        if "story_class" not in result:
                            raise ValueError(
                                "Missing 'story_class' in response")
                        if result["story_class"] not in ["Story", "Not Story"]:
                            result["story_class"] = "Not Story" if result["story_class"].lower(
                            ) in ["not story", "not a story", "no story"] else "Story"

                        for key in ["suspense", "curiosity", "surprise"]:
                            if key not in result:
                                result[key] = 1
                            else:
                                try:
                                    result[key] = max(
                                        1, min(5, int(result[key])))
                                except (ValueError, TypeError):
                                    result[key] = 1
                        return result
                    except:
                        pass
                story_match = re.search(
                    r'story_class["\']\s*:\s*["\']([^"\']+)["\']', response_text, re.IGNORECASE)
                story_class = "Not Story"
                if story_match:
                    extracted = story_match.group(1)
                    story_class = "Story" if "story" in extracted.lower(
                    ) and "not" not in extracted.lower() else "Not Story"
                ratings = {}
                for key in ["suspense", "curiosity", "surprise"]:
                    rating_match = re.search(
                        f'{key}["\']\s*:\s*(\d+)', response_text, re.IGNORECASE)
                    ratings[key] = int(rating_match.group(1)
                                       ) if rating_match else 1
                    ratings[key] = max(1, min(5, ratings[key]))
                return {
                    "story_class": story_class,
                    "suspense": ratings["suspense"],
                    "curiosity": ratings["curiosity"],
                    "surprise": ratings["surprise"],
                    "extraction_method": "regex_fallback"
                }
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            return {
                "story_class": "Not Story",
                "suspense": 1,
                "curiosity": 1,
                "surprise": 1,
                "error": str(e)
            }


def main():
    client = OpenAI(base_url=BASEURL, api_key=APIKEY)
    #input_file = "gs-train.json"
    input_file = "gs-test.json"
    n_splits = 5
    evaluator = CrossValidationEvaluator(client, input_file, n_splits=n_splits)
    all_results = evaluator.run_cross_validation(model_name=MODEL)
    sample_text = "I believe that the conclusions reached by the House Select Committee on Assassinations (HSCA) regarding the assassination of JFK on Nov. 22, 1963 are true and accurate. CMV.\n\nI believe that Lee Harvey Oswald assassinated JFK using a single rifle, fired all of the shots that killed him and wounded Gov. Connally, and did so as an individual actor (not as the agent of a nation or organized group). My knowledge of the Kennedy Assassination is admittedly limited; I've studied it in school and participated in several university level class discussions about the investigations and conspiracies surrounding the issue. After looking at the conclusions reached by the Warren Commission, I was somewhat skeptical of the notion that Lee Harvey Oswald acted alone, but the HSCA report clarifies some of the issues surrounding the timing and circumstances of the assassination, and leaves open the idea that Oswald may have had a partner. Additionally, I do feel that the events following the assassination are slightly suspicious (notably LBJ appointing and overseeing the Warren Commission), but are not at present sufficient for me to discount the accepted explanation of the assassination.  The number of conspiracy or alternate theories surrounding the assassination is prodigious and confusing, to say the least, but I am willing to accept an alternate theory and CMV if someone can present a compelling alternate explanation of the JFK assassination supported by clear and provable facts."
    analyser = PostAnalyser(client=client, model_name=MODEL, shot="multi", temperature=0.6,
                            chain_of_thought=True, use_structured_output=True)
    result = analyser.analyse_text(sample_text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
