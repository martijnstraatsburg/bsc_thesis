#!/usr/bin/env python3

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

BASEURL = "http://localhost:8000/v1/"
APIKEY = "EMPTY"
MODEL = "Qwen/Qwen3-4B" # Un-comment to use 4B variant
#MODEL = "Qwen/Qwen3-8B"

class CrossValidationEvaluator:
    def __init__(self, client, input_file: str, n_splits: int = 5, output_dir: str = "4B-train-cv_results"):
        """
        Initialize the cross-validation evaluator.
        Args:
            client: OpenAI client instance.
            input_file (str): Path to the input JSON file containing the dataset.
            n_splits (int): Number of folds for cross-validation.
            output_dir (str): Directory to save evaluation results.
        """
        self.client = client
        self.n_splits = n_splits
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
            
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        self.configurations = [
            {"shot": "zero", "temperature": 0.1, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "one", "temperature": 0.1, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "few", "temperature": 0.1, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "multi", "temperature": 0.1, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "zero", "temperature": 0.2, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "one", "temperature": 0.2, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "few", "temperature": 0.2, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "multi", "temperature": 0.2, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "zero", "temperature": 0.3, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "one", "temperature": 0.3, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "few", "temperature": 0.3, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "multi", "temperature": 0.3, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "zero", "temperature": 0.4, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "one", "temperature": 0.4, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "few", "temperature": 0.4, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "multi", "temperature": 0.4, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "zero", "temperature": 0.5, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "one", "temperature": 0.5, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "few", "temperature": 0.5, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "multi", "temperature": 0.5, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "zero", "temperature": 0.6, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "one", "temperature": 0.6, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "few", "temperature": 0.6, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "multi", "temperature": 0.6, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "zero", "temperature": 0.7, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "one", "temperature": 0.7, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "few", "temperature": 0.7, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "multi", "temperature": 0.7, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "zero", "temperature": 0.8, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "one", "temperature": 0.8, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "few", "temperature": 0.8, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "multi", "temperature": 0.8, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "zero", "temperature": 0.9, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "one", "temperature": 0.9, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "few", "temperature": 0.9, "chain_of_thought": True, "use_structured_output": True},
            {"shot": "multi", "temperature": 0.9, "chain_of_thought": True, "use_structured_output": True}
        ]
    
    def run_cross_validation(self, model_name: str = MODEL):
        """
        Run cross-validation for all configurations.
        Args:
            model_name (str): Name of the model to use.
        """
        all_results = {}
        
        for config in self.configurations:
            config_name = f"{config['shot']}_shot_temp{config['temperature']}_cot{'Yes' if config['chain_of_thought'] else 'No'}_structured{'Yes' if config['use_structured_output'] else 'No'}"
            print(f"\n--- Cross-validating with {config_name} ---")
            
            fold_results = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(self.kf.split(self.dataset)):
                print(f"  Fold {fold_idx+1}/{self.n_splits}")
                
                train_data = [self.dataset[i] for i in train_idx]
                test_data = [self.dataset[i] for i in test_idx]
                
                analyser = PostAnalyser(
                    client=self.client, 
                    model_name=model_name, 
                    shot=config['shot'],
                    temperature=config['temperature'],
                    chain_of_thought=config['chain_of_thought'],
                    use_structured_output=config['use_structured_output']
                )
                
                test_predictions = copy.deepcopy(test_data)
                
                for i, entry in enumerate(tqdm(test_predictions, desc=f"Analyzing fold {fold_idx+1}")):
                    text = entry["body"]
                    analysis = analyser.analyse_text(text)
                    
                    if "error" not in analysis:
                        entry["predicted_story_class"] = analysis["story_class"]
                        entry["predicted_suspense"] = analysis["suspense"]
                        entry["predicted_curiosity"] = analysis["curiosity"]
                        entry["predicted_surprise"] = analysis["surprise"]
                    else:
                        print(f"Error analyzing entry {i+1}: {analysis['error']}")
                        entry["predicted_story_class"] = "Not Story"
                        entry["predicted_suspense"] = 1
                        entry["predicted_curiosity"] = 1
                        entry["predicted_surprise"] = 1
                
                fold_metric = self.calculate_metrics(test_data, test_predictions)
                fold_results.append(fold_metric)
                
                fold_output = os.path.join(self.output_dir, f"{config_name}_fold{fold_idx+1}_predictions.json")
                with open(fold_output, 'w', encoding='utf-8') as f:
                    json.dump(test_predictions, f, indent=2, ensure_ascii=False)
            
            avg_metrics = self.calculate_average_metrics(fold_results)
            all_results[config_name] = {
                "average_metrics": avg_metrics,
                "fold_metrics": fold_results
            }
            
            config_output = os.path.join(self.output_dir, f"{config_name}_cv_results.json")
            with open(config_output, 'w', encoding='utf-8') as f:
                json.dump(all_results[config_name], f, indent=2, ensure_ascii=False)
            
            print(f"\nAverage metrics for {config_name}:")
            print(json.dumps(avg_metrics, indent=2))
        
        self.create_summary_report(all_results)
        
        return all_results
    
    def calculate_metrics(self, true_data: List[Dict], pred_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate evaluation metrics by comparing true labels with predictions.
        Args:
            true_data (List[Dict]): List of entries with true labels.
            pred_data (List[Dict]): List of entries with predicted labels.
        Returns:
            Dict[str, Any]: Dictionary of evaluation metrics.
        """
        y_true_story = [1 if item["story_class"] == "Story" else 0 for item in true_data]
        y_pred_story = [1 if item["predicted_story_class"] == "Story" else 0 for item in pred_data]
        
        # Story classification metrics
        story_accuracy = accuracy_score(y_true_story, y_pred_story)
        
        # Micro average (treats all instances equally)
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true_story, y_pred_story, average='micro'
        )
        
        # Macro average (treats all classes equally)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true_story, y_pred_story, average='macro'
        )
        
        # Matthews correlation coefficient
        story_mcc = matthews_corrcoef(y_true_story, y_pred_story)
        
        # Rating metrics for suspense, curiosity, surprise
        rating_metrics = {}
        for rating in ["suspense", "curiosity", "surprise"]:
            y_true_rating = [item[rating] for item in true_data]
            y_pred_rating = [item[f"predicted_{rating}"] for item in pred_data]
            
            # Perfect accuracy (exact match)
            perfect_accuracy = accuracy_score(y_true_rating, y_pred_rating)
            
            # Off-by-one accuracy
            off_by_one_accuracy = sum(abs(y_true - y_pred) <= 1 for y_true, y_pred in zip(y_true_rating, y_pred_rating)) / len(y_true_rating)
            
            # Root Mean Squared Error
            rmse = np.sqrt(mean_squared_error(y_true_rating, y_pred_rating))
            
            # Mean Absolute Error
            mae = mean_absolute_error(y_true_rating, y_pred_rating)
            
            # Quadratic Weighted Kappa
            weighted_kappa = cohen_kappa_score(y_true_rating, y_pred_rating, weights='quadratic')
            
            rating_metrics[rating] = {
                "perfect_accuracy": perfect_accuracy,
                "off_by_one_accuracy": off_by_one_accuracy,
                "rmse": rmse,
                "mae": mae,
                "quadratic_weighted_kappa": weighted_kappa
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
                "mcc": story_mcc
            },
            "rating_metrics": rating_metrics
        }

    def calculate_average_metrics(self, fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate average metrics across all folds.
        Args:
            fold_metrics (List[Dict[str, Any]]): List of metrics from each fold.
        Returns:
            Dict[str, Any]: Dictionary of average metrics with standard deviations.
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
            "mcc": []
        }
        
        rating_metrics = {
            "suspense": {
                "perfect_accuracy": [], "off_by_one_accuracy": [],
                "rmse": [], "mae": [], "quadratic_weighted_kappa": []
            },
            "curiosity": {
                "perfect_accuracy": [], "off_by_one_accuracy": [],
                "rmse": [], "mae": [], "quadratic_weighted_kappa": []
            },
            "surprise": {
                "perfect_accuracy": [], "off_by_one_accuracy": [],
                "rmse": [], "mae": [], "quadratic_weighted_kappa": []
            }
        }
        
        for fold_metric in fold_metrics:
            for key in story_metrics.keys():
                story_metrics[key].append(fold_metric["story_classification"][key])
            
            for rating in ["suspense", "curiosity", "surprise"]:
                for metric_key in rating_metrics[rating].keys():
                    rating_metrics[rating][metric_key].append(fold_metric["rating_metrics"][rating][metric_key])
        
        for key in story_metrics.keys():
            avg_metrics["story_classification"][key]["mean"] = np.mean(story_metrics[key])
            avg_metrics["story_classification"][key]["std"] = np.std(story_metrics[key])
        
        for rating in ["suspense", "curiosity", "surprise"]:
            avg_metrics["rating_metrics"][rating] = {}
            for metric_key in rating_metrics[rating].keys():
                avg_metrics["rating_metrics"][rating][metric_key] = {
                    "mean": np.mean(rating_metrics[rating][metric_key]),
                    "std": np.std(rating_metrics[rating][metric_key])
                }
        
        return avg_metrics

    def create_summary_report(self, all_results: Dict[str, Any]):
        """
        Create a summary report comparing all configurations.
        Args:
            all_results (Dict[str, Any]): Results from all configurations.
        """
        # Create story classification summary
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
                "MCC": f"{metrics['mcc']['mean']:.4f} ± {metrics['mcc']['std']:.4f}"
            }
            story_rows.append(row)
        
        story_df = pd.DataFrame(story_rows)
        
        # Create rating metrics summary for each rating
        rating_dfs = {}
        for rating in ["suspense", "curiosity", "surprise"]:
            rows = []
            for config_name, results in all_results.items():
                metrics = results["average_metrics"]["rating_metrics"][rating]
                row = {
                    "Configuration": config_name,
                    "Perfect Accuracy": f"{metrics['perfect_accuracy']['mean']:.4f} ± {metrics['perfect_accuracy']['std']:.4f}",
                    "Off-by-One Accuracy": f"{metrics['off_by_one_accuracy']['mean']:.4f} ± {metrics['off_by_one_accuracy']['std']:.4f}",
                    "RMSE": f"{metrics['rmse']['mean']:.4f} ± {metrics['rmse']['std']:.4f}",
                    "MAE": f"{metrics['mae']['mean']:.4f} ± {metrics['mae']['std']:.4f}",
                    "Quadratic Weighted Kappa": f"{metrics['quadratic_weighted_kappa']['mean']:.4f} ± {metrics['quadratic_weighted_kappa']['std']:.4f}"
                }
                rows.append(row)
            rating_dfs[rating] = pd.DataFrame(rows)
        
        # Save CSV files
        story_df.to_csv(os.path.join(self.output_dir, "story_classification_summary.csv"), index=False)
        for rating, df in rating_dfs.items():
            df.to_csv(os.path.join(self.output_dir, f"{rating}_rating_summary.csv"), index=False)
        
        # Create summary plots
        self.create_summary_plots(all_results)
        
        print("\nSummary reports created in the output directory.")

    def create_summary_plots(self, all_results: Dict[str, Any]):
        """
        Create summary plots comparing different configurations.
        Args:
            all_results (Dict[str, Any]): Results from all configurations.
        """
        configs = list(all_results.keys())
        
        # Plot for story classification metrics
        story_metrics = ["accuracy", "micro_precision", "micro_recall", "micro_f1", 
                         "macro_precision", "macro_recall", "macro_f1"]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(configs))
        width = 0.1  # Reduced width to accommodate more metrics
        
        for i, metric in enumerate(story_metrics):
            means = [all_results[config]["average_metrics"]["story_classification"][metric]["mean"] for config in configs]
            stds = [all_results[config]["average_metrics"]["story_classification"][metric]["std"] for config in configs]
            
            offset = width * (i - len(story_metrics)/2)
            rects = ax.bar(x + offset, means, width, label=metric.replace('_', ' ').title(), yerr=stds)
        
        ax.set_ylabel('Score')
        ax.set_title('Story Classification Metrics by Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels([c.split('_shot')[0] for c in configs], rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "story_classification_metrics.png"))
        
        # Create a separate plot just for MCC
        fig, ax = plt.subplots(figsize=(12, 6))
        means = [all_results[config]["average_metrics"]["story_classification"]["mcc"]["mean"] for config in configs]
        stds = [all_results[config]["average_metrics"]["story_classification"]["mcc"]["std"] for config in configs]
        
        ax.bar(x, means, width=0.6, yerr=stds, color='purple')
        
        ax.set_ylabel('MCC Score')
        ax.set_title('Matthews Correlation Coefficient (MCC) by Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels([c.split('_shot')[0] for c in configs], rotation=45, ha='right')
        ax.set_ylim(-1, 1)  # MCC ranges from -1 to 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "mcc_comparison.png"))
        
        # Plot for each rating metric
        for rating in ["suspense", "curiosity", "surprise"]:
            # Accuracy comparison plot
            fig, ax = plt.subplots(figsize=(12, 6))
            accuracy_types = ["perfect_accuracy", "off_by_one_accuracy"]
            labels = ["Perfect Match", "Off-by-One"]
            
            x = np.arange(len(configs))
            width = 0.35
            
            for i, acc_type in enumerate(accuracy_types):
                means = [all_results[config]["average_metrics"]["rating_metrics"][rating][acc_type]["mean"] for config in configs]
                stds = [all_results[config]["average_metrics"]["rating_metrics"][rating][acc_type]["std"] for config in configs]
                
                offset = width * (i - 0.5)
                rects = ax.bar(x + offset, means, width, label=labels[i], yerr=stds)
            
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{rating.capitalize()} Rating - Accuracy Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([c.split('_shot')[0] for c in configs], rotation=45, ha='right')
            ax.legend(loc='upper left')
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{rating}_accuracy_comparison.png"))
            
            # Error metrics plot
            fig, ax = plt.subplots(figsize=(12, 6))
            error_metrics = ["rmse", "mae"]
            labels = ["RMSE", "MAE"]
            
            x = np.arange(len(configs))
            width = 0.35
            
            for i, metric in enumerate(error_metrics):
                means = [all_results[config]["average_metrics"]["rating_metrics"][rating][metric]["mean"] for config in configs]
                stds = [all_results[config]["average_metrics"]["rating_metrics"][rating][metric]["std"] for config in configs]
                
                offset = width * (i - 0.5)
                rects = ax.bar(x + offset, means, width, label=labels[i], yerr=stds)
            
            ax.set_ylabel('Error')
            ax.set_title(f'{rating.capitalize()} Rating - Error Metrics')
            ax.set_xticks(x)
            ax.set_xticklabels([c.split('_shot')[0] for c in configs], rotation=45, ha='right')
            ax.legend(loc='upper left')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{rating}_error_metrics.png"))
        
        # Quadratic Weighted Kappa comparison plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ratings = ["suspense", "curiosity", "surprise"]
        
        x = np.arange(len(configs))
        width = 0.25
        
        for i, rating in enumerate(ratings):
            means = [all_results[config]["average_metrics"]["rating_metrics"][rating]["quadratic_weighted_kappa"]["mean"] for config in configs]
            stds = [all_results[config]["average_metrics"]["rating_metrics"][rating]["quadratic_weighted_kappa"]["std"] for config in configs]
            
            offset = width * (i - 1)
            rects = ax.bar(x + offset, means, width, label=rating.capitalize(), yerr=stds)
        
        ax.set_ylabel('Quadratic Weighted Kappa')
        ax.set_title('Quadratic Weighted Kappa Comparison Across Ratings')
        ax.set_xticks(x)
        ax.set_xticklabels([c.split('_shot')[0] for c in configs], rotation=45, ha='right')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "quadratic_weighted_kappa_comparison.png"))


class PostAnalyser:
    def __init__(self, client, model_name: str = MODEL, shot: str = "multi", temperature: float = 0.6, 
                 chain_of_thought: bool = True, use_structured_output: bool = True):
        """
        Initialise PostAnalyser with OpenAI client and model parameters.
        Args:
            client: OpenAI client instance.
            model_name (str): Name of the model to use.
            shot (str): Shot technique to use ('zero', 'one', 'few', 'multi').
            temperature (float): Temperature for sampling.
            chain_of_thought (bool): Whether to use chain of thought reasoning.
            use_structured_output (bool): Whether to use structured output format.
        """
        self.client = client
        self.model = model_name
        self.shot = shot
        self.temperature = temperature
        self.chain_of_thought = chain_of_thought
        self.use_structured_output = use_structured_output
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """
        Create the system prompt with persona and instructions.
        Returns:
            str: The system prompt for the model.
        """
        base = """You are an empathic, caring, world-curious, eager linguist specializing in narrative analysis and text classification.
        
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
        Analyse the provided text and return structured results.
        Args:
            text (str): The text to analyse.
        Returns:
            Dict[str, Any]: Analysis results including story classification and ratings.
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
                response_format={"type": "json_object"} if self.use_structured_output else None
            )

            response_text = response.choices[0].message.content

            try:
                result = json.loads(response_text)
                if "story_class" not in result:
                    raise ValueError("Missing 'story_class' in response")
                if result["story_class"] not in ["Story", "Not Story"]:
                    result["story_class"] = "Not Story" if result["story_class"].lower() in ["not story", "not a story", "no story"] else "Story"
                
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
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        if "story_class" not in result:
                            raise ValueError("Missing 'story_class' in response")
                        if result["story_class"] not in ["Story", "Not Story"]:
                            result["story_class"] = "Not Story" if result["story_class"].lower() in ["not story", "not a story", "no story"] else "Story"
                        
                        for key in ["suspense", "curiosity", "surprise"]:
                            if key not in result:
                                result[key] = 1
                            else:
                                try:
                                    result[key] = max(1, min(5, int(result[key])))
                                except (ValueError, TypeError):
                                    result[key] = 1
                        
                        return result
                    except:
                        pass
                
                story_match = re.search(r'story_class["\']\s*:\s*["\']([^"\']+)["\']', response_text, re.IGNORECASE)
                story_class = "Not Story"
                if story_match:
                    extracted = story_match.group(1)
                    story_class = "Story" if "story" in extracted.lower() and "not" not in extracted.lower() else "Not Story"
                
                ratings = {}
                for key in ["suspense", "curiosity", "surprise"]:
                    rating_match = re.search(f'{key}["\']\s*:\s*(\d+)', response_text, re.IGNORECASE)
                    ratings[key] = int(rating_match.group(1)) if rating_match else 1
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
    input_file = "gs-train.json"
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
