import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Tuple
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    cohen_kappa_score, 
    mean_squared_error, 
    roc_auc_score, 
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


class NarrativeEvaluator:
    """
    Evaluates narrative analysis results against gold standard data,
    calculating various metrics for both nominal and ordinal classifications.
    """
    
    def __init__(self, gold_standard_file: str, results_dir: str):
        """
        Initialize the evaluator with paths to gold standard and results.
        
        Args:
            gold_standard_file: Path to the gold standard JSON file
            results_dir: Directory containing the different model results
        """
        self.gold_standard_file = gold_standard_file
        self.results_dir = results_dir
        self.gold_standard = self._load_gold_standard()
        self.result_files = self._get_result_files()
        self.metrics = {}
        
    def _load_gold_standard(self) -> List[Dict[str, Any]]:
        """Load the gold standard data from file."""
        with open(self.gold_standard_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_result_files(self) -> List[str]:
        """Get all result files from the results directory."""
        return [f for f in os.listdir(self.results_dir) 
                if f.startswith("gold_standard_") and f.endswith(".json")]
    
    def _load_results(self, result_file: str) -> List[Dict[str, Any]]:
        """Load results from a specific result file."""
        with open(os.path.join(self.results_dir, result_file), 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_paired_data(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, List]]:
        """
        Create paired data for evaluation by matching gold standard with results.
        
        Returns a dictionary with keys for each metric type and values containing
        the true and predicted values.
        """
        # Create id-to-index mapping for faster lookup
        gold_id_map = {item["id"]: i for i, item in enumerate(self.gold_standard)}
        
        # Initialize containers for paired data
        paired_data = {
            "story_class": {"true": [], "pred": []},
            "suspense": {"true": [], "pred": []},
            "curiosity": {"true": [], "pred": []},
            "surprise": {"true": [], "pred": []}
        }
        
        for result in results:
            result_id = result["id"]
            if result_id in gold_id_map:
                gold_idx = gold_id_map[result_id]
                gold_item = self.gold_standard[gold_idx]
                
                # Skip entries with errors or missing values
                if (result.get("story_class") == "Error" or 
                    not all(key in result for key in ["story_class", "suspense", "curiosity", "surprise"])):
                    continue
                
                # Story class (nominal)
                paired_data["story_class"]["true"].append(gold_item["story_class"])
                paired_data["story_class"]["pred"].append(result["story_class"])
                
                # Ratings (ordinal)
                for rating in ["suspense", "curiosity", "surprise"]:
                    paired_data[rating]["true"].append(gold_item[rating])
                    paired_data[rating]["pred"].append(result[rating])
        
        return paired_data
    
    def _calculate_binary_metrics(self, true_values: List[str], pred_values: List[str]) -> Dict[str, float]:
        """Calculate metrics for binary classification (Story vs. Not Story)."""
        # Convert string labels to binary (1 for "Story", 0 for "Not Story")
        y_true = np.array([1 if v == "Story" else 0 for v in true_values])
        y_pred = np.array([1 if v == "Story" else 0 for v in pred_values])
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate precision, recall, f1-score
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        
        # Calculate Cohen's kappa
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Calculate weighted kappa (equal to standard kappa for binary classification)
        weighted_kappa = cohen_kappa_score(y_true, y_pred, weights='linear')
        
        # Calculate ROC-AUC
        try:
            roc_auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            # Handle cases where there might be only one class
            roc_auc = float('nan')
        
        # Create confusion matrix for later visualization
        conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cohen_kappa": kappa,
            "weighted_kappa": weighted_kappa,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix.tolist()
        }
    
    def _calculate_ordinal_metrics(self, true_values: List[int], pred_values: List[int]) -> Dict[str, float]:
        """Calculate metrics for ordinal ratings (1-5 Likert scale)."""
        y_true = np.array(true_values)
        y_pred = np.array(pred_values)
        
        # Mean Squared Error
        mse = mean_squared_error(y_true, y_pred)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Weighted Cohen's kappa for ordinal data
        weighted_kappa = cohen_kappa_score(y_true, y_pred, weights='linear')
        
        # Calculate accuracy (exact matches)
        accuracy = np.mean(y_true == y_pred)
        
        # Calculate "off-by-one" accuracy (predictions within ±1 of true value)
        off_by_one = np.mean(np.abs(y_true - y_pred) <= 1)
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "weighted_kappa": weighted_kappa,
            "accuracy": accuracy,
            "off_by_one_accuracy": off_by_one
        }
    
    def evaluate_all(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Evaluate all result files and compute metrics.
        
        Returns:
            A nested dictionary with metrics for each shot technique and metric type.
        """
        for result_file in self.result_files:
            # Extract shot technique from filename
            shot = result_file.replace("gold_standard_", "").replace("_shot.json", "")
            
            # Load results for this shot technique
            results = self._load_results(result_file)
            
            # Create paired data for evaluation
            paired_data = self._create_paired_data(results)
            
            # Initialize metrics for this shot technique
            self.metrics[shot] = {}
            
            # Calculate binary metrics for story classification
            self.metrics[shot]["story_class"] = self._calculate_binary_metrics(
                paired_data["story_class"]["true"],
                paired_data["story_class"]["pred"]
            )
            
            # Calculate ordinal metrics for ratings
            for rating in ["suspense", "curiosity", "surprise"]:
                self.metrics[shot][rating] = self._calculate_ordinal_metrics(
                    paired_data[rating]["true"],
                    paired_data[rating]["pred"]
                )
        
        return self.metrics
    
    def _calculate_cross_validation_stats(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate mean and standard deviation for cross-validation results.
        
        Args:
            metrics_list: List of metric dictionaries from each fold
            
        Returns:
            Dictionary with mean and std for each metric
        """
        all_metrics = {}
        
        # Get all unique metric names
        metric_names = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())
        
        # Remove 'confusion_matrix' as it can't be averaged directly
        if 'confusion_matrix' in metric_names:
            metric_names.remove('confusion_matrix')
        
        # Calculate mean and std for each metric
        for metric in metric_names:
            values = [m[metric] for m in metrics_list if metric in m]
            if values:
                all_metrics[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values)
                }
        
        return all_metrics
    
    def simulate_cross_validation(self, num_folds: int = 5, seed: int = 42) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
        """
        Simulate cross-validation by randomly splitting the data multiple times.
        
        Args:
            num_folds: Number of folds for cross-validation
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with cross-validation results for each shot technique and metric type
        """
        np.random.seed(seed)
        
        # Initialize cross-validation results
        cv_results = {}
        
        for result_file in self.result_files:
            # Extract shot technique from filename
            shot = result_file.replace("gold_standard_", "").replace("_shot.json", "")
            
            # Load results for this shot technique
            results = self._load_results(result_file)
            
            # Create full paired data
            full_paired_data = self._create_paired_data(results)
            
            # Initialize metrics for each fold
            fold_metrics = {
                "story_class": [],
                "suspense": [],
                "curiosity": [],
                "surprise": []
            }
            
            # Get total number of samples
            n_samples = len(full_paired_data["story_class"]["true"])
            
            # Create indices for each fold
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            fold_size = n_samples // num_folds
            
            # Simulate cross-validation
            for fold in range(num_folds):
                # Get test indices for this fold
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < num_folds - 1 else n_samples
                test_indices = indices[start_idx:end_idx]
                
                # Create fold data
                fold_data = {
                    "story_class": {"true": [], "pred": []},
                    "suspense": {"true": [], "pred": []},
                    "curiosity": {"true": [], "pred": []},
                    "surprise": {"true": [], "pred": []}
                }
                
                # Fill fold data
                for idx in test_indices:
                    for key in fold_data:
                        fold_data[key]["true"].append(full_paired_data[key]["true"][idx])
                        fold_data[key]["pred"].append(full_paired_data[key]["pred"][idx])
                
                # Calculate metrics for this fold
                fold_metrics["story_class"].append(
                    self._calculate_binary_metrics(
                        fold_data["story_class"]["true"],
                        fold_data["story_class"]["pred"]
                    )
                )
                
                for rating in ["suspense", "curiosity", "surprise"]:
                    fold_metrics[rating].append(
                        self._calculate_ordinal_metrics(
                            fold_data[rating]["true"],
                            fold_data[rating]["pred"]
                        )
                    )
            
            # Calculate cross-validation stats
            cv_results[shot] = {}
            for key, metrics_list in fold_metrics.items():
                cv_results[shot][key] = self._calculate_cross_validation_stats(metrics_list)
        
        return cv_results
    
    def generate_summary_table(self, cross_val_results: Dict = None) -> pd.DataFrame:
        """
        Generate a summary table with all metrics.
        
        Args:
            cross_val_results: Optional cross-validation results to include
            
        Returns:
            DataFrame with summary metrics
        """
        # Determine which results to use
        results_to_use = cross_val_results if cross_val_results else self.metrics
        
        # Prepare data for DataFrame
        table_data = []
        
        for shot, shot_metrics in results_to_use.items():
            for category, metrics in shot_metrics.items():
                for metric_name, metric_value in metrics.items():
                    # Skip confusion matrix
                    if metric_name == 'confusion_matrix':
                        continue
                    
                    # Format value based on whether it's from cross-validation
                    if cross_val_results:
                        value = f"{metric_value['mean']:.4f} ± {metric_value['std']:.4f}"
                    else:
                        value = f"{metric_value:.4f}"
                    
                    table_data.append({
                        "Shot": shot,
                        "Category": category,
                        "Metric": metric_name,
                        "Value": value
                    })
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Pivot for better readability
        pivot_df = df.pivot_table(
            index=["Category", "Metric"],
            columns=["Shot"],
            values="Value",
            aggfunc=lambda x: x
        )
        
        return pivot_df
    
    def save_summary_table(self, output_file: str, cross_val_results: Dict = None):
        """
        Save the summary table to a CSV file.
        
        Args:
            output_file: Path to save the CSV file
            cross_val_results: Optional cross-validation results to include
        """
        summary_table = self.generate_summary_table(cross_val_results)
        summary_table.to_csv(output_file)
        print(f"Summary table saved to {output_file}")
    
    def plot_confusion_matrices(self, output_dir: str = "plots"):
        """
        Plot confusion matrices for story classification.
        
        Args:
            output_dir: Directory to save plots
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for shot, shot_metrics in self.metrics.items():
            if "story_class" in shot_metrics and "confusion_matrix" in shot_metrics["story_class"]:
                conf_matrix = np.array(shot_metrics["story_class"]["confusion_matrix"])
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    conf_matrix, 
                    annot=True, 
                    fmt='d',
                    cmap='Blues',
                    xticklabels=["Not Story", "Story"],
                    yticklabels=["Not Story", "Story"]
                )
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix for {shot}-shot')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'confusion_matrix_{shot}.png'))
                plt.close()


def run_evaluation(gold_standard_file: str, results_dir: str, output_dir: str = "evaluation_results", cv_folds: int = 5):
    """
    Run the full evaluation and save results.
    
    Args:
        gold_standard_file: Path to the gold standard JSON file
        results_dir: Directory containing the different model results
        output_dir: Directory to save evaluation results
        cv_folds: Number of folds for cross-validation
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = NarrativeEvaluator(gold_standard_file, results_dir)
    
    # Run standard evaluation
    print("Calculating standard metrics...")
    metrics = evaluator.evaluate_all()
    
    # Save standard metrics
    with open(os.path.join(output_dir, "metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate and save summary table
    evaluator.save_summary_table(os.path.join(output_dir, "metrics_summary.csv"))
    
    # Plot confusion matrices
    evaluator.plot_confusion_matrices(os.path.join(output_dir, "plots"))
    
    # Run cross-validation
    print(f"Simulating {cv_folds}-fold cross-validation...")
    cv_results = evaluator.simulate_cross_validation(num_folds=cv_folds)
    
    # Save cross-validation results
    with open(os.path.join(output_dir, "cv_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(cv_results, f, indent=2)
    
    # Generate and save cross-validation summary table
    evaluator.save_summary_table(
        os.path.join(output_dir, "cv_metrics_summary.csv"),
        cross_val_results=cv_results
    )
    
    print(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    gold_standard_file = "gs-test.json"
    results_dir = "llm_analysis_results"
    output_dir = "evaluation_results"
    
    run_evaluation(
        gold_standard_file=gold_standard_file,
        results_dir=results_dir,
        output_dir=output_dir,
        cv_folds=5
    )
