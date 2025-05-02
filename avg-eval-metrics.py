import json
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_records(path):
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)

def extract_labels(records, key):
    return [rec.get(key) for rec in records]

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate prediction metrics against golden standard')
    parser.add_argument('--gold', required=True, help='Path to golden standard JSON file')
    parser.add_argument('--pred', required=True, help='Path to predictions JSON file')
    args = parser.parse_args()

    gold = load_records(args.gold)
    pred = load_records(args.pred)

    if len(gold) != len(pred):
        raise ValueError('Number of records in gold and pred must match')

    fields = ['story_class', 'suspense', 'curiosity', 'surprise']
    overall_accuracy = []
    results = {}

    for field in fields:
        y_true = extract_labels(gold, field)
        y_pred = extract_labels(pred, field)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0)

        results[field] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        overall_accuracy.append(acc)

    # Overall accuracy average across fields
    avg_accuracy = sum(overall_accuracy) / len(overall_accuracy)

    # Print results
    print(f"Average accuracy across fields: {avg_accuracy:.4f}\n")
    for field, metrics in results.items():
        print(f"Metrics for {field}:")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1_score']:.4f}\n")

if __name__ == '__main__':
    main()
