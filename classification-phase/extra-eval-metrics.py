#!/usr/bin/env python3

# Author: Martijn Straatsburg
# Name: extra-eval-metrics.py
# Description: This script evaluates the predictions of a model on the golden standard.
# It extra calculates the level and binary transformed evaluation metrics for 3 RPs.

import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

LEVEL_MAP = {'low': 0, 'medium': 1, 'high': 2}
DEFAULT_LEVEL = 1
BIN_THRESHOLD = 2.5


def load_preds(path):
    data = json.load(open(path, 'r', encoding='utf-8'))
    true = {
        'suspense':    [d['suspense'] for d in data],
        'curiosity':   [d['curiosity'] for d in data],
        'surprise':    [d['surprise'] for d in data],
        'suspense_lv':   [LEVEL_MAP.get(str(d.get('suspense_level', '')).lower(), DEFAULT_LEVEL) for d in data],
        'curiosity_lv':  [LEVEL_MAP.get(str(d.get('curiosity_level', '')).lower(), DEFAULT_LEVEL) for d in data],
        'surprise_lv':   [LEVEL_MAP.get(str(d.get('surprise_level', '')).lower(), DEFAULT_LEVEL) for d in data],
        'suspense_bin':  [1 if d['suspense'] > BIN_THRESHOLD else 0 for d in data],
        'curiosity_bin': [1 if d['curiosity'] > BIN_THRESHOLD else 0 for d in data],
        'surprise_bin':  [1 if d['surprise'] > BIN_THRESHOLD else 0 for d in data],
    }
    pred = {
        'suspense':    [d.get('predicted_suspense', d.get('suspense', 0)) for d in data],
        'curiosity':   [d.get('predicted_curiosity', d.get('curiosity', 0)) for d in data],
        'surprise':    [d.get('predicted_surprise', d.get('surprise', 0)) for d in data],
        'suspense_lv':  [LEVEL_MAP.get(str(d.get('suspense_level_pred', d.get('predicted_suspense_level', ''))).lower(), DEFAULT_LEVEL) for d in data],
        'curiosity_lv': [LEVEL_MAP.get(str(d.get('curiosity_level_pred', d.get('predicted_curiosity_level', ''))).lower(), DEFAULT_LEVEL) for d in data],
        'surprise_lv':  [LEVEL_MAP.get(str(d.get('surprise_level_pred', d.get('predicted_surprise_level', ''))).lower(), DEFAULT_LEVEL) for d in data],
        'suspense_bin':  [1 if d.get('predicted_suspense', d.get('suspense', 0)) > BIN_THRESHOLD else 0 for d in data],
        'curiosity_bin': [1 if d.get('predicted_curiosity', d.get('curiosity', 0)) > BIN_THRESHOLD else 0 for d in data],
        'surprise_bin':  [1 if d.get('predicted_surprise', d.get('surprise', 0)) > BIN_THRESHOLD else 0 for d in data],
    }
    return true, pred


def eval_fold(true, pred):
    metrics = {}
    for emo in ('suspense', 'curiosity', 'surprise'):
        y_true = true[emo]
        y_pred = pred[emo]
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1,2,3,4,5], average='macro', zero_division=0)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1,2,3,4,5], average='weighted', zero_division=0)
        metrics[f'{emo}_p_macro']    = p_mac
        metrics[f'{emo}_r_macro']    = r_mac
        metrics[f'{emo}_f1_macro']   = f1_mac
        metrics[f'{emo}_p_weighted'] = p_w
        metrics[f'{emo}_r_weighted'] = r_w
        metrics[f'{emo}_f1_weighted'] = f1_w

    for emo_lv in ('suspense_lv', 'curiosity_lv', 'surprise_lv'):
        metrics[f'{emo_lv}_acc'] = accuracy_score(true[emo_lv], pred[emo_lv])

    for emo_bin in ('suspense_bin', 'curiosity_bin', 'surprise_bin'):
        metrics[f'{emo_bin}_acc'] = accuracy_score(true[emo_bin], pred[emo_bin])

    return metrics


def main(paths):
    all_metrics = []
    for p in paths:
        true, pred = load_preds(p)
        m = eval_fold(true, pred)
        print(f'=== {p} ===')
        for k, v in sorted(m.items()):
            print(f'{k:25s}: {v:.4f}')
        print()
        all_metrics.append(m)

    print('=== Mean & Std across folds ===')
    keys = sorted(all_metrics[0].keys())
    for k in keys:
        vals = [m[k] for m in all_metrics]
        mean, std = np.mean(vals), np.std(vals, ddof=1)
        print(f'{k:25s}: mean {mean:.4f}, std {std:.4f}')


if __name__ == '__main__':
    paths = [
        'multi_shot_temp0.6_cotYes_structuredYes_fold1_predictions.json',
        'multi_shot_temp0.6_cotYes_structuredYes_fold2_predictions.json',
        'multi_shot_temp0.6_cotYes_structuredYes_fold3_predictions.json',
        'multi_shot_temp0.6_cotYes_structuredYes_fold4_predictions.json',
        'multi_shot_temp0.6_cotYes_structuredYes_fold5_predictions.json'
    ]
    main(paths)
