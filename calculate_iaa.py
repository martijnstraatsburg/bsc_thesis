#!/usr/bin/env python3

# Name: detailed_iaa.py
# Author: Martijn Straatsburg (updated)
# Description: Performs detailed inter-annotator agreement analysis including
# Fleiss' Kappa, ICC, pairwise Cohen's Kappa with linear and quadratic weighting
# for multiple categories (curiosity, surprise, suspense).

import json
from collections import defaultdict
import pandas as pd
import numpy as np
import pingouin as pg
from statsmodels.stats.inter_rater import fleiss_kappa
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

# Load annotations
with open('numbers-annotations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Group by content
content_groups = defaultdict(list)
for entry in data:
    content_groups[entry['name']].append(entry)

# Fleiss' Kappa for story classification
fleiss_data = []
for name, entries in content_groups.items():
    story_counts = {'Story': 0, 'Not Story': 0}
    for entry in entries:
        story_counts['Story' if entry['story_class']=='Story' else 'Not Story'] += 1
    fleiss_data.append([story_counts['Story'], story_counts['Not Story']])
kappa = fleiss_kappa(fleiss_data)
print(f"Fleiss' Kappa for story_class: {kappa:.3f}")

# Prepare ICC data
def prepare_icc_data(entries, rating_key):
    icc_data = []
    for entry in entries:
        icc_data.append({
            'target': entry['name'],
            'rater': entry['annotator'],
            'rating': entry[rating_key][0]['rating']
        })
    return pd.DataFrame(icc_data)

# Categories for numeric ratings
categories = ['curiosity', 'surprise', 'suspense']

# Compute ICC(3,k) for each category
icc_results = {}
for cat in categories:
    df_icc = prepare_icc_data(data, cat)
    icc_df = pg.intraclass_corr(data=df_icc, targets='target', raters='rater', ratings='rating')
    icc_val = icc_df.loc[icc_df['Type']=='ICC3k', 'ICC'].values[0]
    icc_results[cat] = icc_val
    print(f"ICC(3,k) for {cat}: {icc_val:.3f}")

# Build structured DataFrame of all annotations
annotators = sorted({e['annotator'] for e in data})
structured = []
for e in data:
    item = {
        'name': e['name'],
        'annotator': e['annotator'],
        'story_class': 1 if e['story_class']=='Story' else 0
    }
    for cat in categories:
        item[cat] = e[cat][0]['rating']
    structured.append(item)
df_ann = pd.DataFrame(structured)

# Pairwise Cohen's Kappa for binary story_class
print("\n==== Pairwise Cohen's Kappa for story_class ====")
for i, a1 in enumerate(annotators):
    for a2 in annotators[i+1:]:
        m1 = df_ann[df_ann['annotator']==a1]
        m2 = df_ann[df_ann['annotator']==a2]
        merged = pd.merge(m1[['name','story_class']], m2[['name','story_class']], on='name', suffixes=('_1','_2'))
        k = cohen_kappa_score(merged['story_class_1'], merged['story_class_2'])
        print(f"{a1} vs {a2}: {k:.3f}")

# Pairwise weighted Cohen's Kappa for numeric ratings
print("\n==== Pairwise Quadratic Weighted Kappa ====")
for cat in categories:
    print(f"\n-- {cat.capitalize()} --")
    # Pivot table: rows=items, cols=annotators, values=ratings
    pivot = df_ann.pivot(index='name', columns='annotator', values=cat)
    w_kappa = pd.DataFrame(index=annotators, columns=annotators, dtype=float)
    for i, a1 in enumerate(annotators):
        for j, a2 in enumerate(annotators):
            if a1 == a2:
                w_kappa.loc[a1, a2] = 1.0
            else:
                # get common items
                s1 = pivot[a1].dropna()
                s2 = pivot[a2].dropna()
                common = s1.index.intersection(s2.index)
                w_kappa.loc[a1, a2] = cohen_kappa_score(
                    s1.loc[common], s2.loc[common], weights='quadratic'
                )
    print(w_kappa.to_string(float_format="%.3f"))

# (rest of visualizations and bias analysis unchanged)
