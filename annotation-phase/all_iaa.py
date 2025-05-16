#!/usr/bin/env python3

# Name: detailed_iaa_with_visuals.py
# Author: Martijn Straatsburg (updated with visualizations & CSV export)
# Description: Performs detailed inter-annotator agreement analysis including
# Fleiss' Kappa, ICC, pairwise Cohen's Kappa with linear and quadratic weighting
# for categories (curiosity, surprise, suspense), outputs CSVs, and generates visualizations.

import json
from collections import defaultdict
import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

# Load annotations
with open('numbers-annotations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Group by content for Fleiss
content_groups = defaultdict(list)
for entry in data:
    content_groups[entry['name']].append(entry)

# Fleiss' Kappa for story_class
fleiss_data = []
for name, entries in content_groups.items():
    counts = {'Story': 0, 'Not Story': 0}
    for e in entries:
        counts['Story' if e['story_class']=='Story' else 'Not Story'] += 1
    fleiss_data.append([counts['Story'], counts['Not Story']])
fleiss_val = fleiss_kappa(fleiss_data)
print(f"Fleiss' Kappa for story_class: {fleiss_val:.3f}")

# Prepare structured DataFrame
annotators = sorted({e['annotator'] for e in data})
categories = ['curiosity', 'surprise', 'suspense']
records = []
for e in data:
    rec = {'name': e['name'], 'annotator': e['annotator'],
           'story_class': 1 if e['story_class']=='Story' else 0}
    for cat in categories:
        rec[cat] = e[cat][0]['rating']
    records.append(rec)
df = pd.DataFrame(records)

# Pairwise Cohen's Kappa for binary story_class
print("\nPairwise Cohen's Kappa (binary story_class):")
binary_mat = pd.DataFrame(index=annotators, columns=annotators, dtype=float)
for a1 in annotators:
    for a2 in annotators:
        if a1 == a2:
            binary_mat.loc[a1, a2] = 1.0
        else:
            m1 = df[df['annotator']==a1][['name','story_class']]
            m2 = df[df['annotator']==a2][['name','story_class']]
            merged = pd.merge(m1, m2, on='name', suffixes=('_1','_2'))
            binary_mat.loc[a1, a2] = cohen_kappa_score(
                merged['story_class_1'], merged['story_class_2']
            )
print(binary_mat)

# Pairwise Quadratic Weighted Kappa for numeric ratings
print("\nPairwise Quadratic Weighted Kappa:")
weighted_mats = {}
for cat in categories:
    pivot = df.pivot(index='name', columns='annotator', values=cat)
    mat = pd.DataFrame(index=annotators, columns=annotators, dtype=float)
    for a1 in annotators:
        for a2 in annotators:
            if a1 == a2:
                mat.loc[a1, a2] = 1.0
            else:
                s1 = pivot[a1].dropna()
                s2 = pivot[a2].dropna()
                common = s1.index.intersection(s2.index)
                mat.loc[a1, a2] = cohen_kappa_score(
                    s1.loc[common], s2.loc[common], weights='quadratic'
                )
    weighted_mats[cat] = mat
    print(f"\n-- {cat.capitalize()} --")
    print(mat.to_string(float_format="%.3f"))

# Export CSVs
pd.DataFrame(fleiss_data, columns=['Story','Not Story']).to_csv('fleiss_counts.csv', index=False)
pd.Series({'fleiss_kappa': fleiss_val}).to_frame().to_csv('fleiss_kappa.csv')
binary_mat.to_csv('cohen_pairwise_binary.csv')
for cat, mat in weighted_mats.items():
    mat.to_csv(f'cohen_weighted_{cat}.csv')

# Visualizations
# 1. Fleiss Kappa
plt.figure()
plt.bar(['story_class'], [fleiss_val])
plt.ylabel("Fleiss' κ")
plt.title("Fleiss' Kappa for Story Classification")
plt.tight_layout()
plt.savefig('fleiss_kappa_story_class.png')
plt.close()

# 3. Pairwise Binary Heatmap
plt.figure()
plt.imshow(binary_mat.values)
plt.xticks(range(len(annotators)), annotators, rotation=90)
plt.yticks(range(len(annotators)), annotators)
plt.title("Pairwise Cohen's κ (binary story_class)")
plt.colorbar()
plt.tight_layout()
plt.savefig('pairwise_cohens_kappa_story_class.png')
plt.close()

# 4. Pairwise Weighted Heatmaps
for cat, mat in weighted_mats.items():
    plt.figure()
    plt.imshow(mat.values)
    plt.xticks(range(len(annotators)), annotators, rotation=90)
    plt.yticks(range(len(annotators)), annotators)
    plt.title(f"Pairwise Quadratic Weighted κ ({cat})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'pairwise_weighted_cohens_kappa_{cat}.png')
    plt.close()
