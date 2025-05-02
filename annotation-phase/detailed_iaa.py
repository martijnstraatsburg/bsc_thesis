#!/usr/bin/env python3

# Name: detailed_iaa.py
# Author: Martijn Straatsburg
# Description: This script performs detailed inter-annotator agreement analysis.
# It calculates Fleiss' Kappa for story classification, ICC for numeric ratings, and pairwise Cohen's Kappa and correlations for multiple annotators.
# It also visualizes the results and identifies content items with high disagreement.

import json
from collections import defaultdict
import pandas as pd
import numpy as np
import pingouin as pg
from statsmodels.stats.inter_rater import fleiss_kappa
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

with open('names-annotations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

content_groups = defaultdict(list)
for entry in data:
    content_groups[entry['name']].append(entry)

fleiss_data = []
for name, entries in content_groups.items():
    story_counts = {'Story': 0, 'Not Story': 0}
    for entry in entries:
        if entry['story_class'] == 'Story':
            story_counts['Story'] += 1
        else:
            story_counts['Not Story'] += 1
    fleiss_data.append([story_counts['Story'], story_counts['Not Story']])

kappa = fleiss_kappa(fleiss_data)
print(f"Fleiss' Kappa for story_class: {kappa:.3f}")

def prepare_icc_data(entries, rating_key):
    icc_data = []
    for entry in entries:
        name = entry['name']
        annotator = entry['annotator']
        rating = entry[rating_key][0]['rating']
        icc_data.append({'target': name, 'rater': annotator, 'rating': rating})
    return pd.DataFrame(icc_data)

categories = ['curiosity', 'surprise', 'suspense']
results = {}

for category in categories:
    df = prepare_icc_data(data, category)
    icc_result = pg.intraclass_corr(data=df, targets='target', raters='rater', ratings='rating')
    icc_value = icc_result[icc_result['Type'] == 'ICC3k']['ICC'].values[0]
    results[category] = icc_value
    print(f"ICC(3,k) for {category}: {icc_value:.3f}")


# All unique annotators
annotators = sorted(list(set([entry['annotator'] for entry in data])))
print(f"\nAnnotators: {annotators}")

# DF with all annotations
structured_data = []
for entry in data:
    item = {
        'name': entry['name'],
        'annotator': entry['annotator'],
        'story_class': 1 if entry['story_class'] == 'Story' else 0
    }
    
    for category in categories:
        item[category] = entry[category][0]['rating']
        
    structured_data.append(item)

df_annotations = pd.DataFrame(structured_data)

# Cohen's Kappa for story_class
print("\n==== Cohen's Kappa for story_class ====")
kappa_matrix = np.zeros((len(annotators), len(annotators)))

for i, anno1 in enumerate(annotators):
    for j, anno2 in enumerate(annotators):
        if i == j:
            kappa_matrix[i, j] = 1.0
            continue

        df1 = df_annotations[df_annotations['annotator'] == anno1]
        df2 = df_annotations[df_annotations['annotator'] == anno2]
        merged = pd.merge(
            df1[['name', 'story_class']], 
            df2[['name', 'story_class']], 
            on='name', 
            suffixes=('_1', '_2')
        )
        
        if len(merged) > 0:
            kappa = cohen_kappa_score(
                merged['story_class_1'].values,
                merged['story_class_2'].values
            )
            kappa_matrix[i, j] = kappa
            print(f"{anno1} vs {anno2}: {kappa:.3f}")

# Pairwise correlation for ratings
correlation_results = {}

for category in categories:
    print(f"\n==== Pairwise correlations for {category} ====")
    pearson_matrix = np.zeros((len(annotators), len(annotators)))
    spearman_matrix = np.zeros((len(annotators), len(annotators)))
    
    for i, anno1 in enumerate(annotators):
        for j, anno2 in enumerate(annotators):
            if i == j:
                pearson_matrix[i, j] = 1.0
                spearman_matrix[i, j] = 1.0
                continue

            df1 = df_annotations[df_annotations['annotator'] == anno1]
            df2 = df_annotations[df_annotations['annotator'] == anno2]
            merged = pd.merge(
                df1[['name', category]], 
                df2[['name', category]], 
                on='name', 
                suffixes=('_1', '_2')
            )
            if len(merged) > 0:
                pearson, p_val_p = pearsonr(merged[f'{category}_1'], merged[f'{category}_2'])
                spearman, p_val_s = spearmanr(merged[f'{category}_1'], merged[f'{category}_2'])
                
                pearson_matrix[i, j] = pearson
                spearman_matrix[i, j] = spearman
                
                print(f"{anno1} vs {anno2}: Pearson r={pearson:.3f} (p={p_val_p:.3f}), Spearman rho={spearman:.3f} (p={p_val_s:.3f})")
    
    correlation_results[category] = {
        'pearson': pearson_matrix,
        'spearman': spearman_matrix
    }

# Simple visualisations of pairwise results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Pairwise Annotator Agreement', fontsize=16)

im0 = axes[0, 0].imshow(kappa_matrix, cmap='viridis', vmin=-1, vmax=1)
axes[0, 0].set_title("Cohen's Kappa for 'story_class'")
axes[0, 0].set_xticks(range(len(annotators)))
axes[0, 0].set_yticks(range(len(annotators)))
axes[0, 0].set_xticklabels(annotators)
axes[0, 0].set_yticklabels(annotators)
fig.colorbar(im0, ax=axes[0, 0])

category = categories[0]
im1 = axes[0, 1].imshow(correlation_results[category]['pearson'], cmap='viridis', vmin=-1, vmax=1)
axes[0, 1].set_title(f"Pearson r for '{category}'")
axes[0, 1].set_xticks(range(len(annotators)))
axes[0, 1].set_yticks(range(len(annotators)))
axes[0, 1].set_xticklabels(annotators)
axes[0, 1].set_yticklabels(annotators)
fig.colorbar(im1, ax=axes[0, 1])

category = categories[1]
im2 = axes[1, 0].imshow(correlation_results[category]['pearson'], cmap='viridis', vmin=-1, vmax=1)
axes[1, 0].set_title(f"Pearson r for '{category}'")
axes[1, 0].set_xticks(range(len(annotators)))
axes[1, 0].set_yticks(range(len(annotators)))
axes[1, 0].set_xticklabels(annotators)
axes[1, 0].set_yticklabels(annotators)
fig.colorbar(im2, ax=axes[1, 0])

category = categories[2]
im3 = axes[1, 1].imshow(correlation_results[category]['pearson'], cmap='viridis', vmin=-1, vmax=1)
axes[1, 1].set_title(f"Pearson r for '{category}'")
axes[1, 1].set_xticks(range(len(annotators)))
axes[1, 1].set_yticks(range(len(annotators)))
axes[1, 1].set_xticklabels(annotators)
axes[1, 1].set_yticklabels(annotators)
fig.colorbar(im3, ax=axes[1, 1])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('annotator_agreement.png')
plt.close()

# Annotator bias and rating patterns
print("\n==== Annotator Rating Patterns ====")
annotator_stats = pd.DataFrame(columns=['Annotator', 'Story%'] + categories)

for i, anno in enumerate(annotators):
    anno_data = df_annotations[df_annotations['annotator'] == anno]
    story_percent = (anno_data['story_class'].mean() * 100)
    category_means = [anno_data[cat].mean() for cat in categories]
    annotator_stats.loc[i] = [anno, story_percent] + category_means

print(annotator_stats.to_string(index=False, float_format="%.2f"))

# Simple distributions visualisations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Rating Distributions by Annotator', fontsize=16)

axes[0, 0].bar(annotators, annotator_stats['Story%'])
axes[0, 0].set_title("Percentage of 'Story' Classifications")
axes[0, 0].set_ylabel("Percentage")
axes[0, 0].set_ylim(0, 100)

for i, category in enumerate(categories):
    axes[(i+1)//2, (i+1)%2].bar(annotators, annotator_stats[category])
    axes[(i+1)//2, (i+1)%2].set_title(f"Mean {category.capitalize()} Ratings")
    axes[(i+1)//2, (i+1)%2].set_ylabel("Mean Rating")
    axes[(i+1)//2, (i+1)%2].set_ylim(1, 5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('annotator_distributions.png')
plt.close()

# Highlighting high disagreement cases and variance
print("\n==== Content Items with High Disagreement ====")
content_variance = defaultdict(dict)
for name in content_groups:
    entries = content_groups[name]
    story_values = [1 if entry['story_class'] == 'Story' else 0 for entry in entries]
    content_variance[name]['story_class'] = np.var(story_values)
    for category in categories:
        ratings = [entry[category][0]['rating'] for entry in entries]
        content_variance[name][category] = np.var(ratings)

df_variance = pd.DataFrame.from_dict(content_variance, orient='index')
df_variance['content'] = df_variance.index
df_variance = df_variance.sort_values('story_class', ascending=False)

print("\nTop disagreements for story_class:")
print(df_variance[['content', 'story_class']].head(5))

for category in categories:
    print(f"\nTop disagreements for {category}:")
    temp_df = df_variance.sort_values(category, ascending=False)
    print(temp_df[['content', category]].head(5))

df_variance.to_csv('content_disagreement.csv')
