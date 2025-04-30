import json
from collections import defaultdict
import pandas as pd
import numpy as np
import pingouin as pg
from statsmodels.stats.inter_rater import fleiss_kappa
import krippendorff

# Load the data
with open('anno.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Group entries by content name
content_groups = defaultdict(list)
for entry in data:
    content_groups[entry['name']].append(entry)

# Calculate Fleiss' Kappa for story classification
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
print(f"Fleiss' Kappa: {kappa:.3f}")

# Function to prepare data for ICC calculation
def prepare_icc_data(entries, rating_key):
    icc_data = []
    for entry in entries:
        name = entry['name']
        annotator = entry['annotator']
        rating = entry[rating_key][0]['rating']
        icc_data.append({'target': name, 'rater': annotator, 'rating': rating})
    return pd.DataFrame(icc_data)

# Function to prepare data for Krippendorff's alpha
def prepare_krippendorff_data(entries, rating_key):
    # Create a dictionary to map annotators and targets to indices
    annotators = list(set(entry['annotator'] for entry in entries))
    targets = list(set(entry['name'] for entry in entries))
    
    # Create a matrix with shape (num_annotators, num_targets)
    reliability_data = np.empty((len(annotators), len(targets)))
    reliability_data[:] = np.nan  # Fill with NaN for missing values
    
    for entry in entries:
        annotator_idx = annotators.index(entry['annotator'])
        target_idx = targets.index(entry['name'])
        rating = entry[rating_key][0]['rating']
        reliability_data[annotator_idx, target_idx] = rating
    
    return reliability_data

# Calculate ICC and Krippendorff's alpha for each category
categories = ['curiosity', 'surprise', 'suspense']
results = {}

for category in categories:
    # Calculate ICC
    df = prepare_icc_data(data, category)
    icc_result = pg.intraclass_corr(data=df, targets='target', raters='rater', ratings='rating')
    icc_value = icc_result[icc_result['Type'] == 'ICC3k']['ICC'].values[0]
    
    # Calculate Krippendorff's alpha
    k_data = prepare_krippendorff_data(data, category)
    k_alpha = krippendorff.alpha(k_data, level_of_measurement='interval')
    
    results[category] = {'ICC': icc_value, 'Krippendorff': k_alpha}
    print(f"Category: {category}")
    print(f"  ICC(3,k): {icc_value:.3f}")
    print(f"  Krippendorff's Alpha: {k_alpha:.3f}")

# Calculate Krippendorff's alpha for story classification
# Convert story classifications to a numerical format
story_data = []
annotators = list(set(entry['annotator'] for entry in data))
targets = list(set(entry['name'] for entry in data))

story_matrix = np.empty((len(annotators), len(targets)))
story_matrix[:] = np.nan

for entry in data:
    annotator_idx = annotators.index(entry['annotator'])
    target_idx = targets.index(entry['name'])
    # Map 'Story' to 1 and 'Not Story' to 0
    story_value = 1 if entry['story_class'] == 'Story' else 0
    story_matrix[annotator_idx, target_idx] = story_value

story_alpha = krippendorff.alpha(story_matrix, level_of_measurement='nominal')
print(f"Krippendorff's Alpha for story classification: {story_alpha:.3f}")

# Identify problematic content items for story classification
print("\n--- Identifying problematic content for story classification ---")
disagreement_scores = {}

for name, entries in content_groups.items():
    story_votes = sum(1 for entry in entries if entry['story_class'] == 'Story')
    not_story_votes = len(entries) - story_votes
    
    # Calculate disagreement score (0 if perfect agreement, 0.5 if maximum disagreement)
    if len(entries) > 0:
        disagreement = min(story_votes, not_story_votes) / len(entries)
        disagreement_scores[name] = disagreement
    
        if disagreement > 0.25:  # Threshold for highlighting problematic items
            print(f"Content '{name}': {story_votes} votes for 'Story', {not_story_votes} votes for 'Not Story'")

# Print the most problematic items (highest disagreement)
print("\nTop 5 most problematic content items:")
top_disagreements = sorted(disagreement_scores.items(), key=lambda x: x[1], reverse=True)[:5]
for name, score in top_disagreements:
    print(f"Content '{name}': Disagreement score {score:.2f}")

# Similarly, identify content with disagreement on ratings
print("\n--- Identifying content with high rating variability ---")
for category in categories:
    print(f"\nCategory: {category}")
    rating_variance = {}
    
    for name, entries in content_groups.items():
        ratings = [entry[category][0]['rating'] for entry in entries if category in entry]
        if len(ratings) > 1:
            variance = np.var(ratings)
            rating_variance[name] = variance
    
    # Print top 3 content items with highest variance in ratings
    top_variances = sorted(rating_variance.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, var in top_variances:
        print(f"Content '{name}': Rating variance {var:.2f}")
