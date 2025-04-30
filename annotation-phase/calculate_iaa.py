import json
from collections import defaultdict
import pandas as pd
import pingouin as pg
from statsmodels.stats.inter_rater import fleiss_kappa

with open('anno.json', 'r', encoding='utf-8') as f:
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
print(f"Fleiss' Kappa: {kappa:.3f}")

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
