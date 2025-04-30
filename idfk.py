import json

# Load the original JSON data
with open('small_gold_standard.json', 'r') as f:
    data = json.load(f)

# Remove unwanted keys from each entry
for item in data:
    item.pop('story_class', None)
    item.pop('suspense', None)
    item.pop('curiosity', None)
    item.pop('surprise', None)

# Save modified data to a new file (or overwrite original)
with open('modified_gold_standard.json', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Keys removed successfully!")