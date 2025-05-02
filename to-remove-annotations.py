import json

# Load the original JSON data
with open('gs-train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Remove unwanted keys from each entry
for item in data:
    item.pop('story_class', None)
    item.pop('suspense', None)
    item.pop('curiosity', None)
    item.pop('surprise', None)

# Save modified data to a new file (or overwrite original)
with open('mod-gs-train.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Keys removed successfully!")