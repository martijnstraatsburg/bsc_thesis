import json

with open('gs-test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    item.pop('story_class', None)
    item.pop('suspense', None)
    item.pop('curiosity', None)
    item.pop('surprise', None)

with open('mod-gs-test.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Keys removed successfully!")