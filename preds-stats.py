import json
import argparse
from collections import Counter

def main(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    kind_counter = Counter()
    story_class_counter = Counter()
    perception_counters = {
        'suspense': Counter(),
        'curiosity': Counter(),
        'surprise': Counter(),
    }
    for entry in data:
        name = entry.get('name', '')
        if name.startswith('t3_'):
            kind_counter['post'] += 1
        elif name.startswith('t1_'):
            kind_counter['comment'] += 1
        else:
            kind_counter['other'] += 1
        story_class = entry.get('story_class', 'Unknown')
        story_class_counter[story_class] += 1

        for key in perception_counters:
            val = entry.get(key)
            if isinstance(val, int) and 1 <= val <= 5:
                perception_counters[key][val] += 1
            else:
                perception_counters[key]['invalid'] += 1
    print("=== Post vs Comment Counts ===")
    for kind, cnt in kind_counter.items():
        print(f"{kind}: {cnt}")
    print("\n=== Story Class Counts ===")
    for cls, cnt in story_class_counter.items():
        print(f"{cls}: {cnt}")
    print("\n=== Readers' Perception Counts ===")
    for key, counter in perception_counters.items():
        print(f"\n-- {key.capitalize()} --")
        for rating in sorted([r for r in counter if isinstance(r, int)]):
            print(f"{rating}: {counter[rating]}")
        if 'invalid' in counter:
            print(f"invalid: {counter['invalid']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Count posts/comments, story classes, and readers\' perceptions in a JSON dataset.'
    )
    parser.add_argument('json_file', help='Path to the JSON file (e.g., 1000-discussions-predictions.json)')
    args = parser.parse_args()
    main(args.json_file)
