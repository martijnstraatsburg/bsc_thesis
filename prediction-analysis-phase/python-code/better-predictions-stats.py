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

    level_counters = {
        'level_suspense': Counter(),
        'level_curiosity': Counter(),
        'level_surprise': Counter(),
    }

    binary_counters = {
        'binary_suspense': Counter(),
        'binary_curiosity': Counter(),
        'binary_surprise': Counter(),
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

        for key, counter in perception_counters.items():
            val = entry.get(key)
            if isinstance(val, int) and 1 <= val <= 5:
                counter[val] += 1
            else:
                counter['invalid'] += 1

        for key, counter in level_counters.items():
            lvl = entry.get(key)
            if isinstance(lvl, str) and lvl:
                counter[lvl] += 1
            else:
                counter['invalid'] += 1

        for key, counter in binary_counters.items():
            bio = entry.get(key)
            if isinstance(bio, str) and bio:
                counter[bio] += 1
            else:
                counter['invalid'] += 1

    print("=== Post vs Comment Counts ===")
    for kind, cnt in kind_counter.items():
        print(f"{kind}: {cnt}")
    print("\n=== Story Class Counts ===")
    for cls, cnt in story_class_counter.items():
        print(f"{cls}: {cnt}")
    print("\n=== Numeric Readers' Perception Counts ===")
    for key, counter in perception_counters.items():
        print(f"\n-- {key.capitalize()} --")
        for rating in sorted([r for r in counter if isinstance(r, int)]):
            print(f"{rating}: {counter[rating]}")
        if 'invalid' in counter:
            print(f"invalid: {counter['invalid']}")
    print("\n=== Level-Based Readers' Perception Counts ===")
    for key, counter in level_counters.items():
        print(f"\n-- {key} --")
        for level, cnt in counter.items():
            print(f"{level}: {cnt}")
    print("\n=== Binary Readers' Perception Counts ===")
    for key, counter in binary_counters.items():
        print(f"\n-- {key} --")
        for val, cnt in counter.items():
            print(f"{val}: {cnt}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Count posts/comments, story classes, and various reader perceptions in a JSON dataset.'
    )
    parser.add_argument('json_file', help='Path to the JSON file')
    args = parser.parse_args()
    main(args.json_file)
