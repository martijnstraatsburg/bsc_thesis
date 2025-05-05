import json
import argparse
import sys
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Flatten a limited number of discussions into one big list from Reddit-style JSON.")
    parser.add_argument(
        "input_json", help="Path to the input JSON file (array of items)."
    )
    parser.add_argument(
        "output_json", help="Path to write the flattened output JSON file.")
    parser.add_argument(
        "--num-discussions", type=int, required=True,
        help="Maximum number of top-level discussions (posts) to include."
    )
    parser.add_argument(
        "--stream", action='store_true',
        help="Use streaming parser (requires ijson) for large files.")
    return parser.parse_args()


def load_items(filepath, use_stream=False):
    if use_stream:
        try:
            import ijson
        except ImportError:
            sys.exit("ijson not installed. Install with: pip install ijson")
        with open(filepath, 'r', encoding='utf-8') as f:
            for item in ijson.items(f, 'item'):
                yield item
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                yield item


def collect_flat(item_id, items, children):
    flat = [items[item_id]]
    for child_id in children.get(item_id, []):
        flat.extend(collect_flat(child_id, items, children))
    return flat


def limit_flat_all(input_json, output_json, num_discussions, use_stream=False):
    items = {}
    children = defaultdict(list)

    # Load all items
    for item in load_items(input_json, use_stream):
        item_id = item.get('name')
        if item_id:
            items[item_id] = item

    # Build parent->children map
    for item in items.values():
        parent_id = item.get('parent_id')
        if parent_id in items:
            children[parent_id].append(item['name'])

    # Collect flat items for up to num_discussions
    output = []
    count = 0
    for item_id in items:
        if item_id.startswith('t3_'):
            if count >= num_discussions:
                break
            output.extend(collect_flat(item_id, items, children))
            count += 1

    # Write one big list
    with open(output_json, 'w', encoding='utf-8') as out_f:
        json.dump(output, out_f, ensure_ascii=False, indent=2)

    print(f"Wrote {count} discussions (flattened) into one list to {output_json}")


if __name__ == '__main__':
    args = parse_args()
    limit_flat_all(
        args.input_json,
        args.output_json,
        args.num_discussions,
        use_stream=args.stream
    )