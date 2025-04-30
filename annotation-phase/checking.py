import json

# Load the JSON file with UTF-8 encoding
with open('anno.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Count total entries
total_entries = len(data)
print(f"Total entries in the dataset: {total_entries}")

# Find entries missing the 'story_class' key
missing_story_class = []
for i, entry in enumerate(data):
    if 'story_class' not in entry:
        # Collect identifying information
        entry_info = {
            'index': i,
            'id': entry.get('id', 'No ID'),
            'name': entry.get('name', 'No Name'),
            'annotator': entry.get('annotator', 'No Annotator')
        }
        missing_story_class.append(entry_info)

# Print results
if missing_story_class:
    print(f"\nFound {len(missing_story_class)} entries missing 'story_class':")
    for entry in missing_story_class:
        print(f"  Index: {entry['index']}, ID: {entry['id']}, Name: {entry['name']}, Annotator: {entry['annotator']}")
    
    # Print the keys available in the first missing entry
    first_missing_idx = missing_story_class[0]['index']
    print("\nKeys available in the first missing entry:")
    for key in sorted(data[first_missing_idx].keys()):
        print(f"  - {key}")
    
    # Check if there might be an alternative field that contains story classification
    print("\nPotential alternative fields in the first missing entry:")
    for key, value in data[first_missing_idx].items():
        if isinstance(value, str) and ('story' in key.lower() or 'class' in key.lower() or 'type' in key.lower()):
            print(f"  - {key}: {value}")
        elif isinstance(value, dict) and any(k for k in value.keys() if 'story' in k.lower() or 'class' in k.lower()):
            print(f"  - {key}: {value}")
        
    # Calculate percentage of missing entries
    missing_percentage = (len(missing_story_class) / total_entries) * 100
    print(f"\n{missing_percentage:.1f}% of entries are missing the 'story_class' key")
else:
    print("\nAll entries have the 'story_class' key.")

# Check if there are any entries that DO have the 'story_class' key
has_story_class = [i for i, entry in enumerate(data) if 'story_class' in entry]
if has_story_class:
    print(f"\nFound {len(has_story_class)} entries WITH 'story_class'")
    first_with_idx = has_story_class[0]
    print(f"Example 'story_class' value from index {first_with_idx}: {data[first_with_idx]['story_class']}")
    
    # Show an example of what a complete entry looks like
    print("\nExample of a complete entry with 'story_class':")
    print(json.dumps(data[first_with_idx], indent=2))
