import csv
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a CSV file of Reddit-style comments to a JSON array of objects."
    )
    parser.add_argument(
        "input_csv", 
        help="Path to the input CSV file."
    )
    parser.add_argument(
        "output_json", 
        help="Path to write the output JSON file."
    )
    return parser.parse_args()


def convert_csv_to_json(input_csv: str, output_json: str) -> None:
    """
    Reads the CSV file at input_csv and writes a JSON file to output_json.
    The CSV is expected to have headers: 
        name, author, created_utc, body, persuasion_success, parent_id
    The JSON output will be a list of objects with the same keys, 
    converting numeric fields to integers.
    """
    data = []
    with open(input_csv, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert types
            try:
                row['created_utc'] = int(row['created_utc'])
            except (ValueError, KeyError):
                # Leave as string if conversion fails or missing
                pass

            try:
                row['persuasion_success'] = int(row['persuasion_success'])
            except (ValueError, KeyError):
                # Leave as string if conversion fails or missing
                pass

            # Ensure all keys exist
            obj = {
                'name': row.get('name', ''),
                'author': row.get('author', ''),
                'created_utc': row.get('created_utc', None),
                'body': row.get('body', ''),
                'parent_id': row.get('parent_id', ''),
                'persuasion_success': row.get('persuasion_success', None)
            }

            data.append(obj)

    # Write JSON output
    with open(output_json, mode='w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    args = parse_args()
    convert_csv_to_json(args.input_csv, args.output_json)
    print(f"Converted '{args.input_csv}' to '{args.output_json}'.")
