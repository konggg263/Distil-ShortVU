import os
import json
import pandas as pd
def json_to_excel(json_path):
    """Convert a JSON file (list of dicts) to an Excel file in the same folder.

    The function will flatten nested dictionaries (pandas.json_normalize) so
    fields like 'aesthetic_score.aesthetic' and 'pass2_sources.*' become columns.
    """
    if not os.path.exists(json_path):
        print(f"JSON file not found: {json_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # If the JSON is a dict with a top-level list under some key, try to detect it
        if isinstance(data, dict) and len(data) == 1:
            first_key = list(data.keys())[0]
            if isinstance(data[first_key], list):
                data = data[first_key]

        # Ensure it's a list of records
        if not isinstance(data, list):
            print(f"Unexpected JSON structure, expected a list of records: {type(data)}")
            return None

        # Flatten nested dicts into columns
        df = pd.json_normalize(data)

        # Choose excel path next to json file
        excel_path = os.path.splitext(json_path)[0] + '.xlsx'
        df.to_excel(excel_path, index=False)
        print(f"Saved Excel: {excel_path}")
        return excel_path
    except Exception as e:
        print(f"Error converting JSON to Excel: {e}")
        return None
    

if __name__ == "__main__":
    json_file = "/Users/macco/Downloads/khoaluanvjp/taolam/data/train_processed.json"  # Example path
    json_to_excel(json_file)