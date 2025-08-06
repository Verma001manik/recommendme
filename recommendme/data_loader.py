import pandas as pd
import json
import os

def convert_to_json(path, field_map=None, default_fields=None):
    """
    Converts a CSV file to a standardized JSON format.
    - `field_map`: maps logical field names to CSV column names.
    - `default_fields`: provides fallback values for missing fields.

    returns : json 
    Mostly title,desc are needed for something very simple 
    """
    df = pd.read_csv(path)
    data = []

    if field_map is None:
        raise ValueError("You must pass a field_map to tell the loader how to map fields")

    for i, row in df.iterrows():
        def get_field(logical_key):
            csv_col = field_map.get(logical_key, "")
            if not csv_col or csv_col not in row:
                return default_fields.get(logical_key, "") if default_fields else ""
            return str(row[csv_col]) if pd.notna(row[csv_col]) else ""

        tags_list = []
        if 'tags' in field_map:
            tags_raw = get_field("tags")
            tags_list = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]

        data.append({
            "id": i,
            "title": get_field("title"),
            "desc": get_field("desc"),
            "tags": tags_list,
            "difficulty": get_field("difficulty")
        })

    os.makedirs("data", exist_ok=True)
    filename = os.path.basename(path)
    output_path = f"data/{filename}.json"

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Converted {path} → {output_path}")
