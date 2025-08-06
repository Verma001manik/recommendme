import pandas as pd
import json
import os

def convert_to_json(path, field_map=None, default_fields=None, list_fields=None):
    """
    Converts a CSV file to a standardized JSON format.

    - `field_map`: dict mapping logical keys to CSV column names.
    - `default_fields`: dict of fallback values for missing fields.
    - `list_fields`: set of field names (from logical keys) that should be treated as comma-separated lists.

    Returns:
        List of JSON records (only when testing).
    """
    df = pd.read_csv(path)
    data = []

    if field_map is None:
        raise ValueError("You must pass a field_map to tell the loader how to map fields")

    list_fields = set(list_fields or [])  # default to empty set

    for i, row in df.iterrows():
        item = {"id": i}
        # print(f"item: {item}")
        for logical_key, csv_col in field_map.items():
            # print(f"row : {row}")
            # print(f"logicalkey: {logical_key}, csvcol: {csv_col}")
            val = ""
            if csv_col in row and pd.notna(row[csv_col]) :
                # print(f"csvcol:{csv_col} in row")
                val = str(row[csv_col])
            elif default_fields and logical_key in default_fields:
                val = default_fields[logical_key]

            # Handle list fields (e.g., tags, genres)
            if logical_key in list_fields:
                item[logical_key] = [v.strip() for v in val.split(",") if v.strip()]
            else:
                item[logical_key] = val
            # print(item)
            
         
        data.append(item)

    os.makedirs("data", exist_ok=True)
    filename = os.path.splitext(os.path.basename(path))[0]
    output_path = f"data/{filename}.json"

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Converted {path} → {output_path}")
    # return data only for testing
    return data

