import pandas as pd
import tempfile
import os

from recommendme.data_loader import convert_to_json  # adjust if your import path differs

def test_convert_to_json_basic():
    df = pd.DataFrame({
        "Title": ["Q1"],
        "desc": ["What is 2+2?"],
        "Tags": ["math, easy"],
        "Difficulty": ["Easy"]
    })

    field_map = {
        "title": "Title",
        "desc": "desc",
        "tags": "Tags",
        "difficulty": "Difficulty"
    }

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w', encoding='utf-8') as tmp:
        df.to_csv(tmp.name, index=False)
        temp_path = tmp.name

    try:
        json_data = convert_to_json(temp_path, field_map=field_map)
        assert isinstance(json_data, list)
        assert len(json_data) == 1
        assert json_data[0]["title"] == "Q1"
        assert json_data[0]["desc"] == "What is 2+2?"
    finally:
        os.remove(temp_path)
