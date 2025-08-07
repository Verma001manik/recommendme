import pandas as pd 
import torch 
import gc 

def drop_na_rows(df, column_name: str) :
    df_clean = df.dropna(subset=[column_name])
    return df_clean

def convert_df_to_csv(df, path):
    return df.to_csv(path, index=False)


def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gc.collect()