import pandas as pd
import os
from crewai.tools import tool

@tool("load_and_clean_data")
def load_and_clean_data(filepath: str) -> str:
    """
    Loads a CSV dataset, cleans it, and saves clean_data.csv.
    Use this tool to ingest and clean the heart disease dataset.
    """
    # טעינת הדאטא
    df = pd.read_csv(filepath)
    original_rows = len(df)

    # הסרת שורות כפולות
    df.drop_duplicates(inplace=True)
    duplicates_removed = original_rows - len(df)

    # טיפול בערכים חסרים
    df.fillna(df.median(numeric_only=True), inplace=True)

    # שמירת הדאטא הנקי
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/clean_data.csv", index=False)

    return (
        f"Dataset loaded successfully.\n"
        f"Original rows: {original_rows}\n"
        f"Duplicates removed: {duplicates_removed}\n"
        f"Final shape: {df.shape}\n"
        f"Columns: {list(df.columns)}\n"
        f"Saved to: outputs/clean_data.csv"
    )