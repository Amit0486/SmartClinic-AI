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
    import matplotlib.pyplot as plt
import seaborn as sns

@tool("analyze_and_visualize_data")
def analyze_and_visualize_data(filepath: str) -> str:
    """
    Reads clean_data.csv, generates 3 charts, and writes
    a business summary. Use this for exploratory data analysis.
    """
    # קריאת הדאטא
    df = pd.read_csv(filepath)

    # יצירת תיקיית outputs אם לא קיימת
    os.makedirs("outputs", exist_ok=True)

    # גרף 1 — התפלגות גילאים
    plt.figure(figsize=(8, 4))
    df['age'].hist(bins=20, color='steelblue', edgecolor='white')
    plt.title('Age Distribution of Patients')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('outputs/chart_age.png')
    plt.close()

    # גרף 2 — התפלגות כולסטרול
    plt.figure(figsize=(8, 4))
    df['chol'].hist(bins=20, color='tomato', edgecolor='white')
    plt.title('Cholesterol Distribution')
    plt.xlabel('Cholesterol (mg/dL)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('outputs/chart_cholesterol.png')
    plt.close()

    # גרף 3 — heatmap של קורלציות
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True),
                annot=True, fmt='.2f',
                cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('outputs/chart_correlation.png')
    plt.close()

    # סיכום עסקי
    high_risk = df[df['condition'] == 1]
    low_risk  = df[df['condition'] == 0]

    summary = f"""
## EDA Summary — Heart Disease Dataset

**Dataset:** {len(df)} patients, {len(df.columns)} features

**Risk distribution:**
- High risk (condition=1): {len(high_risk)} patients ({len(high_risk)/len(df)*100:.1f}%)
- Low risk  (condition=0): {len(low_risk)}  patients ({len(low_risk)/len(df)*100:.1f}%)

**Age:** mean={df['age'].mean():.1f}, min={df['age'].min()}, max={df['age'].max()}
**Cholesterol:** mean={df['chol'].mean():.1f}, min={df['chol'].min()}, max={df['chol'].max()}
**Max Heart Rate:** mean={df['thalach'].mean():.1f}

**Charts saved:** chart_age.png, chart_cholesterol.png, chart_correlation.png
"""

    # שמירת הסיכום
    with open('outputs/insights.md', 'w') as f:
        f.write(summary)

    return summary
    import json

@tool("generate_dataset_contract")
def generate_dataset_contract(filepath: str) -> str:
    """
    Reads clean_data.csv and generates a dataset_contract.json.
    Use this to create a formal schema definition for Crew 2.
    """
    df = pd.read_csv(filepath)
    os.makedirs("outputs", exist_ok=True)

    features = []
    for col in df.columns:
        features.append({
            "name": col,
            "type": "binary" if df[col].nunique() <= 2 else "numeric",
            "min": int(df[col].min()),
            "max": int(df[col].max()),
            "nullable": bool(df[col].isnull().any()),
            "medical_notes": f"Column {col} - range [{int(df[col].min())}, {int(df[col].max())}]"
        })

    contract = {
        "contract_version": "1.0",
        "dataset_name": "heart_disease_cleaned",
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "features": features,
        "target_column": "condition",
        "target_classes": ["0 - no disease", "1 - disease present"],
        "approved_for_modeling": True
    }

    with open("outputs/dataset_contract.json", "w") as f:
        json.dump(contract, f, indent=2)

    return f"Contract saved to outputs/dataset_contract.json\nRows: {len(df)}\nColumns: {len(df.columns)}\nApproved for modeling: True"

@tool("engineer_features")
def engineer_features(filepath: str) -> str:
    """
    Reads clean_data.csv and creates new features.
    Saves features.csv to outputs/.
    Use this for feature engineering before model training.
    """
    df = pd.read_csv(filepath)
    os.makedirs("outputs", exist_ok=True)

    # feature 1 — קבוצת גיל
    df['age_group'] = pd.cut(df['age'],
                              bins=[0, 40, 55, 100],
                              labels=['young', 'middle', 'senior'])
    df['age_group'] = df['age_group'].astype(str)

    # feature 2 — קטגוריית לחץ דם
    df['bp_category'] = pd.cut(df['trestbps'],
                                bins=[0, 120, 140, 999],
                                labels=['normal', 'elevated', 'high'])
    df['bp_category'] = df['bp_category'].astype(str)

    # feature 3 — סיכון כולסטרול
    df['chol_risk'] = (df['chol'] > 240).astype(int)

    # feature 4 — קצב לב מקסימלי נמוך
    df['low_max_hr'] = (df['thalach'] < 140).astype(int)

    # שמירה
    df.to_csv("outputs/features.csv", index=False)

    return (
        f"Feature engineering complete.\n"
        f"New features added: age_group, bp_category, chol_risk, low_max_hr\n"
        f"Final shape: {df.shape}\n"
        f"Saved to: outputs/features.csv"
    )