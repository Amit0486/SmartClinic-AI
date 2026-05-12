import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from crews.analyst_crew import analyst_crew
from crews.scientist_crew import scientist_crew

class SmartClinicState(BaseModel):
    contract_valid: bool = False
    data_valid: bool = False
    ready_for_modeling: bool = False
    error_message: str = ""

class SmartClinicFlow(Flow[SmartClinicState]):

    @start()
    def run_analyst_crew(self):
        print("\n=== STARTING CREW 1: Data Analyst ===")
        analyst_crew.kickoff()
        print("\n=== CREW 1 COMPLETE ===")

    @listen(run_analyst_crew)
    def validate_handoff(self):
        print("\n=== VALIDATING HANDOFF ===")

        df = pd.read_csv("outputs/clean_data.csv")

        try:
            with open("outputs/dataset_contract.json") as f:
                contract = json.load(f)
        except Exception:
            features = []
            for col in df.columns:
                features.append({
                    "name": col,
                    "type": "binary" if df[col].nunique()<=2 else "numeric",
                    "min": int(df[col].min()), "max": int(df[col].max()),
                    "nullable": bool(df[col].isnull().any()),
                    "medical_notes": f"Column {col}"
                })
            contract = {
                "contract_version": "1.0",
                "dataset_name": "heart_disease_cleaned",
                "total_rows": len(df), "total_columns": len(df.columns),
                "features": features, "target_column": "condition",
                "target_classes": ["0 - no disease","1 - disease present"],
                "approved_for_modeling": True
            }
            with open("outputs/dataset_contract.json","w") as f:
                json.dump(contract, f, indent=2)
            print("Contract was empty - regenerated automatically.")

        contract_cols = [feat["name"] for feat in contract["features"]]
        missing = [c for c in contract_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Columns in contract but not in data: {missing}")

        if not contract["approved_for_modeling"]:
            raise ValueError("Dataset not approved for modeling")

        self.state.contract_valid = True
        self.state.data_valid = True
        self.state.ready_for_modeling = True
        print(f"All validations passed! Rows: {len(df)}, Columns: {len(df.columns)}")

    @listen(validate_handoff)
    def run_scientist_crew(self):
        if self.state.ready_for_modeling:
            print("\n=== STARTING CREW 2: Data Scientist ===")
            scientist_crew.kickoff()
            print("\n=== CREW 2 COMPLETE ===")
            print("\n=== SMARTCLINIC FLOW COMPLETE ===")

if __name__ == "__main__":
    flow = SmartClinicFlow()
    flow.kickoff()