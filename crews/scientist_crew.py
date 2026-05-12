from crewai import Agent, Task, Crew
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.data_tools import engineer_features
from dotenv import load_dotenv

load_dotenv()

# Agent 4 - Feature Engineering
feature_agent = Agent(
    role="Feature Engineering Expert",
    goal="Create new meaningful features from the clean heart disease dataset",
    backstory="""You are a machine learning engineer with deep expertise
    in clinical data. You know which features are most predictive for
    cardiovascular disease and how to engineer them from raw data.""",
    tools=[engineer_features],
    verbose=True
)

feature_task = Task(
    description="""
    Read outputs/clean_data.csv and perform feature engineering.
    Create new features: age_group, bp_category, chol_risk, low_max_hr.
    Save the enriched dataset to outputs/features.csv.
    Report what features were added and the final dataset shape.
    """,
    expected_output="Summary of new features added and path to features.csv",
    agent=feature_agent
)

# Crew
scientist_crew = Crew(
    agents=[feature_agent],
    tasks=[feature_task],
    verbose=True
)

if __name__ == "__main__":
    result = scientist_crew.kickoff()
    print("\n=== RESULT ===")
    print(result)