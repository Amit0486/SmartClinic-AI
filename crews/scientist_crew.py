from crewai import Agent, Task, Crew
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.data_tools import engineer_features, train_and_evaluate_models, generate_model_card
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
# Agent 5 - Model Training
model_agent = Agent(
    role="Machine Learning Engineer",
    goal="Train and evaluate predictive models for heart disease detection",
    backstory="""You are an ML engineer with expertise in clinical predictive
    modeling. You always compare multiple models and choose the best one
    based on ROC-AUC score.""",
    tools=[train_and_evaluate_models],
    verbose=True
)

model_task = Task(
    description="""
    Read outputs/features.csv and train two models:
    Logistic Regression and Random Forest.
    Compare them using accuracy, F1, and ROC-AUC.
    Save the best model to outputs/model.pkl
    Save the comparison report to outputs/evaluation_report.md
    """,
    expected_output="Evaluation report comparing both models with winner declared",
    agent=model_agent
)
# Agent 6 - Model Card
model_card_agent = Agent(
    role="Senior Medical AI Researcher",
    goal="Write a complete model card for the heart disease prediction model",
    backstory="""You are a medical AI researcher who documents AI models
    for clinical use. You always include ethical considerations and
    limitations. You write for both technical and non-technical audiences.""",
    tools=[generate_model_card],
    verbose=True
)

model_card_task = Task(
    description="""
    Read outputs/evaluation_report.md and write a complete model card.
    Include: model purpose, training data, performance metrics,
    limitations, and ethical considerations.
    Save to outputs/model_card.md
    """,
    expected_output="A complete model card saved to outputs/model_card.md",
    agent=model_card_agent
)
# Crew
scientist_crew = Crew(
    agents=[feature_agent, model_agent, model_card_agent],
    tasks=[feature_task, model_task, model_card_task],
    verbose=True
)


if __name__ == "__main__":
    result = scientist_crew.kickoff()
    print("\n=== RESULT ===")
    print(result)