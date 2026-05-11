from crewai import Agent, Task, Crew
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.data_tools import load_and_clean_data, analyze_and_visualize_data
from dotenv import load_dotenv

load_dotenv()

# ---- סוכן 1 ----
data_ingestion_agent = Agent(
    role="Data Ingestion Specialist",
    goal="Load, validate and clean the heart disease dataset",
    backstory="""You are a meticulous data engineer with 10 years
    of experience in medical data quality. You never pass dirty
    data to the next stage.""",
    tools=[load_and_clean_data],
    verbose=True
)

# ---- משימה 1 ----
ingestion_task = Task(
    description="""
    Load the heart disease dataset from data/heart_cleveland_upload.csv
    Clean it and save the result to outputs/clean_data.csv
    Report exactly what you found and what you fixed.
    """,
    expected_output="A summary of the dataset: rows, columns, issues found and fixed",
    agent=data_ingestion_agent
)

# ---- סוכן 2 ----
eda_agent = Agent(
    role="Senior Medical Data Scientist",
    goal="Perform exploratory data analysis on the clean heart disease dataset",
    backstory="""You are a data scientist with 15 years of experience
    in cardiovascular research. You turn raw data into clear business
    insights and always support your findings with charts.""",
    tools=[analyze_and_visualize_data],
    verbose=True
)

# ---- משימה 2 ----
eda_task = Task(
    description="""
    Read outputs/clean_data.csv and perform full exploratory data analysis.
    Generate charts for age, cholesterol, and feature correlations.
    Write a business summary of the key findings.
    Save results to outputs/insights.md
    """,
    expected_output="A markdown summary with key statistics and chart locations",
    agent=eda_agent
)

# ---- הקרוז ----
analyst_crew = Crew(
    agents=[data_ingestion_agent, eda_agent],
    tasks=[ingestion_task, eda_task],
    verbose=True
)

# ---- הרצה ----
if __name__ == "__main__":
    result = analyst_crew.kickoff()
    print("\n=== RESULT ===")
    print(result)