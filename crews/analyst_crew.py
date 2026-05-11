from crewai import Agent, Task, Crew
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.data_tools import load_and_clean_data
from dotenv import load_dotenv

load_dotenv()

# ---- הסוכן ----
data_ingestion_agent = Agent(
    role="Data Ingestion Specialist",
    goal="Load, validate and clean the heart disease dataset",
    backstory="""You are a meticulous data engineer with 10 years 
    of experience in medical data quality. You never pass dirty 
    data to the next stage.""",
    tools=[load_and_clean_data],
    verbose=True
)

# ---- המשימה ----
ingestion_task = Task(
    description="""
    Load the heart disease dataset from data/heart_cleveland_upload.csv
    Clean it and save the result to outputs/clean_data.csv
    Report exactly what you found and what you fixed.
    """,
    expected_output="A summary of the dataset: rows, columns, issues found and fixed",
    agent=data_ingestion_agent
)

# ---- הקרוז ----
analyst_crew = Crew(
    agents=[data_ingestion_agent],
    tasks=[ingestion_task],
    verbose=True
)

# ---- הרצה ----
if __name__ == "__main__":
    result = analyst_crew.kickoff()
    print("\n=== RESULT ===")
    print(result)