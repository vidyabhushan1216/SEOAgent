import os
import io
import logging
import concurrent.futures
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentExecutor
import streamlit as st

# Load the environment variables from .env (for local development)
load_dotenv()

# Use Streamlit's secrets management to retrieve the OpenAI API key during deployment
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

# Check if the API key is present
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found. Ensure it is set in Streamlit's Secrets or in the .env file.")

# Initialize the OpenAI LLM
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo",
    temperature=0
)

# Define custom Agents
class SEOAgent:
    def __init__(self, role, goal, llm):
        self.role = role
        self.goal = goal
        self.llm = llm

    def run(self, inputs):
        prompt = f"You are an expert {self.role}. {self.goal.format(**inputs)}"
        response = self.llm.generate([prompt])  # Updated to use generate() instead of generate_responses()
        return response[0].text  # Ensure this returns the text of the response

# Define tasks as simple functions
def run_planner(topic):
    planner = SEOAgent(
        role="Content Planner",
        goal="Plan engaging and factually accurate content on the topic: {topic}.",
        llm=llm
    )
    return planner.run({"topic": topic})

def run_writer(topic):
    writer = SEOAgent(
        role="Content Writer",
        goal="Write an insightful opinion piece on the topic: {topic}.",
        llm=llm
    )
    return writer.run({"topic": topic})

def run_editor(topic):
    editor = SEOAgent(
        role="Editor",
        goal="Edit the blog post to align with journalistic standards and voice alignment for {topic}.",
        llm=llm
    )
    return editor.run({"topic": topic})

def run_keyword_research(topic):
    keyword_research_agent = SEOAgent(
        role="SEO Specialist",
        goal="Identify high-ranking keywords and SEO strategies for {topic}.",
        llm=llm
    )
    return keyword_research_agent.run({"topic": topic})

# Define a function to run all tasks concurrently
def run_tasks_concurrently(topic):
    tasks = {
        "plan": run_planner,
        "write": run_writer,
        "edit": run_editor,
        "keyword_research": run_keyword_research
    }
    
    # Running tasks concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_task = {executor.submit(task, topic): task_name for task_name, task in tasks.items()}
        for future in concurrent.futures.as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                output = future.result()
                print(f"{task_name} task completed with output: {output}")
            except Exception as exc:
                print(f"{task_name} task generated an exception: {exc}")

# Define the main function to execute tasks
def run_crew(topic):
    logger = logging.getLogger('seoai')
    logger.setLevel(logging.DEBUG)
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    run_tasks_concurrently(topic)

    process_logs = log_stream.getvalue()
    logger.removeHandler(handler)
    handler.close()

    return {
        "process_logs": process_logs,
        "final_output": "Generated SEO-optimized article based on the topic."
    }
