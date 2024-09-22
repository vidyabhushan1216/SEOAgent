import os
import io
import logging
import concurrent.futures
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is present
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not found. Please set it in your .env file or system environment.")

# Initialize the OpenAI LLM
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-3.5-turbo",  # This model has a free usage tier.
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
        response = self.llm.generate([prompt])
        return response[0].text  # Return the text of the response

# Define tasks as functions
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

# Run all tasks concurrently
def run_tasks_concurrently(topic):
    tasks = {
        "plan": run_planner,
        "write": run_writer,
        "edit": run_editor,
        "keyword_research": run_keyword_research
    }

    # Capture outputs from all tasks
    task_outputs = {}
    
    # Run tasks concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_task = {executor.submit(task, topic): task_name for task_name, task in tasks.items()}
        for future in concurrent.futures.as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                task_outputs[task_name] = future.result()
            except Exception as exc:
                task_outputs[task_name] = f"{task_name} task generated an exception: {exc}"

    return task_outputs

# Function to capture logs and final output
def run_crew(topic):
    logger = logging.getLogger('seoai')
    logger.setLevel(logging.DEBUG)
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # Capture outputs from all tasks
    tasks_outputs = run_tasks_concurrently(topic)

    process_logs = log_stream.getvalue()
    logger.removeHandler(handler)
    handler.close()

    # Collect final article output
    final_output = tasks_outputs.get("write", "No final article generated.")  # Ensures the written content is returned

    return {
        "process_logs": process_logs,
        "final_output": final_output  # The generated article
    }
