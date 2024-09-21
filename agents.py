import os
import io
import logging
import concurrent.futures
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentExecutor

# Load the environment variables from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI LLM (you can also use Hugging Face models for free)
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
        response = self.llm.generate_responses([prompt])
        return response[0]

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

# Example of execution with concurrency
if __name__ == "__main__":
    # Define the topic for content creation and SEO research
    topic = "The impact of AI on healthcare"
    
    # Configure logging to capture logs
    logger = logging.getLogger('seoai')
    logger.setLevel(logging.DEBUG)
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    # Run tasks concurrently
    run_tasks_concurrently(topic)

    # Retrieve logs
    process_logs = log_stream.getvalue()
    logger.removeHandler(handler)
    handler.close()

    # Output the logs
    print("Process Logs:")
    print(process_logs)
