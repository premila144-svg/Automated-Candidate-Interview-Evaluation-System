from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from dotenv import load_dotenv
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.base import TaskResult
import os

load_dotenv()

# ✅ Use GROQ API instead of OpenAI
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in environment variables.")

# ✅ LLaMA 3.3 Model Config
model_client = OpenAIChatCompletionClient(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "family": "unknown",
    },
)


# ---------------- AGENTS ---------------- #
async def my_agents(job_position="AI Engineer"):

    interviewer = AssistantAgent(
        name="Interviewer",
        model_client=model_client,
        description=f"An AI agent that conducts interviews for a {job_position} position.",
        system_message=f"""
You are a professional interviewer for a {job_position} position.
Ask one clear question at a time and wait for user response.
Ignore evaluator feedback while asking questions.
Ask questions based on candidate answers and your expertise.

Ask exactly 3 questions:
1. Technical
2. Problem-solving
3. Culture fit

After 3 questions, say 'TERMINATE'.
Keep each question under 50 words.
""",
    )

    candidate = UserProxyAgent(
        name="Candidate",
        description=f"A candidate for {job_position}",
        input_func=input
    )

    evaluator = AssistantAgent(
        name="Evaluator",
        model_client=model_client,
        description="Career Coach",
        system_message=f"""
You are a career coach for {job_position}.
Give short feedback on each answer.
After interview, summarize performance with suggestions.

Max 100 words.
""",
    )

    terminate_condition = TextMentionTermination(text="TERMINATE")

    team = RoundRobinGroupChat(
        participants=[interviewer, candidate, evaluator],
        termination_condition=terminate_condition,
        max_turns=20
    )

    return team


# ---------------- RUN INTERVIEW ---------------- #
async def run_interview(team):
    async for message in team.run_stream(task="Start the interview with the first question."):

        if isinstance(message, TaskResult):
            yield f"Interview completed: {message.stop_reason}"

        else:
            yield f"{message.source}: {message.content}"


# ---------------- MAIN ---------------- #
async def main():
    job_position = "AI Engineer"

    team = await my_agents(job_position)

    async for message in run_interview(team):
        print("-" * 60)
        print(message)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())