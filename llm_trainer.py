import os
import glob
import asyncio
import torch
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.tools.code_execution import PythonCodeExecutionTool


WORK_DIR = "gen_code"
MAX_TOKENS = 100000  # Limit to stay below the model's 128k token limit

# Azure OpenAI
llm = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4o",
    model="gpt-4o",
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

def truncate_messages(messages, max_tokens=MAX_TOKENS):
    """Truncate messages to ensure they fit within the model's token limit."""
    total_tokens = 0
    truncated_messages = []
    
    for message in reversed(messages):  # Start from the latest message
        message_tokens = len(message.split())  # Rough estimate
        if total_tokens + message_tokens > max_tokens:
            break  # Stop adding messages before exceeding limit
        truncated_messages.insert(0, message)  # Maintain order
        total_tokens += message_tokens
    
    return truncated_messages

async def main() -> None:
    model_client = llm
    tool = PythonCodeExecutionTool(LocalCommandLineCodeExecutor(timeout=1000, work_dir=WORK_DIR))
    code_executor = LocalCommandLineCodeExecutor(timeout=1000, work_dir=WORK_DIR)

    assistant = AssistantAgent("assistant", model_client, tools=[tool], reflect_on_tool_use=True)
    code_executor_agent = CodeExecutorAgent("code_executor_agent", code_executor)

    team = MagenticOneGroupChat(
        [assistant, code_executor_agent],
        model_client=model_client,
        max_stalls=20,
        max_turns=50,
    )

    fine_tune_task = """
    Fine-tune meta-llama/Meta-Llama-3-8B with a DPO dataset (trl-lib/ultrafeedback_binarized) from Hugging Face.
    Use the following configurations:
    - Mixed Precision: bf16 or fp16
    - Memory Optimization: Gradient Checkpointing
    - Ensure max_tokens=4096 to avoid truncation issues
    - Save trained model for inference
    - Log all terminal outputs for debugging
    """

    try:
        truncated_task = truncate_messages([fine_tune_task])  # Ensure task size is within limit
        await Console(team.run_stream(task=truncated_task[0]))
    except Exception as e:
        print("Error occurred:", str(e))

    rename_generated_code()

def rename_generated_code():
    """Renames the most recent generated Python file in `gen_code`."""
    files = glob.glob(os.path.join(WORK_DIR, "*.py"))
    if not files:
        print("No Python files found in", WORK_DIR)
        return

    latest_file = max(files, key=os.path.getctime)
    existing_files = glob.glob(os.path.join(WORK_DIR, "trial_*.py"))
    new_index = len(existing_files) + 1
    new_filename = os.path.join(WORK_DIR, f"trial_{new_index}.py")

    os.rename(latest_file, new_filename)
    print(f"Renamed {latest_file} → {new_filename}")

asyncio.run(main())
