# Trainer Agent for LLM Fine-Tuning
This project automates code execution and fine-tuning tasks using Azure OpenAI's GPT-4o model in conjunction with AutoGen's agent framework. It facilitates structured collaboration between an assistant agent and a code execution agent to handle fine-tuning workflows.

## Features
+ Utilizes Azure OpenAI's GPT-4o for natural language processing.
+ Implements an AssistantAgent for task execution and a CodeExecutorAgent for running Python code.
+ Manages conversations with MagenticOneGroupChat for multi-agent collaboration.
+ Handles generated code by renaming and organizing Python scripts in the gen_code directory.
+ Supports fine-tuning llm model's with a dataset from Hugging Face.

## Prerequisites
Ensure you have the following installed:
```
Python 3.10+
```

## Environment Variables
Set up the following environment variables to authenticate with Azure OpenAI:
```
export AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
export AZURE_OPENAI_API_KEY="your_api_key"
```

## Installation
Clone the repository and install the required dependencies:
```
git clone <repository_url>
cd <repository_name>
huggingface-cli login --token <huggingface_token>
pip install -r requirements.txt
```

## Usage
Run the main script to start the fine-tuning workflow:
```
python llm_trainer.py | tee output.txt
```
The script will:
+ Initialize Azure OpenAI's GPT-4o client.
+ Configure an AssistantAgent and a CodeExecutorAgent.
+ Execute a fine-tuning task for meta-llama/Meta-Llama-3-8B.
+ Rename generated Python files sequentially in the gen_code directory.

## File Organization
+ main.py: Core script to execute the fine-tuning workflow.
+ gen_code/: Directory where generated code is stored and renamed.
+ requirements.txt: Required dependencies for the project.

## License
This project is open-source. Feel free to modify and distribute.

## Contributions
Contributions are welcome! Feel free to submit issues or pull requests.