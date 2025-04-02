huggingface-cli login --token <huggingface_token>
conda install python=3.10.9
pip install autogen-agentchat==0.4.4
pip install aiofiles
pip install playwright
pip install markitdown
python llm_trainer.py | tee output.txt
