# app.py
import os
from os.path import dirname
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory

# load .env locally; in Azure we'll use App Settings instead
current_dir = dirname(os.path.abspath(__file__))
root_dir = dirname(dirname(current_dir))
env_file = os.path.join(root_dir, '.env')
load_dotenv(env_file)

# read Azure OpenAI creds
DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
API_KEY = os.environ["AZURE_OPENAI_API_KEY"]

# set up kernel + service once (reused per request)
kernel = Kernel()
chat_service = AzureChatCompletion(
    deployment_name=DEPLOYMENT,
    endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version="2024-12-01-preview"
)
kernel.add_service(chat_service)
chat_completion = kernel.get_service(type=AzureChatCompletion)

app = FastAPI()

class ChatIn(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(body: ChatIn):
    chat_history = ChatHistory()
    chat_history.add_user_message(body.prompt)

    settings = chat_completion.get_prompt_execution_settings_class()()
    response = await chat_completion.get_chat_message_contents(
        chat_history=chat_history,
        settings=settings,
        kernel=kernel
    )
    return {"reply": (response[0].content if response else "No response")}