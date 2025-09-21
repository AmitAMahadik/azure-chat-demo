from dotenv import load_dotenv
import os
from os.path import dirname
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory

current_dir = dirname(os.path.abspath(__file__))
root_dir = dirname(dirname(current_dir))
env_file = os.path.join(root_dir, '.env')


async def main():
    # Load the .env file. Replace the path with the path to your .env file.
    load_dotenv(env_file)
    deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]

    kernel = Kernel()

    # Add the chat service
    chat_service = AzureChatCompletion(
        deployment_name=deployment_name,
        endpoint=endpoint,
        api_key=api_key,
        api_version="2024-12-01-preview"
    )
    kernel.add_service(chat_service)

    # Create chat history with the prompt
    chat_history = ChatHistory()
    chat_history.add_user_message("""
    I need to understand what are the variables involved in making outstanding espresso besides
    a good machine. For example what is the combination of roast, grind, tamp, and water temperature.
    Include 3 practical steps to practice and improve each variable.
    """)

    # Get the chat completion service and get a response
    chat_completion = kernel.get_service(type=AzureChatCompletion)
    
    # Get execution settings
    execution_settings = chat_completion.get_prompt_execution_settings_class()()

    # Get a response from the chat completion service
    response = await chat_completion.get_chat_message_contents(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel
    )

    if response:
        print(response[0].content)
    else:
        print("No response received")


# Run the main function
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
