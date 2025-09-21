from dotenv import load_dotenv
import os
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function


class TravelWeather:
    @kernel_function(
        description="Takes a city and a month and returns the average temperature for that month.",
        name="travel_weather",
    )
    def weather(self, city: str, month: str) -> str:
        return f"The average temperature in {city} in {month} is 75 degrees."


async def main():
    # Load the .env file from the examples directory or current directory
    env_paths = [
        #'./examples/1-simple/.env',
        './.env'
    ]
    
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
    
    # Try both old and new environment variable names
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")

    if not all([deployment_name, endpoint, api_key]):
        print("Error: Missing required environment variables. Please check your .env file.")
        print("Expected variables: AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY")
        return

    # Initialize the kernel
    kernel = Kernel()
    
    # Add the chat service
    chat_service = AzureChatCompletion(
        deployment_name=deployment_name,
        endpoint=endpoint,
        api_key=api_key,
        api_version="2024-12-01-preview"
    )
    kernel.add_service(chat_service)

    # Add the weather plugin
    weather_plugin = TravelWeather()
    kernel.add_plugin(weather_plugin, plugin_name="Travel")

    # Create chat history
    chat_history = ChatHistory()
    chat_history.add_system_message("You are a boisterous travel weather chat bot. Your name is Frederick. You are trying to help people find the average temperature in a city in a month.")
    chat_history.add_user_message("What is the average temperature in San Francisco in June?")

    # Get the chat completion service
    chat_completion = kernel.get_service(type=AzureChatCompletion)
    
    # Enable function calling
    execution_settings = chat_completion.get_prompt_execution_settings_class()(
        function_call_behavior="auto", temperature=1.0, max_completion_tokens=5000, prompt_template="You are a travel weather chat bot. Your name is Frederick. You are trying to help people find the average temperature in a city in a month."
    )

    # Get a response from the chat completion service
    response = await chat_completion.get_chat_message_contents(
        chat_history=chat_history,
        settings=execution_settings,
        kernel=kernel
    )

    if response:
        print(f"Assistant: {response[0].content}")
        chat_history.add_assistant_message(str(response[0].content))
    else:
        print("No response received")


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())