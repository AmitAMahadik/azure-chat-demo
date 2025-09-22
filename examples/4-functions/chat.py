import os
from os.path import dirname
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelFunctionFromPrompt
from semantic_kernel.functions import KernelArguments


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


    # Load the wine pairing prompt template
    prompt_template = """Generate 3 potential dishes for pairing the given wine. The wine can be a distinct grape, a type of wine like red, white, or bubbly. Sometimes the given wine description might not be super accurate, so make sure you always suggest dishes regardless.
Wine: {{$input}}
"""

    # Create the function from prompt
    wine_pairing_function = KernelFunctionFromPrompt(
        function_name="somellier",
        plugin_name="WinePlugin",
        prompt=prompt_template,
        description="Pair a type of wine with possible dishes."
    )

    # Add the function to the kernel
    kernel.add_function(plugin_name="WinePlugin", function=wine_pairing_function)

    # Invoke the wine pairing function with proper arguments
    arguments = KernelArguments(input="Gewürztraminer")
    
    result = await kernel.invoke(
        function_name="somellier",
        plugin_name="WinePlugin",
        arguments=arguments
    )
    
    print(result)


# Run the main function
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
