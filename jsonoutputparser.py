# from langchain_ollama import ChatOllama
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser

# load_dotenv()

# # Define the model
# model = ChatOllama(
#     model="tinydolphin",
#     task="text-generation"
# )

# parser = JsonOutputParser()

# template = PromptTemplate(
#     template='Give me 5 facts about {topic} \n {format_instruction}',
#     input_variables=['topic'],
#     partial_variables={'format_instruction': parser.get_format_instructions()}
# )

# chain = template | model | parser

# result = chain.invoke({'topic':'black hole'})

# print(result)

from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Define the model
model = ChatOllama(
    model="tinydolphin",
    task="text-generation"
)

# Define the parser
parser = JsonOutputParser()

# Create a prompt with format instructions
# template = """
# You are a helpful assistant that provides information about various topics.
# Please provide 5 facts about {topic} in JSON format.

# {format_instructions}
# """

template = """
You are a JSON API. Respond ONLY with valid JSON. Do NOT include any units (like kg, km, AU, etc.) in the values. 
All numbers must be plain numbers, not strings, and not include units. 

Example:
{
  "BlackHoles": [
    {
      "Name": "Example Black Hole",
      "Mass": 123456,
      "Radius": 789,
      "Diameter": 1000,
      "GravitationalFieldStrength": 0.0001,
      "EnergyContent": "Some fact about the black hole."
    }
  ]
}

Now, provide 5 facts about black holes in this exact JSON structure. DO NOT include any units or extra text.
{format_instructions}
"""



prompt = PromptTemplate(
    template=template,
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Create the chain
chain = prompt | model | parser

try:
    result = chain.invoke({"topic": "black hole"})
    print("Successfully parsed JSON:")
    print(result)
except Exception as e:
    print(f"Error: {str(e)}")
    print("\nPlease try again. The model might have returned invalid JSON format.")


# result = chain.invoke({'topic':'black hole'})

# print(result)