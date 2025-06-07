from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define the model
model = ChatOllama(
    model="tinydolphin",
    temperature=0.3,
)

class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='''Generate a fictional person from {place}. 

Return ONLY a valid JSON object with these exact keys:
- name: A full name from {place}
- age: An integer above 18
- city: A city in {place}

IMPORTANT: Return ONLY the JSON object, nothing else. Do not include any explanations, markdown, or additional text.



Now generate a person from {place} following this exact format:
{format_instruction}''',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'place':'Saudi Arabia'})

print(final_result)
