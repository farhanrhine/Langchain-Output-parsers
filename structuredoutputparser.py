#structured output parser using chain
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# langchain core = keeps most imp libraries and langchain keep less imp libraries 

load_dotenv()

# Define the model

model = ChatOllama(model="tinydolphin", task="text-generation")

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='''Give 3 facts about {topic} in JSON format with keys fact_1, fact_2, fact_3. Only respond with valid JSON, no other text.
{format_instruction}''',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)