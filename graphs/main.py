from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from rich.markdown import Markdown
from operator import add
from typing import Annotated, List
from typing_extensions import TypedDict
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

import dotenv
import os

dotenv.load_dotenv()

console = Console(style="bold bright_green on black")

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    enhanced_schema=True,
)

# Graph Schema
console.print(Markdown(graph.schema))

# Graph Cypher QA Chain
query = "What was the cast of the Casino?"
console.print(f"\n\nQuery: {query}\n\n")

# llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
)
chain = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=llm,
    verbose=True,
    allow_dangerous_requests=True,
)
response = chain.invoke({"query": query})

console.print(response)


# Advanced Implementation
class InputState(TypedDict):
    question: str


class OverallState(TypedDict):
    question: str
    next_action: str
    cypher_statement: str
    cypher_errors: List[str]
    database_records: List[dict]
    steps: Annotated[List[str], add]


class OutputState(TypedDict):
    answer: str
    steps: List[str]
    cypher_statement: str


# Guardrails
guardrails_system = """
As an intelligent assistant, your primary objective is to decide whether a given question is related to movies or not. 
If the question is related to movies, output "movie". Otherwise, output "end".
To make this decision, assess the content of the question and determine if it refers to any movie, actor, director, film industry, 
or related topics. Provide only the specified output: "movie" or "end".
"""
guardrails_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            guardrails_system,
        ),
        (
            "human",
            ("{question}"),
        ),
    ]
)


class GuardrailsOutput(BaseModel):
    decision: Literal["movie", "end"] = Field(
        description="Decision on whether the question is related to movies"
    )


guardrails_chain = guardrails_prompt | llm.with_structured_output(
    GuardrailsOutput)


def guardrails(state: InputState) -> OverallState:
    """
    Decides if the question is related to movies or not.
    """
    guardrails_output = GuardrailsOutput.model_validate(
        guardrails_chain.invoke({"question": state.get("question")})
    )
    database_records = []
    if guardrails_output.decision == "end":
        database_records = [
            {"error": "This questions is not about movies or their cast. Therefore I cannot answer this question."}]
    return {
        "question": state.get("question", ""),
        "next_action": guardrails_output.decision,
        "cypher_statement": "",
        "cypher_errors": [],
        "database_records": database_records,
        "steps": ["guardrail"],
    }
