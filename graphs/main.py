from langchain_neo4j import Neo4jGraph, GraphCypherQAChain, Neo4jVector
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from rich.markdown import Markdown
from operator import add
from typing import Annotated, List
from typing_extensions import TypedDict
from typing import Literal, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema
from neo4j.exceptions import CypherSyntaxError
from langgraph.graph import END, START, StateGraph
from langchain_huggingface import HuggingFaceEmbeddings

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


console.print("\n\nImplementing guardrails\n\n")

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


console.print("\n\nFew-shot prompting\n\n")

# Few-shot prompting
examples = [
    {
        "question": "How many artists are there?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)",
    },
    {
        "question": "Which actors played in the movie Casino?",
        "query": "MATCH (m:Movie {title: 'Casino'})<-[:ACTED_IN]-(a) RETURN a.name",
    },
    {
        "question": "How many movies has Tom Hanks acted in?",
        "query": "MATCH (a:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN count(m)",
    },
    {
        "question": "List all the genres of the movie Schindler's List",
        "query": "MATCH (m:Movie {title: 'Schindler's List'})-[:IN_GENRE]->(g:Genre) RETURN g.name",
    },
    {
        "question": "Which actors have worked in movies from both the comedy and action genres?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name",
    },
    {
        "question": "Which directors have made movies with at least three different actors named 'John'?",
        "query": "MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person) WHERE a.name STARTS WITH 'John' WITH d, COUNT(DISTINCT a) AS JohnsCount WHERE JohnsCount >= 3 RETURN d.name",
    },
    {
        "question": "Identify movies where directors also played a role in the film.",
        "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name",
    },
    {
        "question": "Find the actor with the highest number of movies in the database.",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1",
    },
]

# Replace the HuggingFace embeddings with OpenAI embeddings
embeddings = OpenAIEmbeddings()  # This produces 1536-dimensional embeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    Neo4jVector,
    k=5,
    input_keys=["question"],
)

console.print("\n\nText to Cypher\n\n")

# Cypher generation chain - text to cypher
text2cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Given an input question, convert it to a Cypher query. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
Here is the schema information
{schema}

Below are a number of examples of questions and their corresponding Cypher queries.

{fewshot_examples}

User input: {question}
Cypher query:"""
            ),
        ),
    ]
)

text2cypher_chain = text2cypher_prompt | llm | StrOutputParser()


def generate_cypher(state: OverallState) -> OverallState:
    """
    Generates a cypher statement based on the provided schema and user input
    """
    NL = "\n"
    fewshot_examples = (NL * 2).join(
        [
            f"Question: {el['question']}{NL}Cypher:{el['query']}"
            for el in example_selector.select_examples(
                {"question": state.get("question")}
            )
        ]
    )
    generated_cypher = text2cypher_chain.invoke(
        {
            "question": state.get("question"),
            "fewshot_examples": fewshot_examples,
            "schema": graph.schema,
        }
    )
    return {
        "question": state.get("question", ""),
        "next_action": "query_validation",
        "cypher_statement": generated_cypher,
        "cypher_errors": [],
        "database_records": [],
        "steps": ["generate_cypher"]
    }


console.print("\n\nQuery validation\n\n")
# Query validation chain
validate_cypher_system = """
You are a Cypher expert reviewing a statement written by a junior developer.
"""

validate_cypher_user = """You must check the following:
* Are there any syntax errors in the Cypher statement?
* Are there any missing or undefined variables in the Cypher statement?
* Are any node labels missing from the schema?
* Are any relationship types missing from the schema?
* Are any of the properties not included in the schema?
* Does the Cypher statement include enough information to answer the question?

Examples of good errors:
* Label (:Foo) does not exist, did you mean (:Bar)?
* Property bar does not exist for label Foo, did you mean baz?
* Relationship FOO does not exist, did you mean FOO_BAR?

Schema:
{schema}

The question is:
{question}

The Cypher statement is:
{cypher}

Make sure you don't make any mistakes!"""

validate_cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            validate_cypher_system,
        ),
        (
            "human",
            (validate_cypher_user),
        ),
    ]
)


class ValidateCypherOutput(BaseModel):
    """
    Represents the validation result of a Cypher query's output.
    """
    errors: List[str] = Field(
        default_factory=list,
        description="A list of syntax or semantical errors in the Cypher statement."
    )
    # Simplify filters to just be strings
    filters: List[str] = Field(
        default_factory=list,
        description="A list of filters found in the Cypher statement"
    )


validate_cypher_chain = validate_cypher_prompt | llm.with_structured_output(
    ValidateCypherOutput
)

# LLMs often struggle with correctly determining relationship directions in generated Cypher statements.
# Since we have access to the schema, we can deterministically correct these directions using the CypherQueryCorrector.

# Cypher query corrector is experimental
corrector_schema = [
    Schema(el["start"], el["type"], el["end"])
    for el in graph.structured_schema.get("relationships", [])
]
cypher_query_corrector = CypherQueryCorrector(corrector_schema)


def validate_cypher(state: OverallState) -> OverallState:
    """
    Validates the Cypher statements and maps any property values to the database.
    """
    errors = []
    mapping_errors = []
    try:
        graph.query(f"EXPLAIN {state.get('cypher_statement')}")
    except CypherSyntaxError as e:
        errors.append(e.message)

    corrected_cypher = cypher_query_corrector(state.get("cypher_statement"))
    if not corrected_cypher:
        errors.append(
            "The generated Cypher statement doesn't fit the graph schema")
    if not corrected_cypher == state.get("cypher_statement"):
        print("Relationship direction was corrected")

    try:
        # Use LLM to find additional potential errors and get the mapping for values
        raw_llm_output = validate_cypher_chain.invoke(
            {
                "question": state.get("question"),
                "schema": graph.schema,
                "cypher": state.get("cypher_statement"),
            }
        )

        # Print raw LLM output for debugging
        print("\nRaw LLM validation output:")
        print(raw_llm_output)
        print("\n")

        llm_output = ValidateCypherOutput.model_validate(raw_llm_output)

        if llm_output.errors:
            errors.extend(llm_output.errors)

        # Skip filter validation for now since we're just debugging
        print("\nFilters found:")
        print(llm_output.filters)

    except Exception as e:
        print(f"Error during validation: {str(e)}")
        errors.append(f"Validation error: {str(e)}")

    if mapping_errors:
        next_action = "end"
    elif errors:
        next_action = "correct_cypher"
    else:
        next_action = "execute_cypher"

    return {
        "question": state.get("question", ""),
        "next_action": next_action,
        "cypher_statement": corrected_cypher,
        "cypher_errors": errors,
        "database_records": [],
        "steps": ["validate_cypher"]
    }


# The Cypher correction step takes the existing Cypher statement,
# any identified errors, and the original question to generate a corrected version of the query.
correct_cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a Cypher expert reviewing a statement written by a junior developer. "
                "You need to correct the Cypher statement based on the provided errors. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """Check for invalid syntax or semantics and return a corrected Cypher statement.

Schema:
{schema}

Note: Do not include any explanations or apologies in your responses.
Do not wrap the response in any backticks or anything else.
Respond with a Cypher statement only!

Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

The question is:
{question}

The Cypher statement is:
{cypher}

The errors are:
{errors}

Corrected Cypher statement: """
            ),
        ),
    ]
)

correct_cypher_chain = correct_cypher_prompt | llm | StrOutputParser()


def correct_cypher(state: OverallState) -> OverallState:
    """
    Correct the Cypher statement based on the provided errors.
    """
    corrected_cypher = correct_cypher_chain.invoke(
        {
            "question": state.get("question"),
            "errors": state.get("cypher_errors"),
            "cypher": state.get("cypher_statement"),
            "schema": graph.schema,
        }
    )

    return {
        "question": state.get("question", ""),
        "next_action": "validate_cypher",
        "cypher_statement": corrected_cypher,
        "cypher_errors": [],
        "database_records": [],
        "steps": ["correct_cypher"],
    }


# We need to add a step that executes the given Cypher statement. If no results are returned,
# we should explicitly handle this scenario, as leaving the context empty can sometimes lead to LLM hallucinations.
no_results = "I couldn't find any relevant information in the database"


def execute_cypher(state: OverallState) -> OverallState:
    """
    Executes the given Cypher statement.
    """

    records = graph.query(state.get("cypher_statement"))
    return {
        "question": state.get("question", ""),
        "next_action": "end",
        "cypher_statement": state.get("cypher_statement", ""),
        "cypher_errors": [],
        "database_records": records if records else [{"error": no_results}],
        "steps": ["execute_cypher"],
    }


# The final step is to generate the answer.
# This involves combining the initial question with the database output to produce a relevant response.
generate_final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant",
        ),
        (
            "human",
            (
                """Use the following results retrieved from a database to provide
a succinct, definitive answer to the user's question.

Respond as if you are answering the question directly.

Results: {results}
Question: {question}"""
            ),
        ),
    ]
)

generate_final_chain = generate_final_prompt | llm | StrOutputParser()


def generate_final_answer(state: OverallState) -> OutputState:
    """
    Decides if the question is related to movies.
    """
    final_answer = generate_final_chain.invoke(
        {"question": state.get("question"),
         "results": state.get("database_records")}
    )
    return {
        "answer": final_answer,
        "steps": ["generate_final_answer"],
        "cypher_statement": state.get("cypher_statement", "")
    }


# Next, we will implement the LangGraph workflow, starting with defining the conditional edge functions.
def guardrails_condition(
    state: OverallState,
) -> Literal["generate_cypher", "generate_final_answer"]:
    if state.get("next_action") == "end":
        return "generate_final_answer"
    elif state.get("next_action") == "movie":
        return "generate_cypher"
    else:
        return "generate_final_answer"  # default fallback


def validate_cypher_condition(
    state: OverallState,
) -> Literal["generate_final_answer", "correct_cypher", "execute_cypher"]:
    if state.get("next_action") == "end":
        return "generate_final_answer"
    elif state.get("next_action") == "correct_cypher":
        return "correct_cypher"
    elif state.get("next_action") == "execute_cypher":
        return "execute_cypher"
    else:
        return "generate_final_answer"  # default fallback


console.print("\n\nLangGraph workflow\n\n")
# Next, we will define the workflow using the LangGraph framework.
langgraph = StateGraph(OverallState, input=InputState, output=OutputState)
langgraph.add_node(guardrails)
langgraph.add_node(generate_cypher)
langgraph.add_node(validate_cypher)
langgraph.add_node(correct_cypher)
langgraph.add_node(execute_cypher)
langgraph.add_node(generate_final_answer)

langgraph.add_edge(START, "guardrails")
langgraph.add_conditional_edges(
    "guardrails",
    guardrails_condition,
)
langgraph.add_edge("generate_cypher", "validate_cypher")
langgraph.add_conditional_edges(
    "validate_cypher",
    validate_cypher_condition,
)
langgraph.add_edge("execute_cypher", "generate_final_answer")
langgraph.add_edge("correct_cypher", "validate_cypher")
langgraph.add_edge("generate_final_answer", END)

langgraph = langgraph.compile()

console.print("\n\nTesting the application\n\n")

console.print("Testing an irrelevant question\n\n")
# We can now test the application by asking an irrelevant question.
response = langgraph.invoke({"question": "What's the weather in Spain?"})
console.print(response)

console.print("\n\nTesting a question about movies\n\n")
# We can also ask a question about movies.
response = langgraph.invoke({"question": "What was the cast of the Casino?"})
console.print(response)
