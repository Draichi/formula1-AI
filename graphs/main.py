from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console
from rich.markdown import Markdown
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
