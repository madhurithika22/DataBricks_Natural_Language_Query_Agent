import json
import logging
from contextlib import AsyncExitStack
from datetime import datetime
from typing import AsyncGenerator

import mlflow
from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.history import normalize_history_items
from agent_server.schema_discovery import get_schema_context
from agent_server.utils import (
    build_mcp_url,
    get_session_id,
    get_user_workspace_client,
    process_agent_stream_events,
)

logger = logging.getLogger(__name__)

# NOTE: this will work for all databricks models OTHER than GPT-OSS, which uses a slightly different API
set_default_openai_client(AsyncDatabricksOpenAI(use_ai_gateway=True))
set_default_openai_api("responses")
set_trace_processors([])  # only use mlflow for trace processing
mlflow.openai.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)


@function_tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


@function_tool
def get_database_schema() -> str:
    """
    Retrieve the approved sales database schema,
    including tables, columns, data types, and
    documented table relationships.

    Use this tool whenever a user asks about available
    data, tables, columns, or relationships, and before
    answering questions that depend on the database schema.
    """

    result = get_schema_context()

    if not result["success"]:
        return json.dumps({
            "success": False,
            "error": result["error"],
        })

    return json.dumps({
        "success": True,
        "schema_context": result["schema_context"],
    })

async def init_mcp_server(workspace_client: WorkspaceClient):
    return McpServer(
        url=build_mcp_url("/api/2.0/mcp/functions/system/ai", workspace_client=workspace_client),
        name="system.ai UC function MCP server",
        workspace_client=workspace_client,
    )


async def connect_healthy_mcp_servers(
    stack: AsyncExitStack, servers: list[McpServer]
) -> tuple[list[McpServer], list[str]]:
    """Connect each MCP server and verify it can actually list its tools.

    The Agents SDK lists each server's tools lazily inside ``Runner.run``, so a server that
    connects but fails at list time (e.g. an unauthorized Genie space) would otherwise crash
    the whole request — including unrelated turns. We list tools here, per server: healthy
    servers are kept; any that fails to connect OR to list is dropped and its name returned,
    so the agent runs with whatever is available instead of erroring out.

    Returns (healthy_servers, unavailable_names).
    """
    healthy: list[McpServer] = []
    unavailable: list[str] = []
    for server in servers:
        name = getattr(server, "name", "MCP server")
        try:
            connected = await stack.enter_async_context(server)
            await connected.list_tools()  # forces the connectivity + authorization check now
            healthy.append(connected)
        except Exception:
            logger.warning("MCP server %r unavailable; continuing without it.", name, exc_info=True)
            unavailable.append(name)
    return healthy, unavailable



def create_agent(mcp_servers: list[McpServer] | None = None) -> Agent:
    return Agent(
        name="Sales Data Query Agent",
        instructions="""
You are a helpful data assistant for the sales analytics database.

You can answer general questions and explain the database structure.

DATABASE RULES:

1. When a user asks about available tables, columns,
   data types, or table relationships, call
   get_database_schema() to retrieve the actual schema.

2. Before answering any question that depends on the
   database structure, retrieve the schema first.

3. Use only table and column names returned by the
   schema tool.

4. Never invent tables, columns, or relationships.

5. Explain database structures clearly and accurately.

6. If schema discovery fails, explain that the database
   schema could not be retrieved. Do not invent a result.

7. Do not execute SQL queries in this phase.

8. Do not claim to have retrieved or analyzed sales
   records unless an approved data-query tool has
   actually executed a query.


9. If a user asks for a data calculation or result that
   requires querying records, explain that data-query
   execution is not yet available.

10. Never invent or suggest database tables or columns.
    Only refer to table and column names returned by
    get_database_schema().


STRICT DATA-GROUNDING RULES:

11. Never invent, guess, or suggest database column names,
    table names, or SQL fields.

12. When a user asks for a calculation that requires
    querying records, explain that SQL execution is
    not yet available.

13. Do not suggest example SQL fields such as order_total,
    line_total, order_amount, or revenue_amount unless
    those exact fields have been returned by the
    get_database_schema tool.

14. If the schema tool returns an error, report that
    the schema lookup failed. Do not claim that the
    schema is unavailable if the tool returned success.

15. If a question requires data that has not been queried,
    do not estimate, infer, or fabricate a numerical result.

16. Distinguish clearly between:
    - Schema information retrieved from the database
    - Data results retrieved by executing SQL
    - Information that cannot yet be verified
""",
        model="system.ai.gpt-oss-120b",
        tools=[get_current_time, get_database_schema],
        mcp_servers=mcp_servers or [],
    )



def is_data_query_request(user_question: str) -> bool:
    """
    Identify questions that require querying actual sales records.

    Schema questions are handled separately by the schema tool.
    This guard prevents the model from inventing data results
    while SQL execution is not yet connected.
    """

    question = user_question.lower().strip()

    data_query_terms = [
        "revenue",
        "sales total",
        "total sales",
        "total amount",
        "how many",
        "how much",
        "count of",
        "number of",
        "average",
        "sum of",
        "top customers",
        "best customers",
        "highest",
        "lowest",
        "most popular",
        "least popular",
        "which customer",
        "which product",
        "how many orders",
        "show me the sales",
        "list the customers",
        "list the orders",
    ]

    schema_terms = [
        "what tables",
        "available tables",
        "what columns",
        "available columns",
        "table structure",
        "schema",
        "relationship",
        "relationships",
        "data types",
    ]

    # Schema questions should continue to the agent.
    if any(term in question for term in schema_terms):
        return False

    return any(term in question for term in data_query_terms)


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    messages = normalize_history_items([i.model_dump() for i in request.input])

    # Find the latest user message.
    latest_user_message = ""

    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content", "")

            if isinstance(content, str):
                latest_user_message = content
            elif isinstance(content, list):
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") in ("text", "input_text")
                ]
                latest_user_message = " ".join(text_parts)

            break

    # Deterministically block data queries until SQL execution is connected.
    if is_data_query_request(latest_user_message):
        final_answer = (
            "I can identify the database tables, columns, and relationships, "
            "but I cannot yet query sales records to calculate this result. "
            "I will not guess a number or invent database fields. "
            "Actual data-query execution is not connected yet."
        )

        return ResponsesAgentResponse(
            output=[
                {
                    "type": "message",
                    "id": f"msg-{datetime.now().timestamp()}",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": final_answer,
                        }
                    ],
                }
            ]
        )

    # Continue normal agent processing for schema and general questions.
    async with AsyncExitStack() as stack:
        agent = create_agent()
        result = await Runner.run(agent, messages)

        final_answer = result.final_output

        if not isinstance(final_answer, str):
            final_answer = str(final_answer)

        return ResponsesAgentResponse(
            output=[
                {
                    "type": "message",
                    "id": f"msg-{datetime.now().timestamp()}",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": final_answer,
                        }
                    ],
                }
            ]
        )


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
    # The agent runs inside an AsyncExitStack so any MCP servers stay open for the whole
    # request. To give the agent MCP tools, connect them with connect_healthy_mcp_servers,
    # which health-checks each server so one unavailable server can't crash the request
    # (the Agents SDK lists each server's tools lazily inside Runner.run):
    #   servers, unavailable = await connect_healthy_mcp_servers(
    #       stack, [await init_mcp_server(WorkspaceClient())])
    #   agent = create_agent(mcp_servers=servers)
    # WorkspaceClient() uses service principal credentials; use get_user_workspace_client()
    # for on-behalf-of user authentication.
    async with AsyncExitStack() as stack:
        agent = create_agent()
        messages = normalize_history_items([i.model_dump() for i in request.input])
        result = Runner.run_streamed(agent, input=messages)

        async for event in process_agent_stream_events(result.stream_events()):
            yield event
