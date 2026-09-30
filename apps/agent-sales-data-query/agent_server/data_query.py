"""
Controlled natural-language-to-SQL workflow.

The workflow retrieves the live database schema, generates
a read-only SQL query, validates it, and executes it through
the configured Databricks SQL Warehouse.
"""

import json
import logging
import re
from datetime import datetime
from decimal import Decimal
from typing import Any

from databricks_openai import AsyncDatabricksOpenAI

from agent_server.schema_discovery import get_schema_context
from agent_server.sql_execution import execute_sql, validate_sql


logger = logging.getLogger(__name__)

MODEL_NAME = "system.ai.gpt-oss-120b"


def _needs_clarification(question: str) -> str | None:
    """
    Return a clarification question when the user's request
    contains an evaluative term without a measurable criterion.
    """

    normalized = question.lower().strip()

    vague_patterns = [
        r"\b(best|top|worst|most valuable)\s+(customers?|products?)\b",
        r"\b(best|top|worst)\s+perform(ing|ance)\b",
    ]

    has_vague_term = any(
        re.search(pattern, normalized)
        for pattern in vague_patterns
    )

    has_criterion = any(
        term in normalized
        for term in [
            "by revenue",
            "by sales",
            "by amount",
            "by profit",
            "by quantity",
            "by order count",
            "by number of orders",
            "by spending",
            "by total",
            "by price",
            "by units",
        ]
    )

    if has_vague_term and not has_criterion:
        return (
            "What metric should I use to define the ranking? "
            "For example, revenue, total spending, order count, "
            "or quantity sold."
        )

    return None


def _extract_response_text(response: Any) -> str:
    """
    Extract generated text from a Responses API result.
    """

    output_text = getattr(response, "output_text", None)

    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    # Fallback for SDK response objects without output_text.
    collected = []

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text_value = getattr(content, "text", None)

            if isinstance(text_value, str):
                collected.append(text_value)

    if collected:
        return "\n".join(collected).strip()

    raise ValueError("The model returned no text output.")


def _parse_json_response(text: str) -> dict:
    """
    Parse a JSON object, tolerating Markdown code fences.
    """

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

    parsed = json.loads(cleaned)

    if not isinstance(parsed, dict):
        raise ValueError("The model response must be a JSON object.")

    return parsed


def _json_safe(value: Any) -> Any:
    """
    Convert common SQL result values into JSON-safe values.
    """

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value


async def generate_sql(user_question: str) -> dict:
    """
    Generate SQL grounded in the current database schema.

    The generated SQL is not executed by this function.
    """

    if not isinstance(user_question, str) or not user_question.strip():
        return {
            "success": False,
            "status": "error",
            "error": "Please provide a non-empty question.",
        }

    schema_result = get_schema_context()

    if not schema_result["success"]:
        return {
            "success": False,
            "status": "error",
            "error": (
                "Could not retrieve the database schema: "
                + str(schema_result["error"])
            ),
        }

    clarification = _needs_clarification(user_question)

    if clarification:
        return {
            "success": True,
            "status": "clarification_needed",
            "clarification_question": clarification,
        }

    schema_json = json.dumps(
        schema_result["schema_context"],
        indent=2,
        default=str,
    )

    system_prompt = f"""
You are a Databricks SQL generation component for a read-only
sales analytics assistant.

Generate exactly one Databricks SQL SELECT query to answer the
user's question. Use only the schema supplied below.

DATABASE SCHEMA:
{schema_json}

STRICT RULES:
- Use only tables and columns present in the supplied schema.
- Fully qualify every physical table as
  main.sales_agent_demo.table_name.
- Use only SELECT statements, including read-only CTEs.
- Never generate INSERT, UPDATE, DELETE, MERGE, CREATE, DROP,
  ALTER, TRUNCATE, or other data-changing statements.
- Do not invent columns, tables, relationships, or values.
- Use only documented relationships when joining tables.
- If the question cannot be answered from the schema, return
  status "clarification_needed" and explain what is missing.
- Do not include Markdown fences.
- Return only one valid JSON object with these keys:
  status, sql, tables_used, columns_used, joins, filters,
  calculations, reasoning, assumptions,
  clarification_question.
- For a query that can be answered, status must be "ready".
- For an ambiguous or unsupported question, status must be
  "clarification_needed" and sql must be null.
"""

    client = AsyncDatabricksOpenAI(use_ai_gateway=True)

    try:
        response = await client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_question,
                },
            ],
        )

        response_text = _extract_response_text(response)
        generated = _parse_json_response(response_text)

    except Exception as exc:
        logger.exception("SQL generation failed.")

        return {
            "success": False,
            "status": "error",
            "error": f"SQL generation failed: {exc}",
        }

    status = generated.get("status")

    if status == "clarification_needed":
        return {
            "success": True,
            "status": "clarification_needed",
            "clarification_question": (
                generated.get("clarification_question")
                or "Please clarify your request."
            ),
            "reasoning": generated.get("reasoning"),
        }

    sql_query = generated.get("sql")

    if status != "ready" or not isinstance(sql_query, str):
        return {
            "success": False,
            "status": "error",
            "error": (
                "The SQL generator did not return a valid "
                "ready response with a SQL query."
            ),
        }

    return {
        "success": True,
        "status": "ready",
        "sql": sql_query,
        "tables_used": generated.get("tables_used", []),
        "columns_used": generated.get("columns_used", []),
        "joins": generated.get("joins", []),
        "filters": generated.get("filters", []),
        "calculations": generated.get("calculations", []),
        "reasoning": generated.get("reasoning", ""),
        "assumptions": generated.get("assumptions", []),
        "clarification_question": None,
    }


async def answer_data_question(user_question: str) -> dict:
    """
    Run the complete controlled data-question workflow.
    """

    generation = await generate_sql(user_question)

    if not generation["success"]:
        return {
            "success": False,
            "status": "error",
            "question": user_question,
            "error": generation.get("error"),
        }

    if generation["status"] == "clarification_needed":
        return {
            "success": True,
            "status": "clarification_needed",
            "question": user_question,
            "clarification_question": generation.get(
                "clarification_question"
            ),
            "reasoning": generation.get("reasoning"),
        }

    sql_query = generation["sql"]

    validation = validate_sql(sql_query)

    if not validation["valid"]:
        return {
            "success": False,
            "status": "validation_failed",
            "question": user_question,
            "sql": sql_query,
            "error": validation["error"],
            "tables_used": validation.get("tables_used", []),
        }

    execution = execute_sql(sql_query)

    if not execution["success"]:
        return {
            "success": False,
            "status": "execution_failed",
            "question": user_question,
            "sql": sql_query,
            "error": execution["error"],
            "tables_used": execution.get("tables_used", []),
        }

    return _json_safe({
        "success": True,
        "status": "completed",
        "question": user_question,
        "sql": sql_query,
        "tables_used": execution["tables_used"],
        "columns_used": generation.get("columns_used", []),
        "joins": generation.get("joins", []),
        "filters": generation.get("filters", []),
        "calculations": generation.get("calculations", []),
        "reasoning": generation.get("reasoning", ""),
        "assumptions": generation.get("assumptions", []),
        "columns": execution["columns"],
        "rows": execution["rows"],
        "row_count": execution["row_count"],
        "truncated": execution["truncated"],
        "error": None,
    })
