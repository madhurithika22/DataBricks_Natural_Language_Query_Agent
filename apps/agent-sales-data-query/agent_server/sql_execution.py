
"""
Read-only SQL execution for the Sales Data Query Agent.

This module validates SQL before executing it through the
configured Databricks SQL Warehouse.
"""

from typing import Any

import sqlglot
from sqlglot import exp

from agent_server.schema_discovery import (
    APPROVED_TABLES,
    FULL_SCHEMA_NAME,
    get_sql_connection,
)


MAX_RESULT_ROWS = 1000


def validate_sql(query: str) -> dict:
    """
    Validate a single read-only SELECT query.

    Checks:
    - Query is not empty.
    - Exactly one SQL statement is supplied.
    - Root statement is SELECT.
    - Physical tables belong to the approved schema.
    - EXPLAIN succeeds in the SQL Warehouse.

    CTE aliases are permitted and are not treated as physical tables.
    """

    if not isinstance(query, str) or not query.strip():
        return {
            "success": False,
            "valid": False,
            "error": "SQL query is empty.",
        }

    try:
        parsed_statements = sqlglot.parse(
            query,
            read="databricks",
        )
    except Exception as exc:
        return {
            "success": False,
            "valid": False,
            "error": f"SQL parsing failed: {exc}",
        }

    statements = [
        statement
        for statement in parsed_statements
        if statement is not None
    ]

    if len(statements) != 1:
        return {
            "success": False,
            "valid": False,
            "error": "Only one SQL statement is allowed.",
        }

    statement = statements[0]

    if not isinstance(statement, exp.Select):
        return {
            "success": False,
            "valid": False,
            "error": "Only SELECT queries are permitted.",
        }

    # Collect CTE names so references to CTEs are not
    # mistaken for references to physical database tables.
    cte_names = {
        cte.alias_or_name.lower()
        for cte in statement.find_all(exp.CTE)
        if cte.alias_or_name
    }

    referenced_tables = []

    for table in statement.find_all(exp.Table):
        table_name = (table.name or "").lower()
        schema_name = (table.db or "").lower()
        catalog_name = (table.catalog or "").lower()

        # A CTE reference has no catalog or schema qualifier.
        if (
            table_name in cte_names
            and not schema_name
            and not catalog_name
        ):
            continue

        if table_name not in APPROVED_TABLES:
            return {
                "success": False,
                "valid": False,
                "error": (
                    f"Table '{table_name}' is not approved."
                ),
            }

        if schema_name != "sales_agent_demo":
            return {
                "success": False,
                "valid": False,
                "error": (
                    f"Table '{table_name}' must be referenced "
                    "from schema main.sales_agent_demo."
                ),
            }

        if catalog_name != "main":
            return {
                "success": False,
                "valid": False,
                "error": (
                    f"Table '{table_name}' must use catalog main."
                ),
            }

        referenced_tables.append(
            f"{catalog_name}.{schema_name}.{table_name}"
        )

    if not referenced_tables:
        return {
            "success": False,
            "valid": False,
            "error": (
                "The query must reference at least one "
                "approved physical table."
            ),
        }

    connection = None

    try:
        connection = get_sql_connection()

        with connection.cursor() as cursor:
            cursor.execute(f"EXPLAIN {query}")
            cursor.fetchall()

        return {
            "success": True,
            "valid": True,
            "error": None,
            "tables_used": sorted(set(referenced_tables)),
        }

    except Exception as exc:
        return {
            "success": False,
            "valid": False,
            "error": f"SQL validation failed: {exc}",
        }

    finally:
        if connection is not None:
            connection.close()


def execute_sql(query: str) -> dict:
    """
    Validate and execute a read-only SQL query.

    Returns structured query results. Results are capped at
    MAX_RESULT_ROWS to protect the app from oversized responses.
    """

    validation = validate_sql(query)

    if not validation["valid"]:
        return {
            "success": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error": validation["error"],
            "tables_used": [],
        }

    connection = None

    try:
        connection = get_sql_connection()

        with connection.cursor() as cursor:
            cursor.execute(query)

            columns = [
                column[0]
                for column in (cursor.description or [])
            ]

            fetched_rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)

            truncated = len(fetched_rows) > MAX_RESULT_ROWS
            fetched_rows = fetched_rows[:MAX_RESULT_ROWS]

            rows = [
                dict(zip(columns, row))
                for row in fetched_rows
            ]

        return {
            "success": True,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "error": None,
            "tables_used": validation["tables_used"],
        }

    except Exception as exc:
        return {
            "success": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "error": str(exc),
            "tables_used": validation["tables_used"],
        }

    finally:
        if connection is not None:
            connection.close()
