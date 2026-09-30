import os
from typing import Any

from databricks import sql
from databricks.sdk import WorkspaceClient


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CATALOG_NAME = "main"
SCHEMA_NAME = "sales_agent_demo"
FULL_SCHEMA_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}"

WAREHOUSE_ID = os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID")

APPROVED_TABLES = {
    "customers",
    "products",
    "sales_orders",
    "sales_order_items",
    "payments",
}


# ---------------------------------------------------------
# Documented table relationships
# ---------------------------------------------------------

TABLE_RELATIONSHIPS = [
    {
        "from_table": "customers",
        "from_column": "customer_id",
        "to_table": "sales_orders",
        "to_column": "customer_id",
        "relationship": "one_to_many",
        "description": "A customer can place multiple orders.",
    },
    {
        "from_table": "sales_orders",
        "from_column": "order_id",
        "to_table": "sales_order_items",
        "to_column": "order_id",
        "relationship": "one_to_many",
        "description": "An order can contain multiple order items.",
    },
    {
        "from_table": "products",
        "from_column": "product_id",
        "to_table": "sales_order_items",
        "to_column": "product_id",
        "relationship": "one_to_many",
        "description": "A product can appear in multiple order items.",
    },
    {
        "from_table": "sales_orders",
        "from_column": "order_id",
        "to_table": "payments",
        "to_column": "order_id",
        "relationship": "one_to_many",
        "description": "An order can have multiple payment records.",
    },
]


# ---------------------------------------------------------
# SQL Warehouse connection
# ---------------------------------------------------------

def get_sql_connection():
    """
    Create a SQL Warehouse connection using the app's
    configured Databricks runtime identity.
    """

    if not WAREHOUSE_ID:
        raise RuntimeError(
            "DATABRICKS_SQL_WAREHOUSE_ID is not configured."
        )

    workspace_client = WorkspaceClient()

    if not workspace_client.config.host:
        raise RuntimeError(
            "Databricks workspace host is not configured."
        )

    return sql.connect(
        server_hostname=workspace_client.config.host.replace(
            "https://", ""
        ),
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        credentials_provider=(
            workspace_client.config.oauth_credentials_provider
        ),
    )


# ---------------------------------------------------------
# SQL execution helper
# ---------------------------------------------------------

def execute_metadata_sql(query: str) -> dict:
    """
    Execute a metadata query through the configured SQL Warehouse.

    Returns:
    - success
    - columns
    - rows
    - row_count
    - error
    """

    connection = None

    try:
        connection = get_sql_connection()

        with connection.cursor() as cursor:
            cursor.execute(query)

            column_names = [
                column[0]
                for column in (cursor.description or [])
            ]

            rows = [
                dict(zip(column_names, row))
                for row in cursor.fetchall()
            ]

        return {
            "success": True,
            "columns": column_names,
            "rows": rows,
            "row_count": len(rows),
            "error": None,
        }

    except Exception as exc:
        return {
            "success": False,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "error": str(exc),
        }

    finally:
        if connection is not None:
            connection.close()


# ---------------------------------------------------------
# Discover tables
# ---------------------------------------------------------

def discover_tables() -> dict:
    """
    Discover approved tables in the configured schema.
    """

    query = f"SHOW TABLES IN {FULL_SCHEMA_NAME}"

    result = execute_metadata_sql(query)

    if not result["success"]:
        return {
            "success": False,
            "catalog": CATALOG_NAME,
            "schema": SCHEMA_NAME,
            "tables": [],
            "table_count": 0,
            "error": result["error"],
        }

    discovered_names = [
        row.get("tableName")
        for row in result["rows"]
        if row.get("tableName")
    ]

    # Restrict discovery to the explicitly approved tables.
    table_names = [
        name
        for name in discovered_names
        if name in APPROVED_TABLES
    ]

    return {
        "success": True,
        "catalog": CATALOG_NAME,
        "schema": SCHEMA_NAME,
        "tables": table_names,
        "table_count": len(table_names),
        "error": None,
    }


# ---------------------------------------------------------
# Discover columns
# ---------------------------------------------------------

def discover_columns(table_name: str) -> dict:
    """
    Discover column metadata for an approved table.
    """

    if table_name not in APPROVED_TABLES:
        return {
            "success": False,
            "table": table_name,
            "columns": [],
            "column_count": 0,
            "error": (
                f"Table '{table_name}' is not in the "
                "approved schema."
            ),
        }

    allowed_tables = discover_tables()

    if not allowed_tables["success"]:
        return {
            "success": False,
            "table": table_name,
            "columns": [],
            "column_count": 0,
            "error": allowed_tables["error"],
        }

    if table_name not in allowed_tables["tables"]:
        return {
            "success": False,
            "table": table_name,
            "columns": [],
            "column_count": 0,
            "error": (
                f"Table '{table_name}' was not found "
                "in the configured schema."
            ),
        }

    full_table_name = f"{FULL_SCHEMA_NAME}.{table_name}"

    result = execute_metadata_sql(
        f"DESCRIBE TABLE {full_table_name}"
    )

    if not result["success"]:
        return {
            "success": False,
            "table": table_name,
            "columns": [],
            "column_count": 0,
            "error": result["error"],
        }

    columns = []

    for row in result["rows"]:
        column_name = row.get("col_name")
        data_type = row.get("data_type")
        comment = row.get("comment")

        if not column_name:
            continue

        if column_name.startswith("#"):
            continue

        columns.append(
            {
                "name": column_name,
                "data_type": data_type,
                "comment": comment,
            }
        )

    return {
        "success": True,
        "table": table_name,
        "columns": columns,
        "column_count": len(columns),
        "error": None,
    }


# ---------------------------------------------------------
# Discover full schema
# ---------------------------------------------------------

def discover_full_schema() -> dict:
    """
    Discover columns for every table in the configured schema.
    """

    table_result = discover_tables()

    if not table_result["success"]:
        return {
            "success": False,
            "catalog": CATALOG_NAME,
            "schema": SCHEMA_NAME,
            "tables": {},
            "error": table_result["error"],
        }

    schema_metadata = {}

    for table_name in table_result["tables"]:
        column_result = discover_columns(table_name)

        if not column_result["success"]:
            return {
                "success": False,
                "catalog": CATALOG_NAME,
                "schema": SCHEMA_NAME,
                "tables": schema_metadata,
                "error": column_result["error"],
            }

        schema_metadata[table_name] = column_result["columns"]

    return {
        "success": True,
        "catalog": CATALOG_NAME,
        "schema": SCHEMA_NAME,
        "tables": schema_metadata,
        "table_count": len(schema_metadata),
        "error": None,
    }


# ---------------------------------------------------------
# Discover relationships
# ---------------------------------------------------------

def discover_relationships(table_name: str = None) -> dict:
    """
    Return documented table relationships.

    If table_name is supplied, return only relationships
    involving that table.
    """

    if table_name is not None:
        allowed_tables = discover_tables()

        if not allowed_tables["success"]:
            return {
                "success": False,
                "relationships": [],
                "relationship_count": 0,
                "error": allowed_tables["error"],
            }

        if table_name not in allowed_tables["tables"]:
            return {
                "success": False,
                "relationships": [],
                "relationship_count": 0,
                "error": (
                    f"Table '{table_name}' is not in "
                    "the approved schema."
                ),
            }

        relationships = [
            relationship
            for relationship in TABLE_RELATIONSHIPS
            if (
                relationship["from_table"] == table_name
                or relationship["to_table"] == table_name
            )
        ]

    else:
        relationships = TABLE_RELATIONSHIPS.copy()

    return {
        "success": True,
        "relationships": relationships,
        "relationship_count": len(relationships),
        "error": None,
    }


# ---------------------------------------------------------
# Build schema context for the AI agent
# ---------------------------------------------------------

def get_schema_context() -> dict:
    """
    Combine table metadata and documented relationships
    into a single context object.
    """

    schema_result = discover_full_schema()
    relationship_result = discover_relationships()

    if not schema_result["success"]:
        return {
            "success": False,
            "schema_context": None,
            "error": schema_result["error"],
        }

    if not relationship_result["success"]:
        return {
            "success": False,
            "schema_context": None,
            "error": relationship_result["error"],
        }

    context = {
        "catalog": CATALOG_NAME,
        "schema": SCHEMA_NAME,
        "tables": schema_result["tables"],
        "relationships": relationship_result["relationships"],
    }

    return {
        "success": True,
        "schema_context": context,
        "error": None,
    }
