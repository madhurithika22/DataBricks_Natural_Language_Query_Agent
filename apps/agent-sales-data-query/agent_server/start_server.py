
from pathlib import Path

from dotenv import load_dotenv
from mlflow.genai.agent_server import (
    AgentServer,
    setup_mlflow_git_based_version_tracking,
)

# Load environment variables before importing the agent.
load_dotenv(
    dotenv_path=Path(__file__).parent.parent / ".env",
    override=True,
)

# Import the agent to register its functions.
import agent_server.agent  # noqa: E402

from agent_server.sql_execution import execute_sql, validate_sql  # noqa: E402


agent_server = AgentServer(
    "ResponsesAgent",
    enable_chat_proxy=True,
)

# Expose the FastAPI application.
app = agent_server.app

# Temporary diagnostic endpoint for Phase 3A verification.
@app.get("/debug/sql")
async def debug_sql():
    """
    Test read-only SQL execution and validation.

    Remove this endpoint after Phase 3A testing.
    """

    test_cases = {
        "customers_count": {
            "query": (
                "SELECT COUNT(*) AS row_count "
                "FROM main.sales_agent_demo.customers"
            ),
            "expected_valid": True,
        },
        "products_count": {
            "query": (
                "SELECT COUNT(*) AS row_count "
                "FROM main.sales_agent_demo.products"
            ),
            "expected_valid": True,
        },
        "sales_orders_count": {
            "query": (
                "SELECT COUNT(*) AS row_count "
                "FROM main.sales_agent_demo.sales_orders"
            ),
            "expected_valid": True,
        },
        "sales_order_items_count": {
            "query": (
                "SELECT COUNT(*) AS row_count "
                "FROM main.sales_agent_demo.sales_order_items"
            ),
            "expected_valid": True,
        },
        "payments_count": {
            "query": (
                "SELECT COUNT(*) AS row_count "
                "FROM main.sales_agent_demo.payments"
            ),
            "expected_valid": True,
        },
        "write_statement_rejected": {
            "query": (
                "INSERT INTO main.sales_agent_demo.customers "
                "(customer_id) VALUES (99999)"
            ),
            "expected_valid": False,
        },
        "unapproved_table_rejected": {
            "query": (
                "SELECT * FROM main.sales_agent_demo.unknown_table"
            ),
            "expected_valid": False,
        },
        "read_only_cte": {
            "query": (
                "WITH customer_counts AS ("
                "SELECT COUNT(*) AS total_customers "
                "FROM main.sales_agent_demo.customers"
                ") SELECT total_customers FROM customer_counts"
            ),
            "expected_valid": True,
        },
    }

    results = {}

    for test_name, test_case in test_cases.items():
        query = test_case["query"]
        expected_valid = test_case["expected_valid"]

        validation = validate_sql(query)

        test_result = {
            "expected_valid": expected_valid,
            "validation": validation,
        }

        if validation["valid"]:
            test_result["execution"] = execute_sql(query)

        test_result["passed"] = (
            validation["valid"] == expected_valid
        )

        if expected_valid and validation["valid"]:
            test_result["passed"] = (
                test_result["execution"]["success"]
            )

        results[test_name] = test_result

    passed = sum(
        1 for result in results.values()
        if result["passed"]
    )

    return {
        "success": passed == len(results),
        "passed": passed,
        "total": len(results),
        "results": results,
    }


setup_mlflow_git_based_version_tracking()


def main():
    agent_server.run(
        app_import_string="agent_server.start_server:app"
    )
