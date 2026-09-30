
from pathlib import Path

from dotenv import load_dotenv
from mlflow.genai.agent_server import (
    AgentServer,
    setup_mlflow_git_based_version_tracking,
)

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

import agent_server.agent  # noqa: E402

agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=True)
app = agent_server.app

setup_mlflow_git_based_version_tracking()


def main():
    agent_server.run(app_import_string="agent_server.start_server:app")
