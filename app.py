#!/usr/bin/env python3
"""OpenCode DeepAgents — AI Coding Agent powered by deepagents, LangGraph, and Gradio.

Usage:
    python app.py                    # Launch the Gradio web UI
    python app.py --share            # Launch with public sharing link
    python app.py --port 8080        # Launch on a specific port
    python app.py --workspace /path  # Set default workspace

Examples:
    python app.py
    python app.py --port 7860 --share
    python app.py --workspace ~/my-project --mode plan
"""

import argparse
import os
import sys
from pathlib import Path

# Load .env before any LangChain imports (LangSmith, API keys, etc.)
from dotenv import load_dotenv
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="OpenCode DeepAgents — AI Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py
  python app.py --port 7860 --share
  python app.py --workspace ~/my-project --mode plan
        """,
    )
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public sharing link")
    parser.add_argument("--workspace", type=str, default=".", help="Default workspace")
    parser.add_argument("--mode", type=str, default="build", choices=["build", "plan"],
                        help="Agent mode: build or plan")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")

    args = parser.parse_args()

    os.environ["DEFAULT_WORKSPACE"] = str(Path(args.workspace).resolve())
    os.environ["DEFAULT_AGENT_MODE"] = args.mode

    from src.ui import create_app, THEME, CSS

    workspace = str(Path(args.workspace).resolve())
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║       OpenCode DeepAgents                                   ║
║  AI Coding Agent · deepagents · LangGraph · Gradio          ║
╠══════════════════════════════════════════════════════════════╣
║  Workspace: {workspace:<48} ║
║  Mode:      {args.mode:<48} ║
║  Backend:   {'LocalShellBackend' if args.mode == 'build' else 'FilesystemBackend (r/o)':<48} ║
║  URL:       http://{args.host}:{args.port:<43} ║
╚══════════════════════════════════════════════════════════════╝
""")

    app = create_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share, theme=THEME, css=CSS)


if __name__ == "__main__":
    main()
