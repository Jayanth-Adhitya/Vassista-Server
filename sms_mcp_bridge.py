#!/usr/bin/env python3
"""
MCP Bridge Server for SMS Search
This runs locally with AgentZero but calls the remote SMS server via HTTP
"""

import asyncio
import logging
import requests
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sms-mcp-bridge")

# Remote SMS server URL
SMS_SERVER_URL = "https://zswok4sc8c44w804kw8gss8g.uptopoint.net"

# Create MCP server instance
app = Server("sms-search-bridge")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    List all available SMS search tools.
    """
    try:
        # Fetch tools from remote server
        response = requests.get(f"{SMS_SERVER_URL}/mcp/tools", timeout=10)
        response.raise_for_status()
        data = response.json()

        # Convert to MCP Tool format
        tools = []
        for tool_info in data.get("tools", []):
            tools.append(Tool(
                name=tool_info["name"],
                description=tool_info["description"],
                inputSchema=tool_info["input_schema"]
            ))

        logger.info(f"Loaded {len(tools)} tools from remote server")
        return tools

    except Exception as e:
        logger.error(f"Failed to fetch tools from server: {e}")
        # Return minimal tool set as fallback
        return [
            Tool(
                name="search_sms",
                description="Search through the user's SMS messages semantically",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results (default: 10)",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_recent_sms",
                description="Get recent SMS messages",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Max messages (default: 10)",
                            "default": 10
                        },
                        "days": {
                            "type": "integer",
                            "description": "Days to look back (default: 7)",
                            "default": 7
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="count_sms",
                description="Count total SMS messages",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "Count from last N days (optional)"
                        }
                    },
                    "required": []
                }
            )
        ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """
    Forward tool calls to the remote SMS server via HTTP.
    """
    try:
        logger.info(f"🔧 Forwarding tool call: {name} with args: {arguments}")

        # Call remote server
        response = requests.post(
            f"{SMS_SERVER_URL}/mcp/call-tool",
            json={
                "tool_name": name,
                "arguments": arguments
            },
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        if result.get("success"):
            logger.info(f"✅ Tool call successful: {name}")
            return [TextContent(
                type="text",
                text=result.get("result", "No result")
            )]
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"❌ Tool call failed: {error_msg}")
            return [TextContent(
                type="text",
                text=f"Error: {error_msg}"
            )]

    except requests.exceptions.Timeout:
        logger.error(f"⏱️ Tool call timeout: {name}")
        return [TextContent(
            type="text",
            text="Error: Request timed out. The SMS server is taking too long to respond."
        )]

    except requests.exceptions.RequestException as e:
        logger.error(f"🌐 Network error calling {name}: {e}")
        return [TextContent(
            type="text",
            text=f"Error: Could not connect to SMS server - {str(e)}"
        )]

    except Exception as e:
        logger.error(f"💥 Unexpected error calling {name}: {e}")
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


async def main():
    """
    Run the MCP bridge server via stdio.
    AgentZero will communicate with this via standard input/output.
    """
    logger.info("🚀 Starting SMS MCP Bridge Server...")
    logger.info(f"📡 Remote SMS Server: {SMS_SERVER_URL}")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
