"""
MCP Server for SMS Search Tools
Exposes SMS search capabilities to AgentZero via Model Context Protocol
"""

import asyncio
import logging
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from sms_rag import SMSVectorStore, SMSQueryAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sms-mcp-server")

# Initialize SMS vector store
sms_store = None
query_analyzer = None

# Create MCP server instance
app = Server("sms-search-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    List all available SMS search tools for AgentZero.
    """
    return [
        Tool(
            name="search_sms",
            description=(
                "Search through the user's SMS messages semantically to find relevant conversations. "
                "Use this when the user asks about messages, texts, money received, OTPs, "
                "appointments, or any information that might be in their SMS history."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant SMS messages (e.g., 'money received from Kotak')"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of most relevant messages to return (default: 10)",
                        "default": 10
                    },
                    "time_window_days": {
                        "type": "integer",
                        "description": "Only search messages from the last N days (optional)",
                        "default": None
                    },
                    "contact": {
                        "type": "string",
                        "description": "Filter by contact phone number or name (optional)",
                        "default": None
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_recent_sms",
            description=(
                "Get the most recent SMS messages in chronological order. "
                "Use this when the user asks about their latest messages or recent conversations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to return (default: 10)",
                        "default": 10
                    },
                    "contact": {
                        "type": "string",
                        "description": "Filter by contact phone number or name (optional)",
                        "default": None
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back (default: 7)",
                        "default": 7
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="count_sms",
            description=(
                "Count the total number of SMS messages in the database. "
                "Use this when the user asks how many messages they have."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "contact": {
                        "type": "string",
                        "description": "Filter by contact phone number or name (optional)",
                        "default": None
                    },
                    "days": {
                        "type": "integer",
                        "description": "Count only messages from the last N days (optional)",
                        "default": None
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="analyze_sms_for_money",
            description=(
                "Analyze SMS messages to find and sum up money transactions. "
                "Specifically designed to extract amounts from bank SMS, payment confirmations, etc. "
                "Use this when the user asks about money received, sent, or total amounts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "transaction_type": {
                        "type": "string",
                        "description": "Type of transaction: 'received', 'sent', 'all' (default: 'all')",
                        "default": "all"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look back (default: 30)",
                        "default": 30
                    },
                    "contact": {
                        "type": "string",
                        "description": "Filter by specific contact/bank (optional)",
                        "default": None
                    }
                },
                "required": []
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """
    Handle tool calls from AgentZero.
    """
    global sms_store, query_analyzer

    # Lazy initialization of SMS store
    if sms_store is None:
        try:
            logger.info("Initializing SMS vector store...")
            sms_store = SMSVectorStore(
                persist_directory="./chroma_sms_db",
                collection_name="sms_messages",
                model_name="all-MiniLM-L6-v2"
            )
            query_analyzer = SMSQueryAnalyzer()
            logger.info("SMS vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SMS store: {e}")
            return [TextContent(
                type="text",
                text=f"Error: SMS search is currently unavailable - {str(e)}"
            )]

    try:
        if name == "search_sms":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 10)
            time_window_days = arguments.get("time_window_days")
            contact = arguments.get("contact")

            logger.info(f"🔍 MCP Tool Call: search_sms(query='{query}', top_k={top_k})")

            results = await asyncio.to_thread(
                sms_store.search_sms,
                query,
                top_k=top_k,
                time_window_days=time_window_days,
                contact=contact
            )

            if not results:
                return [TextContent(
                    type="text",
                    text="No relevant SMS messages found."
                )]

            # Format results
            formatted_results = []
            for i, msg in enumerate(results, 1):
                date_iso = msg.get('date_iso', '')
                address = msg.get('address', 'unknown')
                body = msg.get('body', '')
                similarity = msg.get('similarity', 0)

                formatted_results.append(
                    f"{i}. [{date_iso}] From: {address} (relevance: {similarity:.2f})\n"
                    f"   Message: {body}"
                )

            result_text = "\n\n".join(formatted_results)
            return [TextContent(
                type="text",
                text=f"Found {len(results)} relevant SMS messages:\n\n{result_text}"
            )]

        elif name == "get_recent_sms":
            limit = arguments.get("limit", 10)
            contact = arguments.get("contact")
            days = arguments.get("days", 7)

            logger.info(f"📱 MCP Tool Call: get_recent_sms(limit={limit}, days={days})")

            results = await asyncio.to_thread(
                sms_store.get_recent_sms,
                limit=limit,
                contact=contact,
                days=days
            )

            if not results:
                return [TextContent(
                    type="text",
                    text="No recent SMS messages found."
                )]

            # Format results
            formatted_results = []
            for i, msg in enumerate(results, 1):
                date_iso = msg.get('date_iso', '')
                address = msg.get('address', 'unknown')
                body = msg.get('body', '')

                formatted_results.append(
                    f"{i}. [{date_iso}] From: {address}\n"
                    f"   Message: {body}"
                )

            result_text = "\n\n".join(formatted_results)
            return [TextContent(
                type="text",
                text=f"Recent SMS messages (last {days} days):\n\n{result_text}"
            )]

        elif name == "count_sms":
            contact = arguments.get("contact")
            days = arguments.get("days")

            logger.info(f"📊 MCP Tool Call: count_sms(days={days})")

            count = await asyncio.to_thread(
                sms_store.count_sms,
                contact=contact,
                days=days
            )

            time_desc = f"in the last {days} days" if days else "total"
            contact_desc = f" from {contact}" if contact else ""

            return [TextContent(
                type="text",
                text=f"Found {count} SMS messages{contact_desc} ({time_desc})."
            )]

        elif name == "analyze_sms_for_money":
            transaction_type = arguments.get("transaction_type", "all")
            days = arguments.get("days", 30)
            contact = arguments.get("contact")

            logger.info(f"💰 MCP Tool Call: analyze_sms_for_money(type={transaction_type}, days={days})")

            # Search for money-related keywords
            keywords = {
                "received": ["credited", "received", "deposited", "added", "refund"],
                "sent": ["debited", "paid", "sent", "transferred", "withdrawn"],
                "all": ["credited", "received", "deposited", "debited", "paid", "sent", "transferred", "Rs", "INR", "₹"]
            }

            search_keywords = keywords.get(transaction_type, keywords["all"])
            query = " OR ".join(search_keywords)

            results = await asyncio.to_thread(
                sms_store.search_sms,
                query,
                top_k=100,  # Get many results for analysis
                time_window_days=days,
                contact=contact
            )

            if not results:
                return [TextContent(
                    type="text",
                    text=f"No money-related SMS messages found in the last {days} days."
                )]

            # Extract amounts using regex
            import re
            total_amount = 0
            transactions = []

            for msg in results:
                body = msg.get('body', '')
                date_iso = msg.get('date_iso', '')
                address = msg.get('address', '')

                # Find amounts (Rs. 1234, INR 1234, ₹1234, etc.)
                amount_patterns = [
                    r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
                    r'INR\s*(\d+(?:,\d+)*(?:\.\d+)?)',
                    r'₹\s*(\d+(?:,\d+)*(?:\.\d+)?)'
                ]

                for pattern in amount_patterns:
                    matches = re.findall(pattern, body, re.IGNORECASE)
                    for match in matches:
                        # Remove commas and convert to float
                        amount = float(match.replace(',', ''))

                        # Determine if received or sent
                        is_received = any(kw in body.lower() for kw in ["credited", "received", "deposited", "added", "refund"])
                        is_sent = any(kw in body.lower() for kw in ["debited", "paid", "sent", "transferred", "withdrawn"])

                        if transaction_type == "received" and not is_received:
                            continue
                        if transaction_type == "sent" and not is_sent:
                            continue

                        total_amount += amount if is_received else -amount if is_sent else amount

                        transactions.append({
                            'amount': amount,
                            'type': 'received' if is_received else 'sent' if is_sent else 'unknown',
                            'date': date_iso,
                            'from': address,
                            'message': body[:100] + '...' if len(body) > 100 else body
                        })

            # Format results
            if not transactions:
                return [TextContent(
                    type="text",
                    text=f"Found {len(results)} SMS messages, but could not extract any monetary amounts."
                )]

            summary = f"Money Analysis (Last {days} days):\n\n"
            summary += f"Total Amount: ₹{total_amount:,.2f}\n"
            summary += f"Transactions Found: {len(transactions)}\n\n"
            summary += "Recent Transactions:\n"

            for i, txn in enumerate(transactions[:20], 1):  # Show top 20
                summary += f"{i}. [{txn['date']}] {txn['type'].upper()}: ₹{txn['amount']:,.2f} from {txn['from']}\n"
                summary += f"   {txn['message']}\n\n"

            return [TextContent(
                type="text",
                text=summary
            )]

        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]


async def main():
    """
    Run the MCP server via stdio.
    AgentZero will communicate with this server via standard input/output.
    """
    logger.info("Starting SMS MCP Server...")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
