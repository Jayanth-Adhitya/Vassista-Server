"""
SMS RAG (Retrieval Augmented Generation) System using ChromaDB
Provides semantic search over SMS messages for AgentZero context
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import re
import logging

logger = logging.getLogger(__name__)


class SMSVectorStore:
    """
    ChromaDB-based vector store for SMS messages with semantic search.
    Supports time-window filtering and contact-based filtering.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_sms_db",
        collection_name: str = "sms_messages",
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize ChromaDB vector store for SMS messages.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            model_name: Sentence transformer model for embeddings
        """
        self.collection_name = collection_name
        self.model_name = model_name

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize sentence transformer model
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=None  # We'll handle embeddings manually
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Created new collection: {collection_name}")

    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess SMS text for better embeddings.

        Args:
            text: Raw SMS text

        Returns:
            Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Expand common SMS abbreviations
        abbreviations = {
            r'\bu\b': 'you',
            r'\bur\b': 'your',
            r'\br\b': 'are',
            r'\bthx\b': 'thanks',
            r'\bthnks\b': 'thanks',
            r'\bpls\b': 'please',
            r'\bmsg\b': 'message',
            r'\btmrw\b': 'tomorrow',
            r'\bmtg\b': 'meeting',
            r'\bbtw\b': 'by the way',
            r'\bidk\b': 'i do not know',
            r'\bomw\b': 'on my way'
        }

        for abbr, full in abbreviations.items():
            text = re.sub(abbr, full, text, flags=re.IGNORECASE)

        return text

    def _extract_metadata(self, sms: Dict) -> Dict:
        """
        Extract metadata from SMS message for filtering and analysis.

        Args:
            sms: SMS message dictionary with 'address', 'body', 'date'

        Returns:
            Metadata dictionary
        """
        body = sms.get('body', '')
        date_ms = sms.get('date', 0)

        # Convert timestamp to datetime
        date_dt = datetime.fromtimestamp(date_ms / 1000) if date_ms else datetime.now()

        metadata = {
            'address': sms.get('address', 'unknown'),
            'date_ms': int(date_ms),
            'date_iso': date_dt.isoformat(),
            'year': date_dt.year,
            'month': date_dt.month,
            'day': date_dt.day,
            'has_url': bool(re.search(r'https?://\S+', body)),
            'has_phone': bool(re.search(r'\d{10,}', body)),
            'has_code': bool(re.search(r'\b\d{4,6}\b', body)),  # OTP codes
            'is_question': '?' in body,
            'word_count': len(body.split()),
            'char_count': len(body)
        }

        return metadata

    def add_sms_batch(self, messages: List[Dict], time_window_days: int = 30) -> Dict:
        """
        Add batch of SMS messages to vector store with duplicate checking.
        Only indexes messages within the specified time window.

        Args:
            messages: List of SMS dictionaries with 'id', 'address', 'body', 'date'
            time_window_days: Only index messages from last N days (default 30)

        Returns:
            Dictionary with indexing statistics
        """
        if not messages:
            return {"indexed": 0, "skipped": 0, "errors": 0}

        # Filter by time window
        cutoff_time = datetime.now() - timedelta(days=time_window_days)
        cutoff_ms = int(cutoff_time.timestamp() * 1000)

        # Get existing message IDs to avoid duplicates
        try:
            existing_ids = set()
            existing_data = self.collection.get()
            if existing_data and existing_data['ids']:
                existing_ids = set(existing_data['ids'])
        except Exception as e:
            logger.warning(f"Could not fetch existing IDs: {e}")
            existing_ids = set()

        # Prepare batch data
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        indexed = 0
        skipped = 0
        errors = 0

        for msg in messages:
            try:
                msg_id = str(msg.get('id', ''))
                msg_date = msg.get('date', 0)
                msg_body = msg.get('body', '')

                # Skip if no body or ID
                if not msg_body or not msg_id:
                    skipped += 1
                    continue

                # Skip if outside time window
                if msg_date < cutoff_ms:
                    skipped += 1
                    continue

                # Skip if already indexed
                if msg_id in existing_ids:
                    skipped += 1
                    continue

                # Preprocess text
                processed_text = self._preprocess_text(msg_body)

                # Extract metadata
                metadata = self._extract_metadata(msg)

                # Generate embedding
                embedding = self.model.encode(processed_text, convert_to_numpy=True)

                # Add to batch
                ids.append(msg_id)
                documents.append(msg_body)  # Store original text
                embeddings.append(embedding.tolist())
                metadatas.append(metadata)

                indexed += 1

            except Exception as e:
                logger.error(f"Error processing message {msg.get('id')}: {e}")
                errors += 1
                continue

        # Batch upsert to ChromaDB
        if ids:
            try:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                logger.info(f"Indexed {indexed} SMS messages to ChromaDB")
            except Exception as e:
                logger.error(f"Error adding batch to ChromaDB: {e}")
                errors += indexed
                indexed = 0

        return {
            "indexed": indexed,
            "skipped": skipped,
            "errors": errors,
            "total": len(messages)
        }

    def search_sms(
        self,
        query: str,
        top_k: int = 10,
        time_window_days: Optional[int] = None,
        contact: Optional[str] = None
    ) -> List[Dict]:
        """
        Semantic search for SMS messages with optional filters.

        Args:
            query: Search query text
            top_k: Number of results to return (default 10)
            time_window_days: Only search messages from last N days
            contact: Filter by contact phone number or name

        Returns:
            List of matching SMS messages with similarity scores
        """
        try:
            # Preprocess query
            processed_query = self._preprocess_text(query)

            # Generate query embedding
            query_embedding = self.model.encode(processed_query, convert_to_numpy=True)

            # Build where filter
            where_filter = {}

            if time_window_days:
                cutoff_time = datetime.now() - timedelta(days=time_window_days)
                cutoff_ms = int(cutoff_time.timestamp() * 1000)
                where_filter["date_ms"] = {"$gte": cutoff_ms}

            if contact:
                # Filter by contact (partial match)
                where_filter["address"] = {"$contains": contact}

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, 100),  # Cap at 100 for performance
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            if results and results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'body': results['documents'][0][i],
                        'address': results['metadatas'][0][i].get('address', 'unknown'),
                        'date': results['metadatas'][0][i].get('date_ms', 0),
                        'date_iso': results['metadatas'][0][i].get('date_iso', ''),
                        'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'metadata': results['metadatas'][0][i]
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def get_recent_sms(
        self,
        limit: int = 10,
        contact: Optional[str] = None,
        days: int = 7
    ) -> List[Dict]:
        """
        Get recent SMS messages (chronological, not semantic).

        Args:
            limit: Maximum number of messages to return
            contact: Optional contact filter
            days: Number of days to look back (default 7)

        Returns:
            List of recent SMS messages
        """
        try:
            # Build where filter
            where_filter = {}

            cutoff_time = datetime.now() - timedelta(days=days)
            cutoff_ms = int(cutoff_time.timestamp() * 1000)
            where_filter["date_ms"] = {"$gte": cutoff_ms}

            if contact:
                where_filter["address"] = {"$contains": contact}

            # Get all matching messages
            results = self.collection.get(
                where=where_filter if where_filter else None,
                include=["documents", "metadatas"]
            )

            # Sort by date and limit
            if results and results['ids']:
                messages = []
                for i in range(len(results['ids'])):
                    messages.append({
                        'id': results['ids'][i],
                        'body': results['documents'][i],
                        'address': results['metadatas'][i].get('address', 'unknown'),
                        'date': results['metadatas'][i].get('date_ms', 0),
                        'date_iso': results['metadatas'][i].get('date_iso', ''),
                        'metadata': results['metadatas'][i]
                    })

                # Sort by date (newest first)
                messages.sort(key=lambda x: x['date'], reverse=True)
                return messages[:limit]

            return []

        except Exception as e:
            logger.error(f"Error getting recent SMS: {e}")
            return []

    def count_sms(self, contact: Optional[str] = None, days: Optional[int] = None) -> int:
        """
        Count total SMS messages in the store.

        Args:
            contact: Optional contact filter
            days: Optional time window in days

        Returns:
            Count of SMS messages
        """
        try:
            where_filter = {}

            if days:
                cutoff_time = datetime.now() - timedelta(days=days)
                cutoff_ms = int(cutoff_time.timestamp() * 1000)
                where_filter["date_ms"] = {"$gte": cutoff_ms}

            if contact:
                where_filter["address"] = {"$contains": contact}

            results = self.collection.get(
                where=where_filter if where_filter else None,
                include=[]  # Only need count
            )

            return len(results['ids']) if results and results['ids'] else 0

        except Exception as e:
            logger.error(f"Error counting SMS: {e}")
            return 0

    def clear_all(self):
        """Clear all SMS messages from the store. Use with caution!"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Cleared all SMS messages from vector store")
        except Exception as e:
            logger.error(f"Error clearing SMS store: {e}")


class SMSQueryAnalyzer:
    """
    Analyzes user queries to determine if SMS context is needed
    and extracts relevant filters.
    """

    def __init__(self):
        self.sms_intent_patterns = [
            r'\b(text|sms|message|msg)\b',
            r'\b(sent|received|told|said|messaged)\b',
            r'\b(from|to)\s+[\w\d]+',
            r'\b(yesterday|today|last\s+\w+)\b.*\b(text|message)',
            r'\b(check|look|find|search)\b.*\b(message|sms|text)',
            r'\b(code|otp|verification|password)\b',
            r'\b(appointment|meeting|reminder)\b',
            r'\b(address|location|directions)\b',
            r'\b(who|when|what).*\b(text|message|sms)',
        ]

    def needs_sms_context(self, query: str) -> bool:
        """
        Determine if the query requires SMS context.

        Args:
            query: User query text

        Returns:
            True if SMS context is likely needed
        """
        query_lower = query.lower()

        # Check for explicit SMS-related patterns
        for pattern in self.sms_intent_patterns:
            if re.search(pattern, query_lower):
                return True

        return False

    def extract_filters(self, query: str) -> Dict:
        """
        Extract search filters from natural language query.

        Args:
            query: User query text

        Returns:
            Dictionary of filters (time_window_days, contact, etc.)
        """
        filters = {}
        query_lower = query.lower()

        # Time-based filters
        if 'today' in query_lower:
            filters['time_window_days'] = 1
        elif 'yesterday' in query_lower:
            filters['time_window_days'] = 2
        elif 'this week' in query_lower or 'past week' in query_lower:
            filters['time_window_days'] = 7
        elif 'last week' in query_lower:
            filters['time_window_days'] = 14
        elif 'this month' in query_lower or 'past month' in query_lower:
            filters['time_window_days'] = 30
        elif 'last month' in query_lower:
            filters['time_window_days'] = 60

        # Extract phone numbers
        phone_match = re.search(r'\b\d{10,}\b', query)
        if phone_match:
            filters['contact'] = phone_match.group()

        # Extract potential contact names (simplified - could be enhanced)
        from_match = re.search(r'\bfrom\s+([\w\s]+?)(?:\s|$|,|\.|\?)', query_lower)
        if from_match:
            filters['contact'] = from_match.group(1).strip()

        to_match = re.search(r'\bto\s+([\w\s]+?)(?:\s|$|,|\.|\?)', query_lower)
        if to_match and 'contact' not in filters:
            filters['contact'] = to_match.group(1).strip()

        return filters
