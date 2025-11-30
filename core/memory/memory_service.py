"""
In-memory memory service with multi-turn conversation tracking.
Replace with Redis/Postgres-backed Memory Bank for production.
Stores sessions, conversation history, user preferences, and long-term facts.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

class ConversationTurn:
    """Represents a single turn in the conversation (user query + bot response)."""
    
    def __init__(self, user_message: str, bot_response: str = None, route: str = None, sources: List[Dict] = None, metadata: Dict = None):
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow().isoformat()
        self.user_message = user_message
        self.bot_response = bot_response
        self.route = route  # "rag", "web", "hybrid", "error"
        self.sources = sources or []  # List of source chunks/links
        self.metadata = metadata or {}  # Additional metadata (latency, model, etc)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "user_message": self.user_message,
            "bot_response": self.bot_response,
            "route": self.route,
            "sources": self.sources,
            "metadata": self.metadata
        }


class MemoryService:
    """
    Manages session state and multi-turn conversation history.
    Tracks context across queries, maintaining both immediate and long-term memory.
    """
    
    def __init__(self):
        # {session_id: session_data}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        # {user_id: long_term_data}
        self.long_term: Dict[str, Dict[str, Any]] = {}

    def create_session(self, user_id: str = None) -> str:
        """Create a new session with empty conversation history."""
        sid = str(uuid.uuid4())
        self.sessions[sid] = {
            "created_at": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "context": {},
            "conversation_history": [],  # List of ConversationTurn objects
            "ingested_docs": set(),  # Track ingested document IDs
            "last_activity": datetime.utcnow().isoformat()
        }
        return sid

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session data."""
        return self.sessions.get(session_id, None)

    def update_session_context(self, session_id: str, key: str, value: Any):
        """Update session context (immediate state)."""
        if session_id not in self.sessions:
            raise KeyError("Session not found")
        self.sessions[session_id]["context"][key] = value
        self.sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()

    def add_conversation_turn(
        self,
        session_id: str,
        user_message: str,
        bot_response: str = None,
        route: str = None,
        sources: List[Dict] = None,
        metadata: Dict = None
    ) -> ConversationTurn:
        """Add a new turn to the conversation history."""
        if session_id not in self.sessions:
            raise KeyError("Session not found")
        
        turn = ConversationTurn(user_message, bot_response, route, sources, metadata)
        self.sessions[session_id]["conversation_history"].append(turn)
        self.sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()
        return turn

    def get_conversation_history(self, session_id: str, max_turns: int = None) -> List[Dict]:
        """Get conversation history for the session."""
        if session_id not in self.sessions:
            raise KeyError("Session not found")
        
        history = self.sessions[session_id]["conversation_history"]
        if max_turns:
            history = history[-max_turns:]
        
        return [turn.to_dict() if isinstance(turn, ConversationTurn) else turn for turn in history]

    def get_context_summary(self, session_id: str) -> str:
        """
        Create a summary of recent conversation context for the LLM.
        Includes last N turns to maintain context window efficiency.
        """
        if session_id not in self.sessions:
            return ""
        
        history = self.sessions[session_id]["conversation_history"]
        if not history:
            return ""
        
        # Build context from last 5 turns (10 messages)
        recent_turns = history[-5:]
        context_lines = ["### Conversation Context:\n"]
        
        for turn in recent_turns:
            if isinstance(turn, ConversationTurn):
                context_lines.append(f"User: {turn.user_message}")
                if turn.bot_response:
                    context_lines.append(f"Assistant: {turn.bot_response[:200]}...")
            else:
                context_lines.append(f"User: {turn.get('user_message', '')}")
                if turn.get('bot_response'):
                    context_lines.append(f"Assistant: {turn.get('bot_response', '')[:200]}...")
        
        return "\n".join(context_lines)

    def register_ingested_doc(self, session_id: str, doc_id: str):
        """Register that a document has been ingested in this session."""
        if session_id not in self.sessions:
            raise KeyError("Session not found")
        self.sessions[session_id]["ingested_docs"].add(doc_id)

    def get_ingested_docs(self, session_id: str) -> set:
        """Get all ingested document IDs for this session."""
        if session_id not in self.sessions:
            return set()
        return self.sessions[session_id]["ingested_docs"]

    def get_long_term(self, user_id: str):
        """Get long-term memory for a user."""
        return self.long_term.get(user_id, {})

    def update_long_term(self, user_id: str, key: str, value: Any):
        """Update long-term memory for a user."""
        if user_id not in self.long_term:
            self.long_term[user_id] = {}
        self.long_term[user_id][key] = value
