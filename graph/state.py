from dataclasses import dataclass
from typing import Any, Optional

from core.types import UserRequest, QueryContext, Intent


@dataclass
class AgentState:
    request: UserRequest
    user_id: str
    session_id: str
    db: Any = None
    ctx: Optional[QueryContext] = None
    intent: Optional[Intent] = None
    active_ctx: Any = None
    session_ctx: Any = None
    target_files: Any = None
    k: int = 10
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    rewritten_query: Optional[str] = None
