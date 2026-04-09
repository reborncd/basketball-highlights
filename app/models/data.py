from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass
class GoalEvent:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = 0.0          # seconds in source video
    confidence: float = 0.0         # detection confidence 0-1
    video_source: str = ""          # source video path
    clip_path: Optional[str] = None # path to extracted clip
    player_id: Optional[str] = None # assigned player id
    confirmed: bool = True          # user confirmed this is a real goal

    @property
    def timestamp_str(self) -> str:
        m = int(self.timestamp // 60)
        s = int(self.timestamp % 60)
        return f"{m:02d}:{s:02d}"


@dataclass
class Player:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "未命名球员"
    goal_ids: list = field(default_factory=list)

    @property
    def goal_count(self) -> int:
        return len(self.goal_ids)
