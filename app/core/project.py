"""
project.py
项目状态管理 — 保存/加载进球时间戳、球员分配、片段路径
"""
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

PROJECT_FILE = "project.json"


@dataclass
class GoalClip:
    """单个进球片段"""
    clip_id: str                  # 唯一 ID，如 "clip_001"
    timestamp: float              # 进球时间戳（秒）
    start_sec: float              # 片段开始时间
    end_sec: float                # 片段结束时间
    player_name: Optional[str] = None   # 分配的球员名
    clip_path: Optional[str] = None     # 已裁剪片段的文件路径
    thumbnail_path: Optional[str] = None
    confidence: str = "auto"     # "auto" | "manual"
    confirmed: bool = False       # 用户是否已确认


@dataclass
class Player:
    name: str
    directory: Optional[str] = None    # 输出目录
    highlight_path: Optional[str] = None  # 合并后集锦路径
    color: str = "#4A90D9"        # UI 显示颜色


@dataclass
class Project:
    video_path: str = ""
    project_dir: str = ""
    clips: list[GoalClip] = field(default_factory=list)
    players: list[Player] = field(default_factory=list)
    detection_done: bool = False
    manual_hoop_rect: Optional[tuple[int, int, int, int]] = None
    last_detection_stats: dict = field(default_factory=dict)
    last_detection_failure_reason: Optional[str] = None
    last_detection_config: dict = field(default_factory=dict)

    # ── 持久化 ──────────────────────────────────────────────
    def save(self):
        path = os.path.join(self.project_dir, PROJECT_FILE)
        data = {
            "video_path": self.video_path,
            "project_dir": self.project_dir,
            "detection_done": self.detection_done,
            "manual_hoop_rect": list(self.manual_hoop_rect) if self.manual_hoop_rect else None,
            "last_detection_stats": self.last_detection_stats,
            "last_detection_failure_reason": self.last_detection_failure_reason,
            "last_detection_config": self.last_detection_config,
            "clips": [asdict(c) for c in self.clips],
            "players": [asdict(p) for p in self.players],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"项目已保存: {path}")

    @classmethod
    def load(cls, project_dir: str) -> "Project":
        path = os.path.join(project_dir, PROJECT_FILE)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        proj = cls(
            video_path=data["video_path"],
            project_dir=data["project_dir"],
            detection_done=data.get("detection_done", False),
            manual_hoop_rect=tuple(data["manual_hoop_rect"]) if data.get("manual_hoop_rect") else None,
            last_detection_stats=data.get("last_detection_stats", {}),
            last_detection_failure_reason=data.get("last_detection_failure_reason"),
            last_detection_config=data.get("last_detection_config", {}),
        )
        proj.clips = [GoalClip(**c) for c in data.get("clips", [])]
        proj.players = [Player(**p) for p in data.get("players", [])]
        logger.info(f"项目已加载: {path}，{len(proj.clips)} 个片段")
        return proj

    # ── 工具方法 ─────────────────────────────────────────────
    def add_goal(self, timestamp: float, pre_roll: float, post_roll: float):
        idx = len(self.clips) + 1
        clip_id = f"clip_{idx:03d}"
        clip = GoalClip(
            clip_id=clip_id,
            timestamp=timestamp,
            start_sec=max(0, timestamp - pre_roll),
            end_sec=timestamp + post_roll,
        )
        self.clips.append(clip)
        return clip

    def add_player(self, name: str) -> Player:
        # 防重名
        existing = [p.name for p in self.players]
        if name in existing:
            return next(p for p in self.players if p.name == name)
        colors = ["#4A90D9", "#E67E22", "#27AE60", "#9B59B6",
                  "#E74C3C", "#1ABC9C", "#F39C12", "#2980B9"]
        color = colors[len(self.players) % len(colors)]
        # 默认在当前项目下生成output文件夹
        player_dir = os.path.join(self.project_dir, "output", name)
        p = Player(name=name, directory=player_dir, color=color)
        self.players.append(p)
        # 确保output目录存在
        os.makedirs(os.path.join(self.project_dir, "output"), exist_ok=True)
        os.makedirs(player_dir, exist_ok=True)
        return p

    def assign_player(self, clip_id: str, player_name: str):
        for clip in self.clips:
            if clip.clip_id == clip_id:
                clip.player_name = player_name
                clip.confirmed = True
                return

    def get_clips_for_player(self, player_name: str) -> list[GoalClip]:
        return [c for c in self.clips if c.player_name == player_name]

    def unassigned_clips(self) -> list[GoalClip]:
        return [c for c in self.clips if not c.player_name]

    def clips_dir(self) -> str:
        d = os.path.join(self.project_dir, "clips")
        os.makedirs(d, exist_ok=True)
        return d

    def thumbnails_dir(self) -> str:
        d = os.path.join(self.project_dir, "thumbnails")
        os.makedirs(d, exist_ok=True)
        return d
