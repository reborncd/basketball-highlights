"""
review_page.py
第②页：复核进球片段 + 分配球员
左侧：片段列表（含缩略图）
右侧：片段预览 + 球员按钮组
"""
import os
import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QSplitter, QGroupBox,
    QInputDialog, QMessageBox, QScrollArea, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QPixmap, QColor, QFont, QIcon

from app.core.project import Project, GoalClip, Player
from app.core.clipper import extract_thumbnail


THUMB_W = 160
THUMB_H = 90


class ClipCard(QListWidgetItem):
    """列表中一行：缩略图 + 基本信息"""
    def __init__(self, clip: GoalClip):
        super().__init__()
        self.clip_id = clip.clip_id
        self._update(clip)

    def _update(self, clip: GoalClip):
        player = clip.player_name or "（未分配）"
        confirmed = "✅" if clip.confirmed else "⬜"
        self.setText(
            f"{confirmed}  {clip.clip_id}\n"
            f"  📍 {clip.timestamp:.1f}s  👤 {player}"
        )
        if clip.player_name:
            self.setBackground(QColor("#E8F5E9"))
        else:
            self.setBackground(QColor("#FAFAFA"))


class ReviewPage(QWidget):
    review_done = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Project | None = None
        self._current_clip: GoalClip | None = None
        self._player_buttons: list[QPushButton] = []
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── 左侧：片段列表 ───────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("进球片段")
        lbl.setStyleSheet("font-weight:bold; font-size:13px;")
        left_layout.addWidget(lbl)

        self.clip_list = QListWidget()
        self.clip_list.setIconSize(QSize(THUMB_W, THUMB_H))
        self.clip_list.setSpacing(4)
        self.clip_list.currentItemChanged.connect(self._on_clip_selected)
        left_layout.addWidget(self.clip_list)

        splitter.addWidget(left)
        splitter.setStretchFactor(0, 1)

        # ── 右侧：预览 + 操作 ────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)

        # 缩略图预览
        self.thumb_label = QLabel("← 从左侧选择一个片段")
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setFixedHeight(240)
        self.thumb_label.setStyleSheet(
            "background:#1a1a2e; border-radius:8px; color:#aaa; font-size:13px;"
        )
        right_layout.addWidget(self.thumb_label)

        # 时间信息
        self.lbl_clip_info = QLabel("")
        self.lbl_clip_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_clip_info.setStyleSheet("color:#555; font-size:12px;")
        right_layout.addWidget(self.lbl_clip_info)

        # 球员选择
        player_group = QGroupBox("分配给球员")
        self.player_btn_layout = QHBoxLayout(player_group)
        self.player_btn_layout.setSpacing(6)
        right_layout.addWidget(player_group)

        # 新建球员
        btn_add_player = QPushButton("＋ 新建球员")
        btn_add_player.setStyleSheet(
            "QPushButton{background:#8E44AD;color:white;border-radius:5px;padding:6px 12px;}"
            "QPushButton:hover{background:#9B59B6;}"
        )
        btn_add_player.clicked.connect(self.add_new_player)
        right_layout.addWidget(btn_add_player, alignment=Qt.AlignmentFlag.AlignLeft)

        # 取消分配
        btn_unassign = QPushButton("✕ 取消分配")
        btn_unassign.setStyleSheet(
            "QPushButton{background:#E74C3C;color:white;border-radius:5px;padding:6px 12px;}"
            "QPushButton:hover{background:#C0392B;}"
        )
        btn_unassign.clicked.connect(self.unassign_clip)
        right_layout.addWidget(btn_unassign, alignment=Qt.AlignmentFlag.AlignLeft)

        right_layout.addStretch()

        # 统计信息
        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet("color:#555; font-size:12px;")
        right_layout.addWidget(self.lbl_stats)

        # 下一步按钮
        self.btn_next = QPushButton("下一步：导出集锦 ▶")
        self.btn_next.setFixedHeight(40)
        self.btn_next.setStyleSheet(
            "QPushButton{background:#27AE60;color:white;border-radius:6px;font-size:14px;}"
            "QPushButton:hover{background:#2ECC71;}"
            "QPushButton:disabled{background:#aaa;}"
        )
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._finish_review)
        right_layout.addWidget(self.btn_next)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 2)

        root.addWidget(splitter)

    # ── 外部调用 ─────────────────────────────────────────────
    def load_project(self, project: Project):
        self.project = project
        self._refresh_clip_list()
        self._refresh_player_buttons()
        self._refresh_stats()
        # 后台生成所有缩略图
        threading.Thread(target=self._generate_thumbnails, daemon=True).start()

    def _generate_thumbnails(self):
        if not self.project:
            return
        for clip in self.project.clips:
            if clip.thumbnail_path and os.path.exists(clip.thumbnail_path):
                continue
            thumb_path = os.path.join(
                self.project.thumbnails_dir(), f"{clip.clip_id}.jpg"
            )
            ok = extract_thumbnail(
                self.project.video_path,
                clip.timestamp,
                thumb_path,
            )
            if ok:
                clip.thumbnail_path = thumb_path
        QTimer.singleShot(0, self._refresh_clip_list)
        self.project.save()

    # ── 列表刷新 ─────────────────────────────────────────────
    def _refresh_clip_list(self):
        if not self.project:
            return
        self.clip_list.clear()
        for clip in self.project.clips:
            item = ClipCard(clip)
            item.setData(Qt.ItemDataRole.UserRole, clip.clip_id)
            # 设置缩略图
            if clip.thumbnail_path and os.path.exists(clip.thumbnail_path):
                icon = QIcon(QPixmap(clip.thumbnail_path).scaled(
                    THUMB_W, THUMB_H,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
                item.setIcon(icon)
            self.clip_list.addItem(item)

    def _refresh_player_buttons(self):
        # 清除旧按钮
        for btn in self._player_buttons:
            btn.deleteLater()
        self._player_buttons.clear()
        if not self.project:
            return
        for player in self.project.players:
            btn = QPushButton(player.name)
            btn.setFixedHeight(34)
            btn.setStyleSheet(
                f"QPushButton{{background:{player.color};color:white;"
                f"border-radius:5px;padding:4px 10px;font-size:13px;}}"
                f"QPushButton:hover{{opacity:0.85;}}"
            )
            btn.clicked.connect(lambda checked, p=player: self._assign_player(p.name))
            self._player_buttons.append(btn)
            self.player_btn_layout.addWidget(btn)

    def _refresh_stats(self):
        if not self.project:
            return
        total = len(self.project.clips)
        assigned = sum(1 for c in self.project.clips if c.player_name)
        unassigned = total - assigned
        self.lbl_stats.setText(
            f"共 {total} 个片段 | ✅ 已分配 {assigned} | ⬜ 未分配 {unassigned}"
        )
        self.btn_next.setEnabled(assigned > 0)

    # ── 事件处理 ─────────────────────────────────────────────
    def _on_clip_selected(self, current, previous):
        if not current or not self.project:
            return
        clip_id = current.data(Qt.ItemDataRole.UserRole)
        clip = next((c for c in self.project.clips if c.clip_id == clip_id), None)
        if not clip:
            return
        self._current_clip = clip
        self.lbl_clip_info.setText(
            f"{clip.clip_id}  |  进球 @ {clip.timestamp:.1f}s  |  "
            f"片段 {clip.start_sec:.1f}s ~ {clip.end_sec:.1f}s  |  "
            f"球员: {clip.player_name or '未分配'}"
        )
        # 显示缩略图
        if clip.thumbnail_path and os.path.exists(clip.thumbnail_path):
            pix = QPixmap(clip.thumbnail_path).scaled(
                self.thumb_label.width(), self.thumb_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.thumb_label.setPixmap(pix)
        else:
            self.thumb_label.setText("缩略图生成中…")
            self.thumb_label.setPixmap(QPixmap())

    def _assign_player(self, player_name: str):
        if not self._current_clip or not self.project:
            return
        self.project.assign_player(self._current_clip.clip_id, player_name)
        self.project.save()
        self._refresh_clip_list()
        self._refresh_stats()
        # 自动跳到下一个未分配片段
        self._goto_next_unassigned()

    def unassign_clip(self):
        if not self._current_clip or not self.project:
            return
        self._current_clip.player_name = None
        self._current_clip.confirmed = False
        self.project.save()
        self._refresh_clip_list()
        self._refresh_stats()

    def _goto_next_unassigned(self):
        for i in range(self.clip_list.count()):
            item = self.clip_list.item(i)
            clip_id = item.data(Qt.ItemDataRole.UserRole)
            clip = next((c for c in self.project.clips if c.clip_id == clip_id), None)
            if clip and not clip.player_name:
                self.clip_list.setCurrentItem(item)
                return

    def add_new_player(self):
        if not self.project:
            return
        name, ok = QInputDialog.getText(self, "新建球员", "请输入球员名称（如：詹姆斯 / 23号）：")
        if ok and name.strip():
            self.project.add_player(name.strip())
            self.project.save()
            self._refresh_player_buttons()

    def _finish_review(self):
        if not self.project:
            return
        assigned = sum(1 for c in self.project.clips if c.player_name)
        if assigned == 0:
            QMessageBox.warning(self, "提示", "还没有给任何片段分配球员")
            return
        self.review_done.emit()
