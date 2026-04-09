"""
export_page.py
第③页：裁剪片段 + 合并集锦 + 显示输出目录
"""
import os
import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QListWidget, QListWidgetItem, QGroupBox,
    QMessageBox, QFileDialog, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor

from app.core.project import Project, Player
from app.core.clipper import clip_segment, concat_clips


class ExportPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Project | None = None
        self._cancel_flag = threading.Event()
        self._progress_val = 0
        self._progress_total = 1
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_progress_ui)
        self._timer.start(300)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        # ── 标题 ────────────────────────────────────────────
        title = QLabel("导出每位球员的进球集锦")
        title.setStyleSheet("font-size:16px; font-weight:bold;")
        root.addWidget(title)

        # ── 球员列表 ─────────────────────────────────────────
        self.player_list = QListWidget()
        self.player_list.setAlternatingRowColors(True)
        self.player_list.setSpacing(2)
        root.addWidget(self.player_list)

        # ── 输出目录 ─────────────────────────────────────────
        dir_row = QHBoxLayout()
        self.lbl_output_dir = QLabel("输出目录：—")
        self.lbl_output_dir.setStyleSheet("color:#555; font-size:12px;")
        dir_row.addWidget(self.lbl_output_dir)
        btn_open_dir = QPushButton("📂 打开目录")
        btn_open_dir.clicked.connect(self._open_output_dir)
        dir_row.addWidget(btn_open_dir)
        root.addLayout(dir_row)

        # ── 分隔线 ──────────────────────────────────────────
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(line)

        # ── 进度 ─────────────────────────────────────────────
        self.lbl_status = QLabel("点击下方按钮开始导出")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.lbl_status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        # ── 操作按钮 ─────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.btn_export_all = QPushButton("🎬  一键导出所有球员集锦")
        self.btn_export_all.setFixedHeight(44)
        self.btn_export_all.setStyleSheet(
            "QPushButton{background:#2980B9;color:white;border-radius:6px;font-size:15px;}"
            "QPushButton:hover{background:#3498DB;}"
            "QPushButton:disabled{background:#aaa;}"
        )
        self.btn_export_all.clicked.connect(self.export_all)
        btn_row.addWidget(self.btn_export_all)

        self.btn_export_selected = QPushButton("导出选中球员")
        self.btn_export_selected.setFixedHeight(44)
        self.btn_export_selected.setStyleSheet(
            "QPushButton{background:#27AE60;color:white;border-radius:6px;font-size:14px;}"
            "QPushButton:hover{background:#2ECC71;}"
            "QPushButton:disabled{background:#aaa;}"
        )
        self.btn_export_selected.clicked.connect(self.export_selected)
        btn_row.addWidget(self.btn_export_selected)

        self.btn_cancel = QPushButton("停止")
        self.btn_cancel.setFixedHeight(44)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_export)
        btn_row.addWidget(self.btn_cancel)

        root.addLayout(btn_row)

    # ── 外部调用 ─────────────────────────────────────────────
    def load_project(self, project: Project):
        self.project = project
        output_dir = os.path.join(project.project_dir, "output")
        self.lbl_output_dir.setText(f"输出目录：{output_dir}")
        self._refresh_player_list()

    def _refresh_player_list(self):
        self.player_list.clear()
        if not self.project:
            return
        for player in self.project.players:
            clips = self.project.get_clips_for_player(player.name)
            status = ""
            if player.highlight_path and os.path.exists(player.highlight_path):
                status = "✅ 已导出"
            item = QListWidgetItem(
                f"👤  {player.name}    |    {len(clips)} 个进球    {status}"
            )
            item.setData(Qt.ItemDataRole.UserRole, player.name)
            if status:
                item.setBackground(QColor("#E8F5E9"))
            self.player_list.addItem(item)

    # ── 导出逻辑 ─────────────────────────────────────────────
    def export_all(self):
        if not self.project:
            return
        players = self.project.players
        if not players:
            QMessageBox.warning(self, "提示", "没有球员数据，请先在第②页分配球员")
            return
        self._start_export(players)

    def export_selected(self):
        if not self.project:
            return
        selected = self.player_list.selectedItems()
        if not selected:
            QMessageBox.warning(self, "提示", "请先在上方列表选中要导出的球员")
            return
        names = [item.data(Qt.ItemDataRole.UserRole) for item in selected]
        players = [p for p in self.project.players if p.name in names]
        self._start_export(players)

    def cancel_export(self):
        self._cancel_flag.set()
        self.lbl_status.setText("正在停止…")

    def _start_export(self, players: list[Player]):
        self._cancel_flag.clear()
        self.btn_export_all.setEnabled(False)
        self.btn_export_selected.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)

        def worker():
            self._do_export(players)

        threading.Thread(target=worker, daemon=True).start()

    def _do_export(self, players: list[Player]):
        project = self.project
        total_steps = sum(
            len(project.get_clips_for_player(p.name)) + 1
            for p in players
        )
        self._progress_total = max(1, total_steps)
        self._progress_val = 0

        for player in players:
            if self._cancel_flag.is_set():
                break

            clips = project.get_clips_for_player(player.name)
            if not clips:
                continue

            player_dir = os.path.join(project.project_dir, "output", player.name)
            os.makedirs(player_dir, exist_ok=True)

            # Step 1: 裁剪每个片段
            clip_files = []
            for clip in clips:
                if self._cancel_flag.is_set():
                    break
                out_file = os.path.join(player_dir, f"{clip.clip_id}.mp4")
                self._set_status(f"裁剪 {player.name} / {clip.clip_id}…")
                ok = clip_segment(
                    project.video_path,
                    clip.start_sec,
                    clip.end_sec,
                    out_file,
                )
                if ok:
                    clip.clip_path = out_file
                    clip_files.append(out_file)
                self._progress_val += 1

            # Step 2: 合并集锦
            if clip_files:
                highlight_path = os.path.join(player_dir, f"{player.name}_集锦.mp4")
                self._set_status(f"合并 {player.name} 集锦…")
                ok = concat_clips(clip_files, highlight_path)
                if ok:
                    player.highlight_path = highlight_path
            self._progress_val += 1

        project.save()

        def finish():
            self.btn_export_all.setEnabled(True)
            self.btn_export_selected.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self.progress.setValue(100)
            self._refresh_player_list()
            done_players = [p.name for p in players if p.highlight_path]
            self.lbl_status.setText(
                f"✅ 导出完成：{', '.join(done_players)}"
            )
            QMessageBox.information(
                self, "完成",
                f"集锦已导出到:\n{os.path.join(project.project_dir, 'output')}"
            )

        QTimer.singleShot(0, finish)

    def _set_status(self, msg: str):
        QTimer.singleShot(0, lambda: self.lbl_status.setText(msg))

    def _update_progress_ui(self):
        if self._progress_total > 0:
            pct = int(self._progress_val / self._progress_total * 100)
            self.progress.setValue(pct)

    def _open_output_dir(self):
        if not self.project:
            return
        output_dir = os.path.join(self.project.project_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        import subprocess, sys
        if sys.platform == "darwin":
            subprocess.Popen(["open", output_dir])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", output_dir])
        else:
            subprocess.Popen(["xdg-open", output_dir])
