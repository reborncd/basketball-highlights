"""
detection_page.py
第①页：导入视频 + 自动检测进球 + 手动补充/删除
"""
import threading
import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QListWidget, QListWidgetItem, QDoubleSpinBox,
    QGroupBox, QFormLayout, QSplitter, QSpinBox, QMessageBox,
    QInputDialog, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QColor

from app.core.project import Project, GoalClip
from app.core.detector import DetectionConfig, run_detection
from app.core.clipper import extract_thumbnail, get_video_info


class DetectionPage(QWidget):
    detection_finished = pyqtSignal(object)   # 发出 Project
    _detection_done_signal = pyqtSignal(object, object)  # 用于在主线程中更新UI
    _detection_error_signal = pyqtSignal(str)  # 用于在主线程中显示错误
    _log_signal = pyqtSignal(str)  # 用于在主线程中添加日志

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Project | None = None
        self._cancel_flag = threading.Event()
        self._detection_thread: threading.Thread | None = None
        self._progress_val = 0
        self._progress_total = 1
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_progress_ui)
        self._timer.start(200)
        self._detection_done_signal.connect(self._on_detection_done_main_thread)
        self._detection_error_signal.connect(self._on_detection_error_main_thread)
        self._log_signal.connect(self._add_log_message)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        # ── 顶部：视频信息 ───────────────────────────────────
        info_row = QHBoxLayout()
        self.lbl_video = QLabel("未加载视频")
        self.lbl_video.setStyleSheet("font-size:13px; color:#555;")
        info_row.addWidget(self.lbl_video)
        
        self.btn_select_video = QPushButton("📁 选择视频")
        self.btn_select_video.setStyleSheet(
            "QPushButton{background:#27AE60;color:white;border-radius:5px;padding:4px 12px;}"
            "QPushButton:hover{background:#2ECC71;}"
        )
        self.btn_select_video.clicked.connect(self.select_video)
        info_row.addWidget(self.btn_select_video)
        
        info_row.addStretch()
        root.addLayout(info_row)

        # ── 参数设置 ─────────────────────────────────────────
        cfg_box = QGroupBox("检测参数")
        cfg_form = QFormLayout(cfg_box)

        self.spin_pre = QDoubleSpinBox()
        self.spin_pre.setRange(1, 15)
        self.spin_pre.setValue(5)
        self.spin_pre.setSuffix(" 秒")
        cfg_form.addRow("进球前保留:", self.spin_pre)

        self.spin_post = QDoubleSpinBox()
        self.spin_post.setRange(1, 10)
        self.spin_post.setValue(3)
        self.spin_post.setSuffix(" 秒")
        cfg_form.addRow("进球后保留:", self.spin_post)

        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(1, 5)
        self.spin_sample.setValue(2)
        self.spin_sample.setSuffix(" (每N帧处理1次)")
        cfg_form.addRow("跳帧加速:", self.spin_sample)

        root.addWidget(cfg_box)

        # ── 检测按钮 + 进度 ──────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_detect = QPushButton("🔍  开始自动检测")
        self.btn_detect.setFixedHeight(40)
        self.btn_detect.setStyleSheet(
            "QPushButton{background:#2980B9;color:white;border-radius:6px;font-size:14px;}"
            "QPushButton:hover{background:#3498DB;}"
            "QPushButton:disabled{background:#aaa;}"
        )
        self.btn_detect.clicked.connect(self.start_detection)
        btn_row.addWidget(self.btn_detect)

        self.btn_cancel = QPushButton("停止")
        self.btn_cancel.setFixedHeight(40)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_detection)
        btn_row.addWidget(self.btn_cancel)
        root.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        root.addWidget(self.progress)

        self.lbl_status = QLabel("等待开始…")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.lbl_status)

        # ── 分隔线 ──────────────────────────────────────────
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(line)

        # ── 日志展示区域 ────────────────────────────────────
        log_group = QGroupBox("检测日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_list = QListWidget()
        self.log_list.setAlternatingRowColors(True)
        self.log_list.setStyleSheet("font-size:12px;")
        log_layout.addWidget(self.log_list)
        root.addWidget(log_group)

        # ── 分隔线 ──────────────────────────────────────────
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(line)

        # ── 进球列表 + 手动操作 ──────────────────────────────
        list_label = QLabel("进球片段列表（可手动添加/删除）")
        list_label.setStyleSheet("font-weight:bold;")
        root.addWidget(list_label)

        self.clip_list = QListWidget()
        self.clip_list.setAlternatingRowColors(True)
        root.addWidget(self.clip_list)

        manual_row = QHBoxLayout()
        self.btn_add_manual = QPushButton("＋ 手动添加时间戳")
        self.btn_add_manual.clicked.connect(self.add_manual_clip)
        manual_row.addWidget(self.btn_add_manual)

        self.btn_delete = QPushButton("✕ 删除选中")
        self.btn_delete.clicked.connect(self.delete_selected)
        manual_row.addWidget(self.btn_delete)

        manual_row.addStretch()

        self.btn_next = QPushButton("下一步：分配球员 ▶")
        self.btn_next.setFixedHeight(36)
        self.btn_next.setStyleSheet(
            "QPushButton{background:#27AE60;color:white;border-radius:6px;font-size:13px;}"
            "QPushButton:hover{background:#2ECC71;}"
            "QPushButton:disabled{background:#aaa;}"
        )
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.finish_detection)
        manual_row.addWidget(self.btn_next)

        root.addLayout(manual_row)

    # ── 外部调用 ─────────────────────────────────────────────
    def load_project(self, project: Project):
        self.project = project
        info = get_video_info(project.video_path)
        name = os.path.basename(project.video_path)
        if info:
            dur = info.get("duration", 0)
            fps = info.get("fps", 0)
            w, h = info.get("width", 0), info.get("height", 0)
            self.lbl_video.setText(
                f"{name}  |  {dur:.0f}s  |  {fps:.1f}fps  |  {w}×{h}"
            )
        else:
            self.lbl_video.setText(name)
        self._refresh_list()

    # ── 检测逻辑 ─────────────────────────────────────────────
    def start_detection(self):
        if not self.project:
            QMessageBox.warning(self, "提示", "请先新建项目并选择视频")
            return
        self._cancel_flag.clear()
        self.btn_detect.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.lbl_status.setText("正在检测…")
        self.log_list.clear()  # 清空日志

        config = DetectionConfig(
            pre_roll=self.spin_pre.value(),
            post_roll=self.spin_post.value(),
            sample_every_n=self.spin_sample.value(),
        )

        def worker():
            try:
                # 创建一个自定义的Handler来直接发送日志
                import logging
                class QtLogHandler(logging.Handler):
                    def __init__(self, signal):
                        super().__init__()
                        self.signal = signal
                    def emit(self, record):
                        msg = self.format(record)
                        self.signal.emit(msg)
                
                # 获取detector模块的logger
                detector_logger = logging.getLogger('app.core.detector')
                # 添加自定义handler
                qt_handler = QtLogHandler(self._log_signal)
                qt_handler.setLevel(logging.INFO)
                formatter = logging.Formatter('%(asctime)s - %(message)s')
                qt_handler.setFormatter(formatter)
                detector_logger.addHandler(qt_handler)
                
                timestamps = run_detection(
                    self.project.video_path,
                    config,
                    progress_callback=self._on_progress,
                    cancel_flag=self._cancel_flag,
                )
                
                # 移除handler
                detector_logger.removeHandler(qt_handler)
                
                self._on_detection_done(timestamps, config)
            except Exception as e:
                self._on_detection_error(str(e))

        self._detection_thread = threading.Thread(target=worker, daemon=True)
        self._detection_thread.start()

    def _add_log_message(self, message):
        """添加日志消息到日志列表"""
        item = QListWidgetItem(message)
        # 根据消息内容设置不同的颜色
        if "✅ 进球检测" in message:
            item.setForeground(QColor(46, 204, 113))  # 绿色
        elif "⚠️ 进球检测" in message:
            item.setForeground(QColor(241, 196, 15))  # 黄色
        elif "检测完成" in message:
            item.setForeground(QColor(52, 152, 219))  # 蓝色
        self.log_list.addItem(item)
        # 自动滚动到底部
        self.log_list.scrollToBottom()

    def cancel_detection(self):
        self._cancel_flag.set()
        self.lbl_status.setText("正在停止…")

    def _on_progress(self, current, total):
        self._progress_val = current
        self._progress_total = total

    def _update_progress_ui(self):
        if self._progress_total > 0:
            pct = int(self._progress_val / self._progress_total * 100)
            self.progress.setValue(pct)
            self.lbl_status.setText(
                f"分析帧 {self._progress_val} / {self._progress_total}  ({pct}%)"
            )

    def _on_detection_done(self, timestamps: list[float], config: DetectionConfig):
        print(f"_on_detection_done 被调用，找到 {len(timestamps)} 个进球")
        # 通过信号在主线程中更新UI
        self._detection_done_signal.emit(timestamps, config)

    def _on_detection_done_main_thread(self, timestamps: list[float], config: DetectionConfig):
        print(f"_on_detection_done_main_thread 被调用，找到 {len(timestamps)} 个进球")
        # 在主线程中更新UI
        self.btn_detect.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(100)
        # 清空旧片段，重新添加
        self.project.clips.clear()
        for ts in timestamps:
            self.project.add_goal(ts, config.pre_roll, config.post_roll)
        self.project.detection_done = True
        self.project.save()
        self._refresh_list()
        self.lbl_status.setText(f"✅ 检测完成，共找到 {len(timestamps)} 个进球")
        self.btn_next.setEnabled(True)

    def _on_detection_error(self, msg: str):
        print(f"_on_detection_error 被调用，错误信息: {msg}")
        # 通过信号在主线程中更新UI
        self._detection_error_signal.emit(msg)

    def _on_detection_error_main_thread(self, msg: str):
        print(f"_on_detection_error_main_thread 被调用，错误信息: {msg}")
        # 在主线程中更新UI
        self.btn_detect.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText(f"❌ 检测出错: {msg}")

    # ── 手动操作 ─────────────────────────────────────────────
    def add_manual_clip(self):
        if not self.project:
            return
        val, ok = QInputDialog.getDouble(
            self, "手动添加进球", "输入进球时间戳（秒）：",
            0, 0, 99999, 1
        )
        if ok:
            config = DetectionConfig(
                pre_roll=self.spin_pre.value(),
                post_roll=self.spin_post.value(),
            )
            clip = self.project.add_goal(val, config.pre_roll, config.post_roll)
            clip.confidence = "manual"
            self.project.save()
            self._refresh_list()
            self.btn_next.setEnabled(True)

    def delete_selected(self):
        if not self.project:
            return
        selected = self.clip_list.selectedItems()
        if not selected:
            return
        clip_ids = [item.data(Qt.ItemDataRole.UserRole) for item in selected]
        self.project.clips = [c for c in self.project.clips if c.clip_id not in clip_ids]
        self.project.save()
        self._refresh_list()

    def select_video(self):
        from PyQt5.QtWidgets import QFileDialog
        import os
        
        video_path, _ = QFileDialog.getOpenFileName(
            self, "选择篮球录像文件", "",
            "视频文件 (*.mp4 *.mov *.avi *.mkv *.m4v);;所有文件 (*)"
        )
        if not video_path:
            return
        
        # 如果没有项目，创建一个新项目，保存到视频所在目录
        if not self.project:
            from app.core.project import Project
            # 获取视频所在目录
            video_dir = os.path.dirname(video_path)
            # 使用视频文件名作为项目名称
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            project_dir = os.path.join(video_dir, f"{video_name}_project")
            # 确保项目目录存在
            os.makedirs(project_dir, exist_ok=True)
            
            self.project = Project(video_path=video_path, project_dir=project_dir)
            self.project.save()
            
            # 通知主窗口更新项目
            if hasattr(self.parent(), 'project'):
                self.parent().project = self.project
        else:
            self.project.video_path = video_path
            self.project.save()
        
        # 加载视频信息
        self.load_project(self.project)

    def finish_detection(self):
        if not self.project or not self.project.clips:
            QMessageBox.warning(self, "提示", "没有进球片段，请先检测或手动添加")
            return
        self.detection_finished.emit(self.project)

    # ── 列表刷新 ─────────────────────────────────────────────
    def _refresh_list(self):
        self.clip_list.clear()
        if not self.project:
            return
        for clip in self.project.clips:
            tag = "🤖" if clip.confidence == "auto" else "✋"
            label = (
                f"{tag}  {clip.clip_id}  |  "
                f"时间戳 {clip.timestamp:.1f}s  |  "
                f"片段 {clip.start_sec:.1f}s ~ {clip.end_sec:.1f}s"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, clip.clip_id)
            self.clip_list.addItem(item)
        if self.project.clips:
            self.btn_next.setEnabled(True)
