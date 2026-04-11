"""
detection_page.py
第①页：导入视频 + 自动检测进球 + 手动补充/删除
"""
import os
import threading

from PyQt5.QtCore import QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.clipper import get_video_info
from app.core.detector import (
    DetectionConfig,
    DetectionRunResult,
    describe_failure_reason,
    run_detection,
)
from app.core.project import Project


class RectSelectionLabel(QLabel):
    """在静态预览图上框选矩形区域"""

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self._base_pixmap = pixmap
        self._selection = QRect()
        self._drag_start: QPoint | None = None
        self.setFixedSize(self._base_pixmap.size())
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._update_preview()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.pos()
            self._selection = QRect(self._drag_start, self._drag_start)
            self._update_preview()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is not None:
            self._selection = QRect(self._drag_start, event.pos()).normalized()
            self._update_preview()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            self._selection = QRect(self._drag_start, event.pos()).normalized()
            self._drag_start = None
            self._update_preview()
        super().mouseReleaseEvent(event)

    def set_selection(self, rect: QRect):
        self._selection = rect.normalized()
        self._update_preview()

    def clear_selection(self):
        self._selection = QRect()
        self._update_preview()

    def selected_rect(self) -> QRect:
        return self._selection.normalized()

    def has_selection(self) -> bool:
        rect = self.selected_rect()
        return rect.width() > 4 and rect.height() > 4

    def _update_preview(self):
        preview = QPixmap(self._base_pixmap)
        if self.has_selection():
            painter = QPainter(preview)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor("#27AE60"), 2))
            painter.fillRect(self._selection, QColor(39, 174, 96, 70))
            painter.drawRect(self._selection)
            painter.end()
        self.setPixmap(preview)


class HoopCalibrationDialog(QDialog):
    """基于首帧手动框选篮筐"""

    def __init__(self, image: QImage, initial_rect=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("手动标定篮筐")
        self._image = image
        self._initial_rect = initial_rect
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("请在首帧上框出篮筐区域，尽量只圈住篮筐和少量边缘。"))

        preview = self._build_preview_pixmap()
        self._scale_x = self._image.width() / max(1, preview.width())
        self._scale_y = self._image.height() / max(1, preview.height())
        self.selector = RectSelectionLabel(preview, self)
        layout.addWidget(self.selector, alignment=Qt.AlignmentFlag.AlignCenter)

        if self._initial_rect:
            x, y, w, h = self._initial_rect
            scaled_rect = QRect(
                int(x / self._scale_x),
                int(y / self._scale_y),
                int(w / self._scale_x),
                int(h / self._scale_y),
            )
            self.selector.set_selection(scaled_rect)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        clear_btn = QPushButton("清除选择")
        clear_btn.clicked.connect(self.selector.clear_selection)
        buttons.addButton(clear_btn, QDialogButtonBox.ButtonRole.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_preview_pixmap(self) -> QPixmap:
        max_w = 960
        max_h = 540
        preview = self._image
        if self._image.width() > max_w or self._image.height() > max_h:
            preview = self._image.scaled(
                max_w,
                max_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        return QPixmap.fromImage(preview)

    def accept(self):
        if not self.selector.has_selection():
            QMessageBox.warning(self, "提示", "请先框选篮筐区域")
            return
        super().accept()

    def selected_rect(self):
        if not self.selector.has_selection():
            return None

        rect = self.selector.selected_rect()
        x = int(round(rect.x() * self._scale_x))
        y = int(round(rect.y() * self._scale_y))
        w = int(round(rect.width() * self._scale_x))
        h = int(round(rect.height() * self._scale_y))
        x = max(0, min(x, self._image.width() - 1))
        y = max(0, min(y, self._image.height() - 1))
        w = max(1, min(w, self._image.width() - x))
        h = max(1, min(h, self._image.height() - y))
        return (x, y, w, h)


class DetectionPage(QWidget):
    detection_finished = pyqtSignal(object)
    _detection_done_signal = pyqtSignal(object, object)
    _detection_error_signal = pyqtSignal(str)
    _log_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Project | None = None
        self._cancel_flag = threading.Event()
        self._detection_thread: threading.Thread | None = None
        self._progress_val = 0
        self._progress_total = 1
        self._is_detecting = False
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
        self.spin_sample.setValue(1)
        self.spin_sample.setSuffix(" (每N帧处理1次)")
        cfg_form.addRow("跳帧加速:", self.spin_sample)

        self.chk_use_yolo = QCheckBox("启用YOLO辅助篮球检测")
        self.chk_use_yolo.setChecked(False)
        cfg_form.addRow("", self.chk_use_yolo)

        self.txt_yolo_model = QLineEdit()
        self.txt_yolo_model.setText("yolo11n.pt")
        self.txt_yolo_model.setPlaceholderText("YOLO模型路径")
        cfg_form.addRow("YOLO模型:", self.txt_yolo_model)

        hoop_row = QWidget()
        hoop_layout = QHBoxLayout(hoop_row)
        hoop_layout.setContentsMargins(0, 0, 0, 0)
        hoop_layout.setSpacing(6)
        self.btn_calibrate_hoop = QPushButton("🎯 标定篮筐")
        self.btn_calibrate_hoop.clicked.connect(self.calibrate_hoop)
        hoop_layout.addWidget(self.btn_calibrate_hoop)
        self.btn_clear_hoop = QPushButton("清除标定")
        self.btn_clear_hoop.clicked.connect(self.clear_manual_hoop)
        hoop_layout.addWidget(self.btn_clear_hoop)
        hoop_layout.addStretch()
        cfg_form.addRow("篮筐标定:", hoop_row)

        self.lbl_hoop_status = QLabel("当前：未手动标定，将使用自动校准")
        self.lbl_hoop_status.setStyleSheet("color:#555; font-size:12px;")
        self.lbl_hoop_status.setWordWrap(True)
        cfg_form.addRow("", self.lbl_hoop_status)

        root.addWidget(cfg_box)

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

        self.lbl_last_summary = QLabel("上次检测：—")
        self.lbl_last_summary.setWordWrap(True)
        self.lbl_last_summary.setStyleSheet("color:#555; font-size:12px;")
        root.addWidget(self.lbl_last_summary)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(line)

        log_group = QGroupBox("检测日志")
        log_layout = QVBoxLayout(log_group)
        self.log_list = QListWidget()
        self.log_list.setAlternatingRowColors(True)
        self.log_list.setStyleSheet("font-size:12px;")
        log_layout.addWidget(self.log_list)
        root.addWidget(log_group)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(line)

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

    def load_project(self, project: Project):
        self.project = project
        info = get_video_info(project.video_path)
        name = os.path.basename(project.video_path)
        if info:
            dur = info.get("duration", 0)
            fps = info.get("fps", 0)
            w, h = info.get("width", 0), info.get("height", 0)
            self.lbl_video.setText(f"{name}  |  {dur:.0f}s  |  {fps:.1f}fps  |  {w}×{h}")
        else:
            self.lbl_video.setText(name)
        self._update_hoop_status()
        self._update_last_summary()
        self._refresh_list()

    def start_detection(self):
        if not self.project:
            QMessageBox.warning(self, "提示", "请先新建项目并选择视频")
            return

        self._cancel_flag.clear()
        self._is_detecting = True
        self.btn_detect.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.lbl_status.setText("正在检测…")
        self.log_list.clear()

        config = DetectionConfig(
            pre_roll=self.spin_pre.value(),
            post_roll=self.spin_post.value(),
            sample_every_n=self.spin_sample.value(),
            use_yolo=self.chk_use_yolo.isChecked(),
            yolo_model_path=self.txt_yolo_model.text().strip() or "yolo11n.pt",
            manual_hoop_rect=self.project.manual_hoop_rect,
        )

        def worker():
            import logging

            class QtLogHandler(logging.Handler):
                def __init__(self, signal):
                    super().__init__()
                    self.signal = signal

                def emit(self, record):
                    self.signal.emit(self.format(record))

            qt_handler = QtLogHandler(self._log_signal)
            qt_handler.setLevel(logging.INFO)
            qt_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            detector_logger = logging.getLogger("app.core.detector")
            yolo_logger = logging.getLogger("app.core.yolo_detector")
            detector_logger.addHandler(qt_handler)
            yolo_logger.addHandler(qt_handler)
            try:
                result = run_detection(
                    self.project.video_path,
                    config,
                    progress_callback=self._on_progress,
                    cancel_flag=self._cancel_flag,
                )
                self._on_detection_done(result, config)
            except Exception as e:
                self._on_detection_error(str(e))
            finally:
                detector_logger.removeHandler(qt_handler)
                yolo_logger.removeHandler(qt_handler)

        self._detection_thread = threading.Thread(target=worker, daemon=True)
        self._detection_thread.start()

    def _add_log_message(self, message: str):
        item = QListWidgetItem(message)
        if "✅ 进球检测" in message or "✅ 篮筐锁定完成" in message:
            item.setForeground(QColor(46, 204, 113))
        elif "⚠️" in message:
            item.setForeground(QColor(241, 196, 15))
        elif "检测摘要" in message:
            item.setForeground(QColor(52, 152, 219))
        self.log_list.addItem(item)
        self.log_list.scrollToBottom()

    def cancel_detection(self):
        self._cancel_flag.set()
        self.lbl_status.setText("正在停止…")

    def _on_progress(self, current, total):
        self._progress_val = current
        self._progress_total = max(1, total)

    def _update_progress_ui(self):
        if not self._is_detecting:
            return
        pct = int(self._progress_val / self._progress_total * 100)
        self.progress.setValue(pct)
        self.lbl_status.setText(f"分析帧 {self._progress_val} / {self._progress_total}  ({pct}%)")

    def _on_detection_done(self, result: DetectionRunResult, config: DetectionConfig):
        self._detection_done_signal.emit(result, config)

    def _on_detection_done_main_thread(self, result: DetectionRunResult, config: DetectionConfig):
        if not self.project:
            return

        self._is_detecting = False
        self.btn_detect.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(100)

        self.project.clips.clear()
        for ts in result.timestamps:
            self.project.add_goal(ts, config.pre_roll, config.post_roll)
        self.project.detection_done = True
        self.project.last_detection_stats = result.stats
        self.project.last_detection_failure_reason = result.failure_reason
        self.project.last_detection_config = result.config_snapshot
        self.project.save()

        self._refresh_list()
        self._update_last_summary()

        if result.failure_reason == "cancelled":
            self.lbl_status.setText("⏹️ 检测已取消")
            self.btn_next.setEnabled(bool(self.project.clips))
            return

        if result.timestamps:
            self.lbl_status.setText(f"✅ 检测完成，共找到 {len(result.timestamps)} 个候选进球")
            self.btn_next.setEnabled(True)
            return

        failure_text = describe_failure_reason(result.failure_reason)
        self.lbl_status.setText(f"⚠️ 未检测到候选进球：{failure_text}")
        self.btn_next.setEnabled(False)
        QMessageBox.information(
            self,
            "未检测到候选进球",
            f"{failure_text}\n\n{result.summary}",
        )

    def _on_detection_error(self, msg: str):
        self._detection_error_signal.emit(msg)

    def _on_detection_error_main_thread(self, msg: str):
        self._is_detecting = False
        self.btn_detect.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText(f"❌ 检测出错: {msg}")

    def _load_preview_image(self):
        if not self.project or not self.project.video_path:
            return None

        import cv2

        cap = cv2.VideoCapture(self.project.video_path)
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        return QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

    def calibrate_hoop(self):
        if not self.project:
            QMessageBox.warning(self, "提示", "请先选择视频")
            return

        image = self._load_preview_image()
        if image is None:
            QMessageBox.warning(self, "提示", "无法读取视频首帧，暂时不能标定篮筐")
            return

        dialog = HoopCalibrationDialog(image, initial_rect=self.project.manual_hoop_rect, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            rect = dialog.selected_rect()
            if rect:
                self.project.manual_hoop_rect = rect
                self.project.save()
                self._update_hoop_status()

    def clear_manual_hoop(self):
        if not self.project:
            return
        self.project.manual_hoop_rect = None
        self.project.save()
        self._update_hoop_status()

    def _update_hoop_status(self):
        if not self.project or not self.project.manual_hoop_rect:
            self.lbl_hoop_status.setText("当前：未手动标定，将使用自动校准")
            return
        x, y, w, h = self.project.manual_hoop_rect
        self.lbl_hoop_status.setText(f"当前：已手动标定篮筐 ({x}, {y}, {w}, {h})")

    def _update_last_summary(self):
        if not self.project or not self.project.last_detection_stats:
            self.lbl_last_summary.setText("上次检测：—")
            return

        stats = self.project.last_detection_stats
        if self.project.last_detection_failure_reason:
            result_text = describe_failure_reason(self.project.last_detection_failure_reason)
        elif stats.get("goal_candidates", 0) > 0:
            result_text = "成功"
        else:
            result_text = "未产出候选"
        sample_every_n = self.project.last_detection_config.get("sample_every_n")
        self.lbl_last_summary.setText(
            "上次检测："
            f"已处理 {stats.get('processed_frames', 0)} 帧，"
            f"篮筐命中 {stats.get('hoop_detected_frames', 0)}，"
            f"篮球命中 {stats.get('ball_detected_frames', 0)}，"
            f"候选进球 {stats.get('goal_candidates', 0)}，"
            f"采样间隔 {sample_every_n or 1}，"
            f"结果 {result_text}"
        )

    def add_manual_clip(self):
        if not self.project:
            return
        val, ok = QInputDialog.getDouble(
            self,
            "手动添加进球",
            "输入进球时间戳（秒）：",
            0,
            0,
            99999,
            1,
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

        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择篮球录像文件",
            "",
            "视频文件 (*.mp4 *.mov *.avi *.mkv *.m4v);;所有文件 (*)",
        )
        if not video_path:
            return

        if not self.project:
            video_dir = os.path.dirname(video_path)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            project_dir = os.path.join(video_dir, f"{video_name}_project")
            os.makedirs(project_dir, exist_ok=True)
            self.project = Project(video_path=video_path, project_dir=project_dir)
            self.project.save()
            if hasattr(self.parent(), "project"):
                self.parent().project = self.project
        else:
            if os.path.abspath(self.project.video_path) != os.path.abspath(video_path):
                self.project.clips.clear()
                self.project.detection_done = False
                self.project.manual_hoop_rect = None
                self.project.last_detection_stats = {}
                self.project.last_detection_failure_reason = None
                self.project.last_detection_config = {}
            self.project.video_path = video_path
            self.project.save()

        self.load_project(self.project)

    def finish_detection(self):
        if not self.project or not self.project.clips:
            QMessageBox.warning(self, "提示", "没有进球片段，请先检测或手动添加")
            return
        self.detection_finished.emit(self.project)

    def _refresh_list(self):
        self.clip_list.clear()
        if not self.project:
            self.btn_next.setEnabled(False)
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

        self.btn_next.setEnabled(bool(self.project.clips))
