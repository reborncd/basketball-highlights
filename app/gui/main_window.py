"""
main_window.py
主窗口 — 三页式 Tab 布局：检测 → 复核 → 导出
"""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget,
    QStatusBar, QMenuBar, QFileDialog, QMessageBox, QLabel, QAction
)
from PyQt5.QtCore import Qt
import os

from app.core.project import Project
from app.gui.detection_page import DetectionPage
from app.gui.review_page import ReviewPage
from app.gui.export_page import ExportPage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("篮球进球集锦剪辑工具")
        self.setMinimumSize(1100, 720)
        self.project: Project | None = None

        self._build_menu()
        self._build_central()
        self._build_statusbar()

    # ── 菜单栏 ───────────────────────────────────────────────
    def _build_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("文件(&F)")

        act_new = QAction("新建项目(&N)", self)
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self.on_new_project)
        file_menu.addAction(act_new)

        act_open = QAction("打开项目(&O)", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self.on_open_project)
        file_menu.addAction(act_open)

        file_menu.addSeparator()

        act_quit = QAction("退出(&Q)", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

    # ── 中央 Widget ──────────────────────────────────────────
    def _build_central(self):
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        self.detection_page = DetectionPage(self)
        self.review_page = ReviewPage(self)
        self.export_page = ExportPage(self)

        self.tabs.addTab(self.detection_page, "① 进球检测")
        self.tabs.addTab(self.review_page,    "② 复核 & 分配球员")
        self.tabs.addTab(self.export_page,    "③ 导出集锦")

        # 只有检测完成后才能切换到后续页
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)

        # 页面间信号连接
        self.detection_page.detection_finished.connect(self.on_detection_finished)
        self.review_page.review_done.connect(self.on_review_done)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.setCentralWidget(container)

    def _build_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪 — 新建项目或打开视频开始")

    # ── 信号处理 ─────────────────────────────────────────────
    def on_detection_finished(self, project: Project):
        self.project = project
        self.tabs.setTabEnabled(1, True)
        self.tabs.setCurrentIndex(1)
        self.review_page.load_project(project)
        self.status.showMessage(
            f"检测完成，共 {len(project.clips)} 个进球片段 — 请在第②页分配球员"
        )

    def on_review_done(self):
        self.tabs.setTabEnabled(2, True)
        self.tabs.setCurrentIndex(2)
        self.export_page.load_project(self.project)
        self.status.showMessage("球员分配完成 — 请在第③页导出集锦")

    # ── 新建 / 打开 ──────────────────────────────────────────
    def on_new_project(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, "选择篮球录像文件", "",
            "视频文件 (*.mp4 *.mov *.avi *.mkv *.m4v);;所有文件 (*)"
        )
        if not video_path:
            return

        proj_dir = QFileDialog.getExistingDirectory(
            self, "选择项目保存目录（会在此目录下存放所有文件）"
        )
        if not proj_dir:
            return

        project = Project(video_path=video_path, project_dir=proj_dir)
        project.save()
        self.project = project
        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(2, False)
        self.tabs.setCurrentIndex(0)
        self.detection_page.load_project(project)
        self.status.showMessage(f"已加载视频: {os.path.basename(video_path)}")

    def on_open_project(self):
        proj_dir = QFileDialog.getExistingDirectory(
            self, "打开项目目录（选择含 project.json 的目录）"
        )
        if not proj_dir:
            return
        proj_file = os.path.join(proj_dir, "project.json")
        if not os.path.exists(proj_file):
            QMessageBox.warning(self, "错误", f"目录中未找到 project.json:\n{proj_dir}")
            return
        try:
            project = Project.load(proj_dir)
            self.project = project
            self.detection_page.load_project(project)
            if project.detection_done and project.clips:
                self.tabs.setTabEnabled(1, True)
                self.review_page.load_project(project)
                self.tabs.setTabEnabled(2, True)
                self.export_page.load_project(project)
                self.tabs.setCurrentIndex(1)
            self.status.showMessage(f"已打开项目: {proj_dir}")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))
