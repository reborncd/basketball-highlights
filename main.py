"""
main.py
程序入口
"""
import sys
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("篮球进球集锦剪辑")
    app.setStyle("Fusion")

    # 全局样式
    app.setStyleSheet("""
        QMainWindow { background: #F5F6FA; }
        QGroupBox {
            border: 1px solid #DDD;
            border-radius: 6px;
            margin-top: 8px;
            padding: 8px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #333;
        }
        QListWidget {
            border: 1px solid #DDD;
            border-radius: 4px;
        }
        QProgressBar {
            border: 1px solid #CCC;
            border-radius: 4px;
            text-align: center;
            height: 20px;
        }
        QProgressBar::chunk {
            background: #2980B9;
            border-radius: 4px;
        }
        QTabBar::tab {
            padding: 8px 20px;
            font-size: 13px;
        }
        QTabBar::tab:selected {
            color: #2980B9;
            font-weight: bold;
        }
    """)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
