# -*- coding: utf-8 -*-
from QtFusion.config import QF_Config
import os  # 导入os模块，用于处理操作系统相关的功能
from sys import argv, exit  # 导入用于处理命令行参数和退出程序的模块
from PySide6.QtWidgets import QApplication  # 导入用于创建应用程序的类
QF_Config.set_verbose(False)

os.environ["QT_FONT_DPI"] = "96"

if __name__ == '__main__':
    from System import RecMainWindow
    app = QApplication(argv)
    win = RecMainWindow()
    win.showTime()
    exit(app.exec())
