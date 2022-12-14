import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
from pyqtgraph import PlotWidget, PlotItem
import pyqtgraph as pg
import os
import pathlib
import scipy.signal as signal
from detection_gui import Detectors


class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(846, 718)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.resume = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.resume.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap("icon-play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.resume.setIcon(icon)
        self.resume.setIconSize(QtCore.QSize(50, 50))
        self.resume.setObjectName("resume")
        self.horizontalLayout.addWidget(self.resume)
        self.pause = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.pause.setFont(font)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(
            QtGui.QPixmap("icon-pause.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.pause.setIcon(icon1)
        self.pause.setIconSize(QtCore.QSize(50, 50))
        self.pause.setObjectName("pause")
        self.horizontalLayout.addWidget(self.pause)
        self.clear = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Yu Gothic UI")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.clear.setFont(font)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(
            QtGui.QPixmap("icon-erase.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.clear.setIcon(icon2)
        self.clear.setIconSize(QtCore.QSize(50, 50))
        self.clear.setObjectName("clear")
        self.horizontalLayout.addWidget(self.clear)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.show_ch1 = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.show_ch1.setFont(font)
        self.show_ch1.setObjectName("show_ch1")
        self.verticalLayout.addWidget(self.show_ch1)
        self.graphicsView = PlotWidget(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout.addWidget(self.graphicsView)
        self.graphicsView.plotItem.showGrid(x=True, y=True)
        self.graphicsView.plotItem.setMenuEnabled(False)
        self.graphicsView.plotItem.setLimits(xMin=0, xMax=15, yMin=-0.7, yMax=0.7)

        self.show_ch2 = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.show_ch2.setFont(font)
        self.show_ch2.setObjectName("show_ch2")
        self.verticalLayout.addWidget(self.show_ch2)
        self.graphicsView_2 = PlotWidget(self.centralwidget)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.verticalLayout.addWidget(self.graphicsView_2)
        self.graphicsView_2.plotItem.showGrid(x=True, y=True)
        self.graphicsView_2.plotItem.setMenuEnabled(False)

        self.show_cch3 = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.show_cch3.setFont(font)
        self.show_cch3.setObjectName("show_cch3")
        self.verticalLayout.addWidget(self.show_cch3)
        self.graphicsView_3 = PlotWidget(self.centralwidget)
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.verticalLayout.addWidget(self.graphicsView_3)
        self.graphicsView_3.plotItem.showGrid(x=True, y=True)
        self.graphicsView_3.plotItem.setMenuEnabled(False)

        self.show_ch4 = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.show_ch4.setFont(font)
        self.show_ch4.setObjectName("show_ch4")
        self.verticalLayout.addWidget(self.show_ch4)
        self.graphicsView_4 = PlotWidget(self.centralwidget)
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.verticalLayout.addWidget(self.graphicsView_4)
        self.graphicsView_4.plotItem.showGrid(x=True, y=True)
        self.graphicsView_4.plotItem.setMenuEnabled(False)

        self.show_ch5 = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.show_ch5.setFont(font)
        self.show_ch5.setObjectName("show_ch5")
        self.verticalLayout.addWidget(self.show_ch5)
        self.graphicsView_5 = PlotWidget(self.centralwidget)
        self.graphicsView_5.setObjectName("graphicsView_5")
        self.verticalLayout.addWidget(self.graphicsView_5)
        self.graphicsView_5.plotItem.showGrid(x=True, y=True)
        self.graphicsView_5.plotItem.setMenuEnabled(False)

        self.show_ch6 = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.show_ch6.setFont(font)
        self.show_ch6.setObjectName("show_ch6")
        self.verticalLayout.addWidget(self.show_ch6)
        self.graphicsView_6 = PlotWidget(self.centralwidget)
        self.graphicsView_6.setObjectName("graphicsView_6")
        self.verticalLayout.addWidget(self.graphicsView_6)
        self.graphicsView_6.plotItem.showGrid(x=True, y=True)
        self.graphicsView_6.plotItem.setMenuEnabled(False)

        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 846, 25))
        self.menubar.setObjectName("menubar")
        self.menuchannel_1 = QtWidgets.QMenu(self.menubar)
        self.menuchannel_1.setObjectName("menuchannel_1")
        self.menuchannel_2 = QtWidgets.QMenu(self.menubar)
        self.menuchannel_2.setObjectName("menuchannel_2")
        self.menuchannel_3 = QtWidgets.QMenu(self.menubar)
        self.menuchannel_3.setObjectName("menuchannel_3")
        self.menuchannel_4 = QtWidgets.QMenu(self.menubar)
        self.menuchannel_4.setObjectName("menuchannel_4")
        self.menuchannel_5 = QtWidgets.QMenu(self.menubar)
        self.menuchannel_5.setObjectName("menuchannel_5")
        self.menuchannel_6 = QtWidgets.QMenu(self.menubar)
        self.menuchannel_6.setObjectName("menuchannel_6")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.open_ch1 = QtWidgets.QAction(MainWindow)
        self.open_ch1.setObjectName("open_ch1")
        self.open_ch2 = QtWidgets.QAction(MainWindow)
        self.open_ch2.setObjectName("open_ch2")
        self.open_ch3 = QtWidgets.QAction(MainWindow)
        self.open_ch3.setObjectName("open_ch3")
        self.open_ch4 = QtWidgets.QAction(MainWindow)
        self.open_ch4.setObjectName("open_ch4")
        self.open_ch5 = QtWidgets.QAction(MainWindow)
        self.open_ch5.setObjectName("open_ch5")
        self.open_ch6 = QtWidgets.QAction(MainWindow)
        self.open_ch6.setObjectName("open_ch6")
        self.menuchannel_1.addAction(self.open_ch1)
        self.menuchannel_2.addAction(self.open_ch2)
        self.menuchannel_3.addAction(self.open_ch3)
        self.menuchannel_4.addAction(self.open_ch4)
        self.menuchannel_5.addAction(self.open_ch5)
        self.menuchannel_6.addAction(self.open_ch6)
        self.menubar.addAction(self.menuchannel_1.menuAction())
        self.menubar.addAction(self.menuchannel_2.menuAction())
        self.menubar.addAction(self.menuchannel_3.menuAction())
        self.menubar.addAction(self.menuchannel_4.menuAction())
        self.menubar.addAction(self.menuchannel_5.menuAction())
        self.menubar.addAction(self.menuchannel_6.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        # creating timers
        self.timer1 = QtCore.QTimer()
        self.timer2 = QtCore.QTimer()
        self.timer3 = QtCore.QTimer()
        self.timer4 = QtCore.QTimer()
        self.timer5 = QtCore.QTimer()
        self.timer6 = QtCore.QTimer()

        # events
        self.open_ch1.triggered.connect(lambda: self.load1())
        self.open_ch2.triggered.connect(lambda: self.load2())
        self.open_ch3.triggered.connect(lambda: self.load3())
        self.open_ch4.triggered.connect(lambda: self.load4())
        self.open_ch5.triggered.connect(lambda: self.load5())
        self.open_ch6.triggered.connect(lambda: self.load5())
        self.pause.clicked.connect(lambda: self.pause_all())
        self.resume.clicked.connect(lambda: self.resume_all())
        self.clear.clicked.connect(lambda: self.clear_all())
        self.show_ch1.stateChanged.connect(lambda: self.hide1())
        self.show_ch2.stateChanged.connect(lambda: self.hide2())
        self.show_cch3.stateChanged.connect(lambda: self.hide3())
        self.show_ch4.stateChanged.connect(lambda: self.hide4())
        self.show_ch5.stateChanged.connect(lambda: self.hide5())
        self.show_ch6.stateChanged.connect(lambda: self.hide6())

    def hide1(self):
        if self.show_ch1.isChecked():
            self.graphicsView.hide()
        else:
            self.graphicsView.show()

    def hide2(self):
        if self.show_ch2.isChecked():
            self.graphicsView_2.hide()
        else:
            self.graphicsView_2.show()

    def hide3(self):
        if self.show_cch3.isChecked():
            self.graphicsView_3.hide()
        else:
            self.graphicsView_3.show()

    def hide4(self):
        if self.show_ch4.isChecked():
            self.graphicsView_4.hide()
        else:
            self.graphicsView_4.show()

    def hide5(self):
        if self.show_ch5.isChecked():
            self.graphicsView_5.hide()
        else:
            self.graphicsView_5.show()

    def hide6(self):
        if self.show_ch6.isChecked():
            self.graphicsView_6.hide()
        else:
            self.graphicsView_6.show()

    def pause_all(self):

        self.timer1.stop()
        self.timer2.stop()
        self.timer3.stop()
        self.timer4.stop()
        self.timer5.stop()
        self.timer6.stop()

    def resume_all(self):
        self.timer1.start()
        self.timer2.start()
        self.timer3.start()
        self.timer4.start()
        self.timer5.start()
        self.timer6.start()

    def clear_all(self):
        self.graphicsView.clear()
        self.graphicsView_2.clear()
        self.graphicsView_3.clear()
        self.graphicsView_4.clear()
        self.graphicsView_5.clear()
        self.graphicsView_6.clear()
        self.pause_all()

    def read_file1(self):
        self.fname1 = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open only txt or CSV or xls",
            os.getenv("HOME"),
            "csv(*.csv);; text(*.txt) ;; xls(*.xls)",
        )
        path = self.fname1[0]
        # self.name1= self.fname1

        if pathlib.Path(path).suffix == ".txt":
            self.data1 = np.genfromtxt(path, delimiter=",")
            self.x1 = self.data1[:, 0]
            self.y1 = self.data1[:, 1]
            self.x1 = list(self.x1[:])
            self.y1 = list(self.y1[:])
        elif pathlib.Path(path).suffix == ".csv":
            self.data1 = np.genfromtxt(path, delimiter=",")
            self.x1 = self.data1[:, 0]
            self.y1 = self.data1[:, 1]
            self.x1 = list(self.x1[:])
            self.y1 = list(self.y1[:])
        elif pathlib.Path(path).suffix == ".xls":
            self.data1 = np.genfromtxt(path, delimiter=",")
            self.x1 = self.data1[:, 0]
            self.y1 = self.data1[:, 1]
            self.x1 = list(self.x1[:])
            self.y1 = list(self.y1[:])

    def read_file2(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open only txt or CSV or xls",
            os.getenv("HOME"),
            "csv(*.csv);; text(*.txt) ;; xls(*.xls)",
        )
        path = fname[0]

        if pathlib.Path(path).suffix == ".txt":
            self.data2 = np.genfromtxt(path, delimiter=",")
            self.x2 = self.data2[:, 0]
            self.y2 = self.data2[:, 1]
            self.x2 = list(self.x2[:])
            self.y2 = list(self.y2[:])
        elif pathlib.Path(path).suffix == ".csv":
            self.data2 = np.genfromtxt(path, delimiter=" ")
            self.x2 = self.data2[:, 0]
            self.y2 = self.data2[:, 1]
            self.x2 = list(self.x2[:])
            self.y2 = list(self.y2[:])
        elif pathlib.Path(path).suffix == ".xls":
            self.data2 = np.genfromtxt(path, delimiter=",")
            self.x2 = self.data2[:, 0]
            self.y2 = self.data2[:, 1]
            self.x2 = list(self.x2[:])
            self.y2 = list(self.y2[:])

    def read_file3(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open only txt or CSV or xls",
            os.getenv("HOME"),
            "csv(*.csv);; text(*.txt) ;; xls(*.xls)",
        )
        path = fname[0]

        if pathlib.Path(path).suffix == ".txt":
            self.data3 = np.genfromtxt(path, delimiter=",")
            self.x3 = self.data3[:, 0]
            self.y3 = self.data3[:, 1]
            self.x3 = list(self.x3[:])
            self.y3 = list(self.y3[:])
        elif pathlib.Path(path).suffix == ".csv":
            self.data3 = np.genfromtxt(path, delimiter=" ")
            self.x3 = self.data3[:, 0]
            self.y3 = self.data3[:, 1]
            self.x3 = list(self.x3[:])
            self.y3 = list(self.y3[:])
        elif pathlib.Path(path).suffix == ".xls":
            self.data3 = np.genfromtxt(path, delimiter=",")
            self.x3 = self.data3[:, 0]
            self.y3 = self.data3[:, 1]
            self.x3 = list(self.x3[:])
            self.y3 = list(self.y3[:])

    def read_file4(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open only txt or CSV or xls",
            os.getenv("HOME"),
            "csv(*.csv);; text(*.txt) ;; xls(*.xls)",
        )
        path = fname[0]

        if pathlib.Path(path).suffix == ".txt":
            self.data4 = np.genfromtxt(path, delimiter=",")
            self.x4 = self.data4[:, 0]
            self.y4 = self.data4[:, 1]
            self.x4 = list(self.x4[:])
            self.y4 = list(self.y4[:])
        elif pathlib.Path(path).suffix == ".csv":
            self.data4 = np.genfromtxt(path, delimiter=" ")
            self.x4 = self.data4[:, 0]
            self.y4 = self.data4[:, 1]
            self.x4 = list(self.x4[:])
            self.y4 = list(self.y4[:])
        elif pathlib.Path(path).suffix == ".xls":
            self.data4 = np.genfromtxt(path, delimiter=",")
            self.x4 = self.data4[:, 0]
            self.y4 = self.data4[:, 1]
            self.x4 = list(self.x4[:])
            self.y4 = list(self.y4[:])

    def read_file5(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open only txt or CSV or xls",
            os.getenv("HOME"),
            "csv(*.csv);; text(*.txt) ;; xls(*.xls)",
        )
        path = fname[0]

        if pathlib.Path(path).suffix == ".txt":
            self.data5 = np.genfromtxt(path, delimiter=",")
            self.x5 = self.data5[:, 0]
            self.y5 = self.data5[:, 1]
            self.x5 = list(self.x5[:])
            self.y5 = list(self.y5[:])

        elif pathlib.Path(path).suffix == ".csv":
            self.data5 = np.genfromtxt(path, delimiter=" ")
            self.x5 = self.data5[:, 0]
            self.y5 = self.data5[:, 1]
            self.x5 = list(self.x5[:])
            self.y5 = list(self.y5[:])
        elif pathlib.Path(path).suffix == ".xls":
            self.data5 = np.genfromtxt(path, delimiter=",")
            self.x5 = self.data5[:, 0]
            self.y5 = self.data5[:, 1]
            self.x5 = list(self.x5[:])
            self.y5 = list(self.y5[:])

    def read_file6(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open only txt or CSV or xls",
            os.getenv("HOME"),
            "csv(*.csv);; text(*.txt) ;; xls(*.xls)",
        )
        path = fname[0]

        if pathlib.Path(path).suffix == ".txt":
            self.data6 = np.genfromtxt(path, delimiter=",")
            self.x6 = self.data6[:, 0]
            self.y6 = self.data6[:, 1]
            self.x6 = list(self.x6[:])
            self.y6 = list(self.y6[:])

        elif pathlib.Path(path).suffix == ".csv":
            self.data6 = np.genfromtxt(path, delimiter=" ")
            self.x6 = self.data6[:, 0]
            self.y6 = self.data6[:, 1]
            self.x6 = list(self.x6[:])
            self.y6 = list(self.y6[:])
        elif pathlib.Path(path).suffix == ".xls":
            self.data6 = np.genfromtxt(path, delimiter=",")
            self.x6 = self.data6[:, 0]
            self.y6 = self.data6[:, 1]
            self.x6 = list(self.x6[:])
            self.y6 = list(self.y6[:])

    def load1(self):
        self.read_file1()
        self.pen_r = pg.mkPen(color=(255, 0, 0))
        self.pen_g = pg.mkPen(color=(0, 255, 0))
        self.pen_b = pg.mkPen(color=(0, 0, 255))
        # self.data_line1 = self.graphicsView.plot(self.x1, self.y1, pen=self.pen_r)
        # self.graphicsView.plotItem.setLimits(xMin =0, xMax=12 , yMin =-0.6, yMax=0.6)
        self.graphicsView.plotItem.setLimits(xMin=0, xMax=12, yMin=-1, yMax=3)

        self.idx1 = 0
        self.timer1.setInterval(100)

        self.timer1.timeout.connect(self.update_plot_data1)
        self.timer1.start()

    def load2(self):

        self.read_file2()
        self.pen = pg.mkPen(color=(0, 160, 0))
        self.data_line2 = self.graphicsView_2.plot(self.x2, self.y2, pen=self.pen)
        self.graphicsView_2.plotItem.setLimits(xMin=0, xMax=12, yMin=-0.6, yMax=0.6)
        # self.graphicsView_2.plotItem.setXRange(0 , 0.5)
        # self.graphicsView.plotItem.enableAutoRange(enable=True)

        self.idx2 = 0
        self.timer2.setInterval(60)
        self.timer2.timeout.connect(self.update_plot_data2)
        self.timer2.start()

    def load3(self):

        self.read_file3()
        self.pen = pg.mkPen(color=(255, 255, 0))
        self.data_line3 = self.graphicsView_3.plot(self.x3, self.y3, pen=self.pen)
        self.graphicsView_3.plotItem.setLimits(xMin=0, xMax=12, yMin=-0.6, yMax=0.6)

        self.idx3 = 0
        self.timer3.setInterval(20)
        self.timer3.timeout.connect(self.update_plot_data3)
        self.timer3.start()

    def load4(self):

        self.read_file4()
        self.pen = pg.mkPen(color=(0, 160, 255))
        self.data_line4 = self.graphicsView_4.plot(self.x4, self.y4, pen=self.pen)
        self.graphicsView_4.plotItem.setLimits(xMin=0, xMax=12, yMin=-0.6, yMax=0.6)

        self.idx4 = 0
        self.timer4.setInterval(20)
        self.timer4.timeout.connect(self.update_plot_data4)
        self.timer4.start()

    def load5(self):

        self.read_file5()
        self.pen = pg.mkPen(color=(0, 255, 255))
        self.data_line5 = self.graphicsView_5.plot(self.x5, self.y5, pen=self.pen)
        self.graphicsView_5.plotItem.setLimits(xMin=0, xMax=12, yMin=-0.6, yMax=0.6)

        self.idx5 = 0
        self.timer5.setInterval(20)
        self.timer5.timeout.connect(self.update_plot_data5)
        self.timer5.start()

    def load6(self):

        self.read_file6()
        self.pen = pg.mkPen(color=(0, 255, 255))
        self.data_line5 = self.graphicsView_6.plot(self.x6, self.y6, pen=self.pen)
        self.graphicsView_6.plotItem.setLimits(xMin=0, xMax=12, yMin=-0.6, yMax=0.6)

        self.idx6 = 0
        self.timer6.setInterval(20)
        self.timer6.timeout.connect(self.update_plot_data6)
        self.timer6.start()

    def update_plot_data1(self):
        x = self.x1[: self.idx1]
        y = self.y1[: self.idx1]
        # self.idx1 +=10
        self.idx1 += 50

        # signal processing
        self.fs = 500
        unfiltered_ecg = y
        f1 = 48 / self.fs
        f2 = 52 / self.fs
        b, a = signal.butter(4, [f1 * 2, f2 * 2], btype="bandstop")
        filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)

        diff = np.zeros(len(filtered_ecg))
        for i in range(4, len(diff)):
            diff[i] = filtered_ecg[i] - filtered_ecg[i - 4]

        ci = [1, 4, 6, 4, 1]
        low_pass = signal.lfilter(ci, 1, diff)
        low_pass[: int(0.2 * self.fs)] = 0

        ms200 = int(0.2 * self.fs)
        ms1200 = int(1.2 * self.fs)
        ms160 = int(0.16 * self.fs)
        neg_threshold = int(0.01 * self.fs)

        M = 0
        M_list = []
        neg_m = []
        MM = []
        M_slope = np.linspace(1.0, 0.6, ms1200 - ms200)

        QRS = []
        r_peaks = []
        r_bottoms = []
        max_dvdts = []
        min_dvdts = []
        unfiltered_section = []
        peaks_x = []
        peaks_y = []

        counter = 0

        thi_list = []
        thi = False
        thf_list = []
        thf = False
        newM5 = False

        for i in range(len(low_pass)):
            if i < 5 * self.fs:
                M = 0.6 * np.max(low_pass[: i + 1])
                MM.append(M)
                if len(MM) > 5:
                    MM.pop(0)
                # print("MM:", i, QRS)
            elif QRS and i < QRS[-1] + ms200:
                newM5 = 0.6 * np.max(low_pass[QRS[-1] : i])
                if newM5 > 1.5 * MM[-1]:
                    newM5 = 1.1 * MM[-1]

            elif newM5 and QRS and i == QRS[-1] + ms200:
                MM.append(newM5)
                if len(MM) > 5:
                    MM.pop(0)
                M = np.mean(MM)

            elif QRS and i > QRS[-1] + ms200 and i < QRS[-1] + ms1200:
                M = np.mean(MM) * M_slope[i - (QRS[-1] + ms200)]

            elif QRS and i > QRS[-1] + ms1200:
                M = 0.6 * np.mean(MM)

            M_list.append(M)
            neg_m.append(-M)

            # potential peak shows up
            if not QRS and low_pass[i] > M:
                QRS.append(i)
                thi_list.append(i)
                thi = True

            elif QRS and i > QRS[-1] + ms200 and low_pass[i] > M:
                QRS.append(i)
                thi_list.append(i)
                thi = True

            if thi and i < thi_list[-1] + ms160:
                if low_pass[i] < -M and low_pass[i - 1] > -M:
                    thf = True

                if thf and low_pass[i] < -M:
                    thf_list.append(i)
                    counter += 1

                elif low_pass[i] > -M and thf:
                    counter = 0
                    thi = False
                    thf = False
            elif thi and i > thi_list[-1] + ms160:
                counter = 0
                thi = False
                thf = False

            if counter > neg_threshold:
                unfiltered_section = unfiltered_ecg[
                    thi_list[-1] - int(0.01 * self.fs) : i
                ]
                r_peaks.append(
                    np.argmax(unfiltered_section) + thi_list[-1] - int(0.01 * self.fs)
                )
                r_bottoms.append(
                    np.argmin(unfiltered_section) + thi_list[-1] - int(0.01 * self.fs)
                )

                counter = 0
                thi = False
                thf = False
        print("self.idx1: ", self.idx1)

        # Check if a cycle is done
        if self.idx1 > len(self.x1):
            self.idx1 = 0

        if self.x1[self.idx1] > 3:
            self.graphicsView.setLimits(
                xMin=min(x, default=0), xMax=max(x, default=0)
            )  # disable paning over xlimits

        print(type(unfiltered_ecg), len(unfiltered_ecg))
        print(r_peaks)
        if len(r_peaks) > 0:
            for i in r_peaks:
                peaks_y.append(unfiltered_ecg[i])
                peaks_x.append(i * 0.002)

        self.graphicsView.plotItem.setXRange(max(x, default=0) - 3, max(x, default=0))
        self.graphicsView.plotItem.clear()
        print("peaks_x: {}, peaks_y: {}".format(peaks_x, peaks_y))
        self.graphicsView.plotItem.plot(
            peaks_x,
            peaks_y,
            pen="r",
            symbol="x",
            symbolBrush=0.4,
        )

        self.graphicsView.plotItem.plot(x, y, pen="g")
        # self.data_line1.setData(x, y)

    def update_plot_data2(self):

        x = self.x2[: self.idx2]
        y = self.y2[: self.idx2]

        self.idx2 += 10
        if self.idx2 > len(self.x2):
            self.idx2 = 0
        if self.x2[self.idx2] > 0.5:
            self.graphicsView_2.setLimits(
                xMin=min(x, default=0), xMax=max(x, default=0)
            )  # disable paning over xlimits

        self.graphicsView_2.plotItem.setXRange(
            max(x, default=0) - 0.5, max(x, default=0)
        )
        self.data_line2.setData(x, y)  # Update the data.

    def update_plot_data3(self):

        x = self.x3[: self.idx3]
        y = self.y3[: self.idx3]
        self.data_line3.setData(x, y)  # Update the data.
        self.idx3 += 10
        if self.idx3 > len(self.x3):
            self.idx3 = 0
        if self.x3[self.idx3] > 0.5:
            self.graphicsView_3.setLimits(
                xMin=min(x, default=0), xMax=max(x, default=0)
            )

        self.graphicsView_3.plotItem.setXRange(
            max(x, default=0) - 0.5, max(x, default=0)
        )

    def update_plot_data4(self):

        x = self.x4[: self.idx4]
        y = self.y4[: self.idx4]
        # self.y2.append( self.ytemp)  #Add a new random value.
        self.data_line4.setData(x, y)  # Update the data.
        self.idx4 += 10
        if self.idx4 > len(self.x4):
            self.idx4 = 0
        if self.x4[self.idx4] > 0.5:
            self.graphicsView_4.setLimits(
                xMin=min(x, default=0), xMax=max(x, default=0)
            )

        self.graphicsView_4.plotItem.setXRange(
            max(x, default=0) - 0.5, max(x, default=0)
        )

    def update_plot_data5(self):

        x = self.x5[: self.idx5]
        y = self.y5[: self.idx5]

        self.data_line5.setData(x, y)  # Update the data.
        self.idx5 += 10
        if self.idx5 > len(self.x5):
            self.idx5 = 0
        if self.x5[self.idx5] > 0.5:
            self.graphicsView_5.setLimits(
                xMin=min(x, default=0), xMax=max(x, default=0)
            )  # disable paning over xlimits

        self.graphicsView_5.plotItem.setXRange(
            max(x, default=0) - 0.5, max(x, default=0)
        )

    def update_plot_data6(self):

        x = self.x6[: self.idx6]
        y = self.y6[: self.idx6]

        self.data_line6.setData(x, y)  # Update the data.
        self.idx6 += 10
        if self.idx6 > len(self.x6):
            self.idx6 = 0
        if self.x6[self.idx6] > 0.5:
            self.graphicsView_6.setLimits(
                xMin=min(x, default=0), xMax=max(x, default=0)
            )  # disable paning over xlimits

        self.graphicsView_6.plotItem.setXRange(
            max(x, default=0) - 0.5, max(x, default=0)
        )

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "LAT Test Platform"))
        MainWindow.setWindowIcon(QtGui.QIcon("logo.png"))
        self.resume.setText(_translate("MainWindow", "Resume"))
        self.pause.setText(_translate("MainWindow", "Pause"))
        self.clear.setText(_translate("MainWindow", "Clear"))
        self.show_ch1.setText(_translate("MainWindow", "Hide channel 1"))
        self.show_ch2.setText(_translate("MainWindow", "Hide channel 2"))
        self.show_cch3.setText(_translate("MainWindow", "Hide channel 3"))
        self.show_ch4.setText(_translate("MainWindow", "Hide channel 4"))
        self.show_ch5.setText(_translate("MainWindow", "Hide channel 5"))
        self.show_ch6.setText(_translate("MainWindow", "Hide channel 6"))
        self.label.setText(
            _translate(
                "MainWindow",
                "                                                                                   To zoom drag the graph to the right",
            )
        )
        self.menuchannel_1.setTitle(_translate("MainWindow", "channel 1"))
        self.menuchannel_2.setTitle(_translate("MainWindow", "channel 2"))
        self.menuchannel_3.setTitle(_translate("MainWindow", "channel 3"))
        self.menuchannel_4.setTitle(_translate("MainWindow", "channel 4"))
        self.menuchannel_5.setTitle(_translate("MainWindow", "channel 5"))
        self.menuchannel_6.setTitle(_translate("MainWindow", "channel 6"))
        self.open_ch1.setText(_translate("MainWindow", "open with"))
        self.open_ch2.setText(_translate("MainWindow", "open with"))
        self.open_ch3.setText(_translate("MainWindow", "open with"))
        self.open_ch4.setText(_translate("MainWindow", "open with"))
        self.open_ch5.setText(_translate("MainWindow", "open with"))
        self.open_ch6.setText(_translate("MainWindow", "open with"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
