# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_pyEMsoftMW(object):
    def setupUi(self, pyEMsoftMW):
        pyEMsoftMW.setObjectName("pyEMsoftMW")
        pyEMsoftMW.resize(1100, 795)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(100)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(pyEMsoftMW.sizePolicy().hasHeightForWidth())
        pyEMsoftMW.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(pyEMsoftMW)
        self.centralwidget.setObjectName("centralwidget")
        self.PHI_Refine = QtWidgets.QCheckBox(self.centralwidget)
        self.PHI_Refine.setGeometry(QtCore.QRect(750, 535, 16, 25))
        self.PHI_Refine.setText("")
        self.PHI_Refine.setObjectName("PHI_Refine")
        self.StartFit = QtWidgets.QPushButton(self.centralwidget)
        self.StartFit.setGeometry(QtCore.QRect(950, 570, 80, 35))
        self.StartFit.setObjectName("StartFit")
        self.phi1_Step = QtWidgets.QLineEdit(self.centralwidget)
        self.phi1_Step.setGeometry(QtCore.QRect(670, 490, 61, 28))
        self.phi1_Step.setObjectName("phi1_Step")
        self.PHI_Step = QtWidgets.QLineEdit(self.centralwidget)
        self.PHI_Step.setGeometry(QtCore.QRect(670, 530, 61, 28))
        self.PHI_Step.setObjectName("PHI_Step")
        self.phi2_Step = QtWidgets.QLineEdit(self.centralwidget)
        self.phi2_Step.setGeometry(QtCore.QRect(670, 570, 61, 28))
        self.phi2_Step.setObjectName("phi2_Step")
        self.phi1_Refine = QtWidgets.QCheckBox(self.centralwidget)
        self.phi1_Refine.setGeometry(QtCore.QRect(750, 495, 16, 25))
        self.phi1_Refine.setText("")
        self.phi1_Refine.setObjectName("phi1_Refine")
        self.phi2_Refine = QtWidgets.QCheckBox(self.centralwidget)
        self.phi2_Refine.setGeometry(QtCore.QRect(750, 575, 16, 25))
        self.phi2_Refine.setText("")
        self.phi2_Refine.setObjectName("phi2_Refine")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(740, 470, 41, 20))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(530, 495, 40, 20))
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(530, 535, 40, 20))
        self.label_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(530, 575, 40, 20))
        self.label_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.Load_Master = QtWidgets.QPushButton(self.centralwidget)
        self.Load_Master.setGeometry(QtCore.QRect(10, 10, 190, 35))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Load_Master.sizePolicy().hasHeightForWidth())
        self.Load_Master.setSizePolicy(sizePolicy)
        self.Load_Master.setObjectName("Load_Master")
        self.Load_Exp = QtWidgets.QPushButton(self.centralwidget)
        self.Load_Exp.setGeometry(QtCore.QRect(10, 50, 190, 35))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Load_Exp.sizePolicy().hasHeightForWidth())
        self.Load_Exp.setSizePolicy(sizePolicy)
        self.Load_Exp.setObjectName("Load_Exp")
        self.DOWN = QtWidgets.QPushButton(self.centralwidget)
        self.DOWN.setGeometry(QtCore.QRect(150, 620, 75, 35))
        self.DOWN.setObjectName("DOWN")
        self.UP = QtWidgets.QPushButton(self.centralwidget)
        self.UP.setGeometry(QtCore.QRect(150, 580, 75, 35))
        self.UP.setObjectName("UP")
        self.RIGHT = QtWidgets.QPushButton(self.centralwidget)
        self.RIGHT.setGeometry(QtCore.QRect(260, 620, 75, 35))
        self.RIGHT.setObjectName("RIGHT")
        self.LEFT = QtWidgets.QPushButton(self.centralwidget)
        self.LEFT.setGeometry(QtCore.QRect(40, 620, 75, 35))
        self.LEFT.setObjectName("LEFT")
        self.CW = QtWidgets.QPushButton(self.centralwidget)
        self.CW.setGeometry(QtCore.QRect(260, 580, 75, 35))
        self.CW.setObjectName("CW")
        self.CCW = QtWidgets.QPushButton(self.centralwidget)
        self.CCW.setGeometry(QtCore.QRect(40, 580, 75, 35))
        self.CCW.setObjectName("CCW")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(560, 610, 111, 20))
        self.label_5.setObjectName("label_5")
        self.IS_TSL = QtWidgets.QCheckBox(self.centralwidget)
        self.IS_TSL.setGeometry(QtCore.QRect(690, 610, 16, 25))
        self.IS_TSL.setText("")
        self.IS_TSL.setObjectName("IS_TSL")
        self.phi1_Val = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.phi1_Val.setGeometry(QtCore.QRect(580, 490, 70, 30))
        self.phi1_Val.setObjectName("phi1_Val")
        self.PHI_Val = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.PHI_Val.setGeometry(QtCore.QRect(580, 530, 70, 30))
        self.PHI_Val.setObjectName("PHI_Val")
        self.phi2_Val = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.phi2_Val.setGeometry(QtCore.QRect(580, 570, 70, 30))
        self.phi2_Val.setObjectName("phi2_Val")
        self.PCX_Step = QtWidgets.QLineEdit(self.centralwidget)
        self.PCX_Step.setGeometry(QtCore.QRect(950, 245, 75, 30))
        self.PCX_Step.setObjectName("PCX_Step")
        self.Distance_Val = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.Distance_Val.setGeometry(QtCore.QRect(840, 165, 100, 30))
        self.Distance_Val.setDecimals(5)
        self.Distance_Val.setObjectName("Distance_Val")
        self.Omega_Step = QtWidgets.QLineEdit(self.centralwidget)
        self.Omega_Step.setGeometry(QtCore.QRect(950, 205, 75, 30))
        self.Omega_Step.setObjectName("Omega_Step")
        self.Distance_Refine = QtWidgets.QCheckBox(self.centralwidget)
        self.Distance_Refine.setGeometry(QtCore.QRect(1040, 170, 20, 20))
        self.Distance_Refine.setText("")
        self.Distance_Refine.setObjectName("Distance_Refine")
        self.Omega_Refine = QtWidgets.QCheckBox(self.centralwidget)
        self.Omega_Refine.setGeometry(QtCore.QRect(1040, 210, 17, 18))
        self.Omega_Refine.setText("")
        self.Omega_Refine.setObjectName("Omega_Refine")
        self.PCX_Val = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.PCX_Val.setGeometry(QtCore.QRect(840, 245, 100, 30))
        self.PCX_Val.setDecimals(5)
        self.PCX_Val.setObjectName("PCX_Val")
        self.Distance_Step = QtWidgets.QLineEdit(self.centralwidget)
        self.Distance_Step.setGeometry(QtCore.QRect(950, 165, 75, 30))
        self.Distance_Step.setObjectName("Distance_Step")
        self.Omega_Val = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.Omega_Val.setGeometry(QtCore.QRect(840, 205, 100, 30))
        self.Omega_Val.setDecimals(5)
        self.Omega_Val.setObjectName("Omega_Val")
        self.PCX_Refine = QtWidgets.QCheckBox(self.centralwidget)
        self.PCX_Refine.setGeometry(QtCore.QRect(1040, 250, 17, 18))
        self.PCX_Refine.setText("")
        self.PCX_Refine.setObjectName("PCX_Refine")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(770, 170, 60, 20))
        self.label_9.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(770, 210, 60, 20))
        self.label_10.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_10.setObjectName("label_10")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(770, 330, 60, 20))
        self.label_12.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_12.setObjectName("label_12")
        self.PCY_Step = QtWidgets.QLineEdit(self.centralwidget)
        self.PCY_Step.setGeometry(QtCore.QRect(950, 285, 75, 30))
        self.PCY_Step.setObjectName("PCY_Step")
        self.Tilt_Step = QtWidgets.QLineEdit(self.centralwidget)
        self.Tilt_Step.setGeometry(QtCore.QRect(950, 325, 75, 30))
        self.Tilt_Step.setObjectName("Tilt_Step")
        self.Tilt_Val = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.Tilt_Val.setGeometry(QtCore.QRect(840, 325, 100, 30))
        self.Tilt_Val.setDecimals(5)
        self.Tilt_Val.setObjectName("Tilt_Val")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(770, 290, 60, 20))
        self.label_13.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_13.setObjectName("label_13")
        self.PCY_Val = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.PCY_Val.setGeometry(QtCore.QRect(840, 285, 100, 30))
        self.PCY_Val.setDecimals(5)
        self.PCY_Val.setObjectName("PCY_Val")
        self.Tilt_Refine = QtWidgets.QCheckBox(self.centralwidget)
        self.Tilt_Refine.setGeometry(QtCore.QRect(1040, 330, 17, 18))
        self.Tilt_Refine.setText("")
        self.Tilt_Refine.setObjectName("Tilt_Refine")
        self.PCY_Refine = QtWidgets.QCheckBox(self.centralwidget)
        self.PCY_Refine.setGeometry(QtCore.QRect(1040, 290, 17, 18))
        self.PCY_Refine.setText("")
        self.PCY_Refine.setObjectName("PCY_Refine")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(540, 430, 241, 20))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(540, 100, 531, 20))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(1030, 140, 41, 20))
        self.label_8.setObjectName("label_8")
        self.Dwel_L = QtWidgets.QLabel(self.centralwidget)
        self.Dwel_L.setGeometry(QtCore.QRect(520, 250, 140, 20))
        self.Dwel_L.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Dwel_L.setObjectName("Dwel_L")
        self.Beam_Current = QtWidgets.QLineEdit(self.centralwidget)
        self.Beam_Current.setGeometry(QtCore.QRect(671, 205, 50, 30))
        self.Beam_Current.setObjectName("Beam_Current")
        self.Dwel = QtWidgets.QLineEdit(self.centralwidget)
        self.Dwel.setGeometry(QtCore.QRect(671, 245, 50, 30))
        self.Dwel.setObjectName("Dwel")
        self.Current_L = QtWidgets.QLabel(self.centralwidget)
        self.Current_L.setGeometry(QtCore.QRect(520, 210, 140, 20))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Current_L.sizePolicy().hasHeightForWidth())
        self.Current_L.setSizePolicy(sizePolicy)
        self.Current_L.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Current_L.setObjectName("Current_L")
        self.Pixel_Size = QtWidgets.QLineEdit(self.centralwidget)
        self.Pixel_Size.setGeometry(QtCore.QRect(670, 165, 50, 30))
        self.Pixel_Size.setDragEnabled(True)
        self.Pixel_Size.setObjectName("Pixel_Size")
        self.Pixel_Size_L = QtWidgets.QLabel(self.centralwidget)
        self.Pixel_Size_L.setGeometry(QtCore.QRect(520, 170, 140, 16))
        self.Pixel_Size_L.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Pixel_Size_L.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Pixel_Size_L.setObjectName("Pixel_Size_L")
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(840, 430, 201, 20))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_20.setFont(font)
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        self.label_24.setGeometry(QtCore.QRect(770, 250, 60, 20))
        self.label_24.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_24.setObjectName("label_24")
        self.debugWindow = QtWidgets.QTextEdit(self.centralwidget)
        self.debugWindow.setGeometry(QtCore.QRect(15, 660, 1070, 75))
        self.debugWindow.setObjectName("debugWindow")
        self.Binning_L = QtWidgets.QLabel(self.centralwidget)
        self.Binning_L.setGeometry(QtCore.QRect(520, 280, 140, 20))
        self.Binning_L.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Binning_L.setObjectName("Binning_L")
        self.label_25 = QtWidgets.QLabel(self.centralwidget)
        self.label_25.setGeometry(QtCore.QRect(770, 370, 60, 20))
        self.label_25.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_25.setObjectName("label_25")
        self.Gamma_Val = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.Gamma_Val.setGeometry(QtCore.QRect(840, 365, 100, 30))
        self.Gamma_Val.setDecimals(5)
        self.Gamma_Val.setObjectName("Gamma_Val")
        self.Gamma_Step = QtWidgets.QLineEdit(self.centralwidget)
        self.Gamma_Step.setGeometry(QtCore.QRect(950, 365, 75, 30))
        self.Gamma_Step.setObjectName("Gamma_Step")
        self.Gamma_Refine = QtWidgets.QCheckBox(self.centralwidget)
        self.Gamma_Refine.setGeometry(QtCore.QRect(1040, 370, 17, 18))
        self.Gamma_Refine.setText("")
        self.Gamma_Refine.setObjectName("Gamma_Refine")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(20, 90, 480, 480))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.PlotWidget = QtWidgets.QWidget(self.tab)
        self.PlotWidget.setGeometry(QtCore.QRect(10, 10, 460, 400))
        self.PlotWidget.setObjectName("PlotWidget")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.ExpPlotWidget = QtWidgets.QWidget(self.tab_2)
        self.ExpPlotWidget.setGeometry(QtCore.QRect(10, 10, 460, 400))
        self.ExpPlotWidget.setObjectName("ExpPlotWidget")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.ExpPlotWidget_2 = QtWidgets.QWidget(self.tab_3)
        self.ExpPlotWidget_2.setGeometry(QtCore.QRect(10, 10, 460, 400))
        self.ExpPlotWidget_2.setObjectName("ExpPlotWidget_2")
        self.tabWidget.addTab(self.tab_3, "")
        self.MasterFile_Disp = QtWidgets.QLabel(self.centralwidget)
        self.MasterFile_Disp.setGeometry(QtCore.QRect(210, 10, 850, 35))
        self.MasterFile_Disp.setText("")
        self.MasterFile_Disp.setObjectName("MasterFile_Disp")
        self.ExpFile_Disp = QtWidgets.QLabel(self.centralwidget)
        self.ExpFile_Disp.setGeometry(QtCore.QRect(210, 50, 850, 35))
        self.ExpFile_Disp.setText("")
        self.ExpFile_Disp.setObjectName("ExpFile_Disp")
        self.Load_JSON = QtWidgets.QPushButton(self.centralwidget)
        self.Load_JSON.setGeometry(QtCore.QRect(800, 620, 125, 35))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Load_JSON.sizePolicy().hasHeightForWidth())
        self.Load_JSON.setSizePolicy(sizePolicy)
        self.Load_JSON.setObjectName("Load_JSON")
        self.Save_JSON = QtWidgets.QPushButton(self.centralwidget)
        self.Save_JSON.setGeometry(QtCore.QRect(940, 620, 125, 35))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Save_JSON.sizePolicy().hasHeightForWidth())
        self.Save_JSON.setSizePolicy(sizePolicy)
        self.Save_JSON.setObjectName("Save_JSON")
        self.MaxIts = QtWidgets.QLineEdit(self.centralwidget)
        self.MaxIts.setGeometry(QtCore.QRect(950, 490, 80, 30))
        self.MaxIts.setObjectName("MaxIts")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(800, 495, 140, 20))
        self.label_16.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_16.setObjectName("label_16")
        self.Error = QtWidgets.QLineEdit(self.centralwidget)
        self.Error.setGeometry(QtCore.QRect(950, 530, 80, 30))
        self.Error.setObjectName("Error")
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setGeometry(QtCore.QRect(800, 535, 140, 20))
        self.label_18.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_18.setObjectName("label_18")
        self.DetectorBinning = QtWidgets.QLineEdit(self.centralwidget)
        self.DetectorBinning.setGeometry(QtCore.QRect(670, 285, 50, 30))
        self.DetectorBinning.setObjectName("DetectorBinning")
        self.Binning_L_2 = QtWidgets.QLabel(self.centralwidget)
        self.Binning_L_2.setGeometry(QtCore.QRect(520, 295, 140, 20))
        self.Binning_L_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Binning_L_2.setObjectName("Binning_L_2")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(840, 140, 41, 20))
        self.label_11.setObjectName("label_11")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(950, 140, 41, 20))
        self.label_14.setObjectName("label_14")
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        self.label_21.setGeometry(QtCore.QRect(580, 470, 41, 20))
        self.label_21.setObjectName("label_21")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(670, 470, 41, 20))
        self.label_22.setObjectName("label_22")
        self.Dwel_L_2 = QtWidgets.QLabel(self.centralwidget)
        self.Dwel_L_2.setGeometry(QtCore.QRect(520, 347, 140, 20))
        self.Dwel_L_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Dwel_L_2.setObjectName("Dwel_L_2")
        self.LinearBack = QtWidgets.QCheckBox(self.centralwidget)
        self.LinearBack.setGeometry(QtCore.QRect(680, 345, 16, 25))
        self.LinearBack.setText("")
        self.LinearBack.setObjectName("LinearBack")
        self.Togle_Step = QtWidgets.QLineEdit(self.centralwidget)
        self.Togle_Step.setGeometry(QtCore.QRect(390, 620, 50, 30))
        self.Togle_Step.setDragEnabled(True)
        self.Togle_Step.setObjectName("Togle_Step")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(370, 590, 81, 20))
        self.label_15.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_15.setObjectName("label_15")
        pyEMsoftMW.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(pyEMsoftMW)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1100, 30))
        self.menubar.setObjectName("menubar")
        pyEMsoftMW.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(pyEMsoftMW)
        self.statusbar.setObjectName("statusbar")
        pyEMsoftMW.setStatusBar(self.statusbar)
        self.actionOpenMaster = QtWidgets.QAction(pyEMsoftMW)
        self.actionOpenMaster.setCheckable(True)
        self.actionOpenMaster.setObjectName("actionOpenMaster")
        self.actionOpen_Config = QtWidgets.QAction(pyEMsoftMW)
        self.actionOpen_Config.setCheckable(True)
        self.actionOpen_Config.setObjectName("actionOpen_Config")
        self.actionSave_As = QtWidgets.QAction(pyEMsoftMW)
        self.actionSave_As.setCheckable(True)
        self.actionSave_As.setObjectName("actionSave_As")

        self.retranslateUi(pyEMsoftMW)
        self.Load_Master.clicked.connect( lambda: self.OpenFile( self.Load_Master.objectName() ) )
        self.Load_Exp.clicked.connect( lambda: self.OpenFile( self.Load_Exp.objectName() ) )
        self.CCW.clicked.connect( lambda: self.RotateEuler( self.CCW.objectName() ) )
        self.UP.clicked.connect( lambda: self.RotateEuler( self.UP.objectName() ) )
        self.CW.clicked.connect( lambda: self.RotateEuler( self.CW.objectName() ) )
        self.RIGHT.clicked.connect( lambda: self.RotateEuler( self.RIGHT.objectName() ) )
        self.DOWN.clicked.connect( lambda: self.RotateEuler( self.DOWN.objectName() ) )
        self.LEFT.clicked.connect( lambda: self.RotateEuler( self.LEFT.objectName() ) )
        self.StartFit.clicked.connect( lambda: self.FitPattern( ) )
        self.phi1_Val.editingFinished.connect( lambda: self.ChangeParam( self.phi1_Val.objectName() ) )
        self.PHI_Val.editingFinished.connect( lambda: self.ChangeParam( self.PHI_Val.objectName() ) )
        self.phi2_Val.editingFinished.connect( lambda: self.ChangeParam( self.phi2_Val.objectName() ) )
        self.Tilt_Val.editingFinished.connect( lambda: self.ChangeParam( self.Tilt_Val.objectName() ) )
        self.PCY_Val.editingFinished.connect( lambda: self.ChangeParam( self.PCY_Val.objectName() ) )
        self.PCX_Val.editingFinished.connect( lambda: self.ChangeParam( self.PCX_Val.objectName() ) )
        self.Omega_Val.editingFinished.connect( lambda: self.ChangeParam( self.Distance_Val.objectName() ) )
        self.Distance_Val.editingFinished.connect( lambda: self.ChangeParam( self.Distance_Val.objectName() ) )
        self.Pixel_Size.editingFinished.connect( lambda: self.ChangeParam( self.Pixel_Size.objectName() ) )
        self.Beam_Current.editingFinished.connect( lambda: self.ChangeParam( self.Beam_Current.objectName() ) )
        self.Dwel.editingFinished.connect( lambda: self.ChangeParam( self.Dwel.objectName() ) )
        self.Distance_Step.textChanged.connect( lambda: self.ChangeStep( self.Distance_Step.objectName() ) )
        self.Omega_Step.textChanged.connect( lambda: self.ChangeStep( self.Omega_Step.objectName() ) )
        self.PCX_Step.textChanged.connect( lambda: self.ChangeStep( self.PCX_Step.objectName() ) )
        self.PCY_Step.textChanged.connect( lambda: self.ChangeStep( self.PCY_Step.objectName() ) )
        self.Tilt_Step.textChanged.connect( lambda: self.ChangeStep( self.Tilt_Step.objectName() ) )
        self.phi1_Step.textChanged.connect( lambda: self.ChangeStep( self.phi1_Step.objectName() ) )
        self.PHI_Step.textChanged.connect( lambda: self.ChangeStep( self.PHI_Step.objectName() ) )
        self.phi2_Step.textChanged.connect( lambda: self.ChangeStep( self.phi2_Step.objectName() ) )            
        self.IS_TSL.clicked.connect( lambda: self.ChangeParam( self.IS_TSL.objectName() ) )
        self.DetectorBinning.textChanged.connect( lambda: self.ChangeParam( self.DetectorBinning.objectName() ) )
        self.Gamma_Val.valueChanged.connect( lambda: self.ChangeParam( self.Gamma_Val.objectName() ) )
        self.Gamma_Step.textChanged.connect( lambda: self.ChangeParam( self.Gamma_Step.objectName() ) )
        self.Load_JSON.clicked.connect( lambda: self.OpenFile( self.Load_JSON.objectName() ) )
        self.Save_JSON.clicked.connect( lambda: self.SaveFile( self.Save_JSON.objectName() ) )        
        self.LinearBack.clicked.connect( lambda: self.ChangeParam( self.LinearBack.objectName() ) )
        QtCore.QMetaObject.connectSlotsByName(pyEMsoftMW)       
        
    @QtCore.pyqtSlot( )
    def ChangeParam( self, objectName ):
        pass
    
    @QtCore.pyqtSlot( )
    def ChangeStep( self, objectName ):
        pass    

    @QtCore.pyqtSlot( )
    def ChangeCamera( self, objectName ):
        pass    

    @QtCore.pyqtSlot( )
    def ReturnName( self, objectName ):
        pass
    
    @QtCore.pyqtSlot( )
    def RotateEuler( self, objectName ):
        pass

    @QtCore.pyqtSlot( )
    def FitPattern( self ):
        pass

    @QtCore.pyqtSlot( )
    def OpenFile( self, ObjectName ):
        pass
    
    @QtCore.pyqtSlot( )
    def SaveFile( self, ObjectName ):
        pass   


    def retranslateUi(self, pyEMsoftMW):
        _translate = QtCore.QCoreApplication.translate
        pyEMsoftMW.setWindowTitle(_translate("pyEMsoftMW", "MainWindow"))
        self.StartFit.setText(_translate("pyEMsoftMW", "Fit"))
        self.label.setText(_translate("pyEMsoftMW", "Refine"))
        self.label_2.setText(_translate("pyEMsoftMW", "phi1"))
        self.label_3.setText(_translate("pyEMsoftMW", "PHI"))
        self.label_4.setText(_translate("pyEMsoftMW", "phi2"))
        self.Load_Master.setText(_translate("pyEMsoftMW", "Load Master Pattern"))
        self.Load_Exp.setText(_translate("pyEMsoftMW", "Load Experimental Pattern"))
        self.DOWN.setText(_translate("pyEMsoftMW", "Down"))
        self.UP.setText(_translate("pyEMsoftMW", "Up"))
        self.RIGHT.setText(_translate("pyEMsoftMW", "Right"))
        self.LEFT.setText(_translate("pyEMsoftMW", "Left"))
        self.CW.setText(_translate("pyEMsoftMW", "CW"))
        self.CCW.setText(_translate("pyEMsoftMW", "CCW"))
        self.label_5.setText(_translate("pyEMsoftMW", "TSL Convention"))
        self.label_9.setText(_translate("pyEMsoftMW", "Distance"))
        self.label_10.setText(_translate("pyEMsoftMW", "Omega"))
        self.label_12.setText(_translate("pyEMsoftMW", "Tilt"))
        self.label_13.setText(_translate("pyEMsoftMW", "PCY"))
        self.label_6.setText(_translate("pyEMsoftMW", "Sample Parameters"))
        self.label_7.setText(_translate("pyEMsoftMW", "Detector Parameters"))
        self.label_8.setText(_translate("pyEMsoftMW", "Refine"))
        self.Dwel_L.setText(_translate("pyEMsoftMW", "Dwell Time (mu s)"))
        self.Current_L.setText(_translate("pyEMsoftMW", "Beam Current (nA)"))
        self.Pixel_Size_L.setText(_translate("pyEMsoftMW", "Pixel Size (micron)"))
        self.label_20.setText(_translate("pyEMsoftMW", "Fit Parameters"))
        self.label_24.setText(_translate("pyEMsoftMW", "PCX"))
        self.Binning_L.setText(_translate("pyEMsoftMW", "Detector Binning"))
        self.label_25.setText(_translate("pyEMsoftMW", "gamma"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("pyEMsoftMW", "Simulated"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("pyEMsoftMW", "Experimental"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("pyEMsoftMW", "Difference"))
        self.Load_JSON.setText(_translate("pyEMsoftMW", "Load Config"))
        self.Save_JSON.setText(_translate("pyEMsoftMW", "Save Config"))
        self.label_16.setText(_translate("pyEMsoftMW", "Max Iterations"))
        self.label_18.setText(_translate("pyEMsoftMW", "Error"))
        self.Binning_L_2.setText(_translate("pyEMsoftMW", "(1, 2, 4, 8)"))
        self.label_11.setText(_translate("pyEMsoftMW", "Value"))
        self.label_14.setText(_translate("pyEMsoftMW", "Step"))
        self.label_21.setText(_translate("pyEMsoftMW", "Value"))
        self.label_22.setText(_translate("pyEMsoftMW", "Step"))
        self.Dwel_L_2.setText(_translate("pyEMsoftMW", "linear background"))
        self.label_15.setText(_translate("pyEMsoftMW", "Angle Step"))
        self.actionOpenMaster.setText(_translate("pyEMsoftMW", "OpenMaster"))
        self.actionOpen_Config.setText(_translate("pyEMsoftMW", "Open Config"))
        self.actionOpen_Config.setShortcut(_translate("pyEMsoftMW", "Meta+O, Ctrl+O"))
        self.actionSave_As.setText(_translate("pyEMsoftMW", "Save As"))
        self.actionSave_As.setShortcut(_translate("pyEMsoftMW", "Ctrl+S, Meta+S"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    pyEMsoftMW = QtWidgets.QMainWindow()
    ui = Ui_pyEMsoftMW()
    ui.setupUi(pyEMsoftMW)
    pyEMsoftMW.show()
    sys.exit(app.exec_())
