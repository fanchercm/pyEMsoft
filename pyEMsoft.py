#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:33:10 2019

@author: hcf
"""

from __future__ import division, absolute_import, print_function
from builtins import super

from PyQt5 import QtWidgets
from PyQt5.QtCore import QRunnable, QThreadPool

from MainWindow import Ui_pyEMsoftMW
from pyEMsoftPlotting import Qt5MplCanvas


from os import path
from os.path import expanduser
home = expanduser("~")
import json
#from time import sleep
import numpy as np
import matplotlib.image
import sys
import lmfit


y = json.load( open('%s/.config/EMsoft/EMsoftConfig.json'%expanduser("~"), 'r') )['EMsoftLibraryLocation']

from pyEMsoftInterface import EMsoftInterface

class ProcessRunnable(QRunnable):
    def __init__(self, target, plotter):
        QRunnable.__init__(self)
        self.Sim    = target
        self.Plot   = plotter

    def run(self):
        self.Sim.CalculateEBSP()
        self.Plot._update_canvas( self.Sim.genericEBSDPatterns )
        
    def start(self):
        QThreadPool.globalInstance().start(self)
        
class MainWindowUIClass( Ui_pyEMsoftMW ):
    def __init__( self ):
        '''Initialize the super class
        '''
        super().__init__()

        self.EMsoft     = EMsoftInterface( json.load( open('%s/.config/EMsoft/EMsoftConfig.json'%expanduser("~"), 'r') )['EMsoftLibraryLocation'] )
        self.MasterFile = None
        self.ExpFile    = None
        self.DetBinning = np.array( [1,2,4,8],dtype=np.int32 )
        self.ExpPatData = None             
        
    def setupUi( self, MW ):
        ''' Setup the UI of the super class, and add here code
        that relates to the way we want our UI to operate.
        '''
        super().setupUi( MW )
            
        self.Plotter = Qt5MplCanvas(self.PlotWidget)
        self.Plotter2 = Qt5MplCanvas(self.ExpPlotWidget)
        self.Plotter3 = Qt5MplCanvas(self.ExpPlotWidget_2)

#       Initalize Default Steps

        self.phi1_Step.setText( "1" )
        self.PHI_Step.setText( "1" )
        self.phi2_Step.setText( "1" )
        self.Tilt_Step.setText( "0.5" )
        self.PCY_Step.setText( "5" )
        self.PCX_Step.setText( "5" )
        self.Omega_Step.setText( "0.5" )
        self.Distance_Step.setText( "250" )
        self.Gamma_Step.setText( "0.01" )
        self.Error.setText("0.000001")
        self.MaxIts.setText("500")
        self.Togle_Step.setText("5.0")

        self.DetectorBinning.setText( "1" )

#       Initalize Min Max values
        self.phi1_Val.setMinimum(-360.)
        self.phi1_Val.setMaximum(360.)
        self.phi1_Val.setSingleStep( np.float( self.phi1_Step.text() ) )
        
        self.PHI_Val.setMinimum(-360.)
        self.PHI_Val.setMaximum(360.)
        self.PHI_Val.setSingleStep( np.float( self.PHI_Step.text() ) )

        self.phi2_Val.setMinimum(-360.)
        self.phi2_Val.setMaximum(360.)   
        self.phi2_Val.setSingleStep( np.float( self.phi2_Step.text() ) )

        self.Tilt_Val.setMinimum(-5.0)
        self.Tilt_Val.setMaximum(15.0)
        self.Tilt_Val.setValue(10.0)
        self.Tilt_Val.setSingleStep( np.float( self.Tilt_Step.text() ) )

        self.PCY_Val.setMinimum(-800.)
        self.PCY_Val.setMaximum(800.)
        self.PCY_Val.setSingleStep( np.float( self.PCY_Step.text() ) )

        self.PCX_Val.setMinimum(-800.)
        self.PCX_Val.setMaximum(800.)
        self.PCX_Val.setSingleStep( np.float( self.PCX_Step.text() ) )

        self.Omega_Val.setMinimum(-5.)
        self.Omega_Val.setMaximum(5.)
        self.Omega_Val.setSingleStep( np.float( self.Omega_Step.text() ) )

        self.Distance_Val.setMinimum(10000.)
        self.Distance_Val.setMaximum(25000.)
        self.Distance_Val.setSingleStep( np.float( self.Distance_Step.text() ) )
        
        self.Gamma_Val.setMinimum(0.01)
        self.Gamma_Val.setMaximum(1.0)
        self.Gamma_Val.setSingleStep( np.float( self.Gamma_Step.text() ) )        

#       Initalize Detector Data
        self.Pixel_Size.setText( "25" )
        self.Beam_Current.setText( "100." )
        self.Dwel.setText( "100." )    
        
        # close the lower part of the splitter to hide the 
        # debug window under normal operations
        #self.splitter.setSizes([300, 0])

    def debugPrint( self, msg ):
        '''Print the message in the text edit at the bottom of the
        horizontal splitter.
        '''
        self.debugWindow.append( msg )
            

    def UpdateOrient(self):
        EU = np.array([self.phi1_Val.value(), self.PHI_Val.value(), self.phi2_Val.value()], dtype=np.float32)
        if not self.IS_TSL.checkState(): EU[0]+=90.        
        self.EMsoft.UpdateQuat( EU  )
        
        
    def ChangeStep( self, objectName ):
        if objectName in 'phi1_Step':
            self.phi1_Val.setSingleStep( np.float( self.phi1_Step.text() ) )     
        elif objectName == 'PHI_Step':
            self.PHI_Val.setSingleStep( np.float( self.PHI_Step.text() ) )     
        elif objectName == 'phi2_Step':
            self.phi2_Val.setSingleStep( np.float( self.phi2_Step.text() ) )                            
        elif objectName == 'Omega_Step':
            self.Omega_Val.setSingleStep( np.float( self.Omega_Step.text() ) )     
        elif objectName == 'PCX_Step':
            self.PCX_Val.setSingleStep( np.float( self.PCX_Step.text() ) )                   #  // pattern center x component (in pixel units)
        elif objectName == 'PCY_Step':
            self.PCY_Val.setSingleStep( np.float( self.PCY_Step.text() ) )                   # // pattern center y component (in pixel units)        
        elif objectName == 'Tilt_Step':
            self.Tilt_Val.setSingleStep( np.float( self.Tilt_Step.text() ) )                        # // detector tilt angle (degrees) from horizontal (positive for detector looking upwards)
        elif objectName == 'Distance_Step':
            self.Distance_Val.setSingleStep( np.float( self.Distance_Step.text() ) )                # // sample-scintillator distance (microns)
        elif objectName == 'Gamma_Step':
            self.Gamma_Val.setSingleStep( np.float( self.Gamma_Step.text() ) )                       # // intensity scaling gamma value                   
                
    def ChangeParam( self, objectName ):
        if  self.MasterFile:            
            if objectName in ['phi1_Val', 'PHI_Val', 'phi2_Val', 'IS_TSL']:
                self.UpdateOrient()            
            elif objectName == 'DetectorBinning':
                self.EMsoft.genericIPar[21] = np.int( self.DetectorBinning.text() )             # Detector Binning
            elif objectName == 'Omega_Val':
                self.EMsoft.genericFPar[1] = self.Omega_Val.value()
            elif objectName == 'PCX_Val':
                self.EMsoft.genericFPar[14] = self.PCX_Val.value()              #  // pattern center x component (in pixel units)
            elif objectName == 'PCY_Val':
                self.EMsoft.genericFPar[15] = self.PCY_Val.value()              # // pattern center y component (in pixel units)        
            elif objectName == 'Pixel_Size':
                self.EMsoft.genericFPar[16] = np.float32( self.Pixel_Size.text() )                  # // pixel size (microns) on scintillator surface
            elif objectName == 'Tilt_Val':
                self.EMsoft.genericFPar[17] = self.Tilt_Val.value()                   # // detector tilt angle (degrees) from horizontal (positive for detector looking upwards)
            elif objectName == 'Distance_Val':
                self.EMsoft.genericFPar[18] = self.Distance_Val.value()           # // sample-scintillator distance (microns)
            elif objectName == 'Beam_Current':
                self.EMsoft.genericFPar[19] = np.float32( self.Beam_Current.text() )                  # // beam current [nA]
            elif objectName == 'Dwel':
                self.EMsoft.genericFPar[20] = np.float32( self.Dwel.text() )                   # // beam dwell time per pattern [micro-seconds]
            elif objectName == 'Gamma_Val':
                self.EMsoft.genericFPar[21] = self.Gamma_Val.value()                  # // intensity scaling gamma value
            elif objectName == 'LinearBack':
                if self.LinearBack.isChecked():
                    self.EMsoft.Background()
                else:
                    self.EMsoft.BackVal = None
                                
            self.background = ProcessRunnable(self.EMsoft, self.Plotter)                
            self.background.start()
        
#            sleep(1)
#            self.EMsoft.CalculateEBSP()
#            self.Plotter._update_canvas( self.EMsoft.genericEBSDPatterns )
#            if self.ExpFile and self.MasterFile:
#                self.Plotter3.difference( self.EMsoft.genericEBSDPatterns, self.ExpFile )
          
    def RotateEuler( self, objectName ):
        
        self.UpdateOrient()
        
        ang = np.deg2rad( np.float(self.Togle_Step.text() ) )/2.
        cang = np.cos(ang);
        sang = np.sin(ang);
        eta =  ( self.EMsoft.sig - self.Tilt_Val.value()) * np.pi / 180.
        delta = np.pi * 0.5 - eta;
        ceta = np.cos(eta);
        seta = np.sin(eta);
        cdelta = np.cos(delta);
        sdelta = np.sin(delta);

        X_QUAT = np.array([cang, 0.0, sang, 0.0])
#        Y_QUAT = np.array([cang, 1./np.sqrt(2)*sang, 0.0, 1./np.sqrt(2)*sang])
#        Z_QUAT = np.array([cang, 1./np.sqrt(2)*sang, 1./np.sqrt(2)*sang, 1./np.sqrt(2)*sang])

        Y_QUAT = np.array([cang, sang * cdelta, 0.0, -sang * sdelta])
        Z_QUAT = np.array([cang, sang * ceta, 0.0, sang * seta])

                
        if objectName == 'CW':
            QUAT = Z_QUAT
        elif objectName == 'CCW':
            QUAT = Z_QUAT
            QUAT = np.array( [QUAT[0], -1*QUAT[1], -1*QUAT[2], -1*QUAT[3]] )
        elif objectName == 'UP':
            QUAT = X_QUAT
        elif objectName == 'DOWN':
            QUAT = X_QUAT
            QUAT = np.array( [QUAT[0], -1*QUAT[1], -1*QUAT[2], -1*QUAT[3]] )
        elif objectName == 'LEFT':
            QUAT = Y_QUAT
            QUAT = np.array( [QUAT[0], -1*QUAT[1], -1*QUAT[2], -1*QUAT[3]] )            
        elif objectName == 'RIGHT':
            QUAT = Y_QUAT
            
        q1w, q1x, q1y, q1z = self.EMsoft.genericQuaternions
        q2w, q2x, q2y, q2z = QUAT
        
        
        quat = np.array([q2x * q1w + q2w * q1x + q2z * q1y - q2y * q1z,
                         q2y * q1w + q2w * q1y + q2x * q1z - q2z * q1x,
                         q2z * q1w + q2w * q1z + q2y * q1x - q2x * q1y,
                         q2w * q1w - q2x * q1x - q2y * q1y - q2z * q1z])
      

        eu = self.EMsoft.QU2EU( quat ) * 180. / np.pi
        
        if not self.IS_TSL.checkState(): eu[0]-=90.        

        self.phi1_Val.setValue(eu[0])
        self.PHI_Val.setValue(eu[1])
        self.phi2_Val.setValue(eu[2])
        self.UpdateOrient()

        self.background = ProcessRunnable(self.EMsoft, self.Plotter)                
        self.background.start()
            
        # slot
    def SaveFile( self, ObjectName ):
        print ('ObjectName')
        if ObjectName == 'Load_Master':
            AllowedFiles = "All Files (*);;HDF5 (*.h5)"
        elif ObjectName == 'Load_Exp':
            AllowedFiles = "All Files (*);;PNG (*.png);;TIFF (*.tif,*.tiff)"
        else:
            AllowedFiles = "All Files (*.json)"
            

        ''' Called when the user presses the Browse button
        '''
        #self.debugPrint( "Browse button pressed" )
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
                        None,
                        "QFileDialog.getOpenFileName()",
                        "",
                        AllowedFiles,
                        options=options)    

        filename, file_extension = path.splitext(fileName)
        
        data = {}  
        data['Gamma_Val'] = self.Gamma_Val.value()
        data['Dwel'] = self.Dwel.text() 
        data['Beam_Current'] = self.Beam_Current.text() 
        data['Distance_Val'] = self.Distance_Val.value()
        data['Tilt_Val'] = self.Tilt_Val.value()
        data['Pixel_Size'] = self.Pixel_Size.text()
        data['PCY_Val'] = self.PCY_Val.value()
        data['PCX_Val'] = self.PCX_Val.value()
        data['Omega_Val'] = self.Omega_Val.value()
        data['DetectorBinning'] =  self.DetectorBinning.text() 
        data['phi1_Val'] = self.phi1_Val.value()
        data['PHI_Val'] = self.PHI_Val.value()
        data['phi2_Val'] = self.phi2_Val.value()
        data['IS_TSL'] = self.IS_TSL.isChecked()
        data['MasterFile'] = self.MasterFile
        data['ExpFile'] = self.ExpFile
        
        with open( "%s.json"%filename, 'w') as outfile:  
            json.dump(data, outfile)
            
    def LoadConfig( self, FileName ):
       
        data = json.load( open( FileName ) )
                           
        for item in list( data.keys() ):
            if item == 'Gamma_Val':
                self.Gamma_Val.setValue( data['Gamma_Val'] )
            elif item == 'Dwel':                
                self.Dwel.setText( data['Dwel'] ) 
            elif item == 'Dwel':                
                self.Beam_Current.setText( data['Beam_Current'] )
            elif item == 'Distance_Val':                        
                self.Distance_Val.setValue( data['Distance_Val'] )
            elif item == 'Tilt_Val':                
                self.Tilt_Val.setValue( data['Tilt_Val'] )
            elif item == 'Pixel_Size':                
                self.Pixel_Size.setText( data['Pixel_Size'] )
            elif item == 'PCY_Val':                
                self.PCY_Val.setValue( data['PCY_Val'] )
            elif item == 'PCX_Val':                
                self.PCX_Val.setValue( data['PCX_Val'] )
            elif item == 'Omega_Val':                
                self.Omega_Val.setValue( data['Omega_Val'] )
            elif item == 'DetectorBinning':                
                self.DetectorBinning.setText( data['DetectorBinning'] )
            elif item == 'phi1_Val':                
                self.phi1_Val.setValue( data['phi1_Val'] )
            elif item == 'PHI_Val':                
                self.PHI_Val.setValue( data['PHI_Val'] )
            elif item == 'phi2_Val':                
                self.phi2_Val.setValue( data['phi2_Val'] )
#        self.IS_TSL.setValue( data['IS_TSL'] )
            elif item == 'MasterFile':                
                self.MasterFile = data['MasterFile']
                self.SetupPatterns('Load_Master') 

            elif item == 'ExpFile':                
                self.ExpFile = data['ExpFile'] 
                self.SetupPatterns('Load_Exp')
                
            
        if self.ExpFile and self.MasterFile:
            self.Plotter3.difference( self.EMsoft.genericEBSDPatterns, self.ExpPatData )
        
    def SetupPatterns(self, caller):
        if caller == 'Load_Master':
            self.debugPrint( "Loading master file: " + self.MasterFile )             
            self.UpdateOrient()
            self.MasterFile_Disp.setText( self.MasterFile )                           
            self.EMsoft.ReadMasterData( self.MasterFile )
            self.EMsoft.setInputs(np.int( self.DetectorBinning.text() ), self.Omega_Val.value(), self.PCX_Val.value(), self.PCY_Val.value(),np.float32( self.Pixel_Size.text() ), self.Tilt_Val.value(), self.Distance_Val.value(), np.float32( self.Beam_Current.text() ), np.float32( self.Dwel.text() ), self.Gamma_Val.value())
            self.EMsoft.CalculateEBSP()
            self.ROIS = []
            H = self.EMsoft.genericIPar[22] / 10.
            W = self.EMsoft.genericIPar[23] / 10.
            for i in range(4):
                for j in range(8):
                    self.ROIS.append( [ np.int((0.5+1*j)*H), np.int((1+1*i)*W), np.int((2.5+1*j)*H), np.int((3+1*i)*W)] )
                    
            self.Plotter._update_canvas( self.EMsoft.genericEBSDPatterns, [])
        else:
            self.debugPrint( "Loading experimental pattern: " + self.ExpFile ) 
            self.ExpFile_Disp.setText( self.ExpFile )
            self.ExpPatData = matplotlib.image.imread( self.ExpFile )
            self.Plotter2.addImage( self.ExpPatData )         
            self.pix_X, self.pix_Y = self.ExpPatData.shape
            
            
    def OpenFile( self, ObjectName ):
        if ObjectName == 'Load_Master':
            AllowedFiles = "All Files (*);;HDF5 (*.h5)"
        elif ObjectName == 'Load_Exp':
            AllowedFiles = "All Files (*);;PNG (*.png);;TIFF (*.tif,*.tiff)"
        else:
            AllowedFiles = "All Files (*.json)"
            

        ''' Called when the user presses the Browse button
        '''
        #self.debugPrint( "Browse button pressed" )
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                        None,
                        "QFileDialog.getOpenFileName()",
                        "",
                        AllowedFiles,
                        options=options)        
        if fileName:
            if ObjectName == 'Load_Master':
                self.MasterFile = fileName
                self.SetupPatterns('Load_Master')                                        
            elif ObjectName == 'Load_Exp':
                self.ExpFile = fileName
                self.SetupPatterns('Load_Exp')                
            elif ObjectName == 'Load_JSON':
                self.debugPrint( "Loading configuration file: " + fileName ) 
                self.LoadConfig( fileName )
                
        if self.ExpFile and self.MasterFile:
            self.Plotter3.difference( self.EMsoft.genericEBSDPatterns, self.ExpPatData )

    def FitPatternLMFIT(self):                   
        
        dB   = self.ExpPatData.reshape(self.pix_X * self.pix_Y)
        dB   = dB / np.linalg.norm( dB )
        
        phi1, PHI, phi2 = [self.phi1_Val.value(), self.PHI_Val.value(), self.phi2_Val.value()]
        
        def CalcDot(params):           
            
            if self.Omega_Refine.isChecked(): self.EMsoft.genericFPar[1] = params['Distance_Refine'].value
            if self.PCX_Refine.isChecked():self.EMsoft.genericFPar[14] = params['PCX_Refine'].value
            if self.PCY_Refine.isChecked():self.EMsoft.genericFPar[15] = params['PCY_Refine'].value
            if self.Tilt_Refine.isChecked():self.EMsoft.genericFPar[17] = params['Tilt_Refine'].value
            if self.Distance_Refine.isChecked():self.EMsoft.genericFPar[18] = params['Distance_Refine'].value
            if self.Gamma_Refine.isChecked():self.EMsoft.genericFPar[21] = params['Gamma_Refine'].value
            
            if self.phi1_Refine.isChecked():phi1 = params['phi1_Refine'].value
            if self.PHI_Refine.isChecked():PHI = params['PHI_Refine'].value
            if self.phi2_Refine.isChecked():phi2 = params['phi2_Refine'].value
                
            self.EMsoft.UpdateOrient(phi1, PHI, phi2, self.IS_TSL.checkState() )
            self.EMsoft.CalculateEBSP()
            
            dA   = self.EMsoft.genericEBSDPatterns.reshape(self.pix_X * self.pix_Y)
            dA   = dA / np.linalg.norm( dA )
            
                    
            return 1. / np.dot( dA, dB)        
        
        params = lmfit.Parameters( )

        if self.Distance_Refine.isChecked():params.add('Distance_Refine', value=self.Distance_Val.value(), min=self.Distance_Val.value()-2.*np.float(self.Distance_Step.text()), max=self.Distance_Val.value()+2.*np.float(self.Distance_Step.text()) )            
        if self.Omega_Refine.isChecked():params.add('Omega_Refine', value=self.Omega_Val.value(), min=-5.0, max=5.0)
        if self.PCX_Refine.isChecked():params.add('PCX_Refine', value=self.PCX_Val.value() )
        if self.PCY_Refine.isChecked():params.add('PCY_Refine', value=self.PCY_Val.value() )
        if self.Tilt_Refine.isChecked():params.add('Tilt_Refine', value=self.Tilt_Val.value(), min=0., max=15.0)
        if self.Gamma_Refine.isChecked():params.add('Gamma_Refine', value=self.Gamma_Val.value(), min=0.001, max=1.)
        if self.phi1_Refine.isChecked():params.add('phi1_Refine', value=self.phi1_Val.value(), min=0., max=360.)
        if self.PHI_Refine.isChecked():params.add('PHI_Refine', value=self.PHI_Val.value(), min=0., max=360.)
        if self.phi2_Refine.isChecked():params.add('phi2_Refine', value=self.phi2_Val.value(), min=0., max=360.)
        

        out = lmfit.minimize( CalcDot, params, args=(phi1, PHI, phi2) )

        print (out)
        
    def FitPattern(self):                   
        from scipy.optimize import basinhopping as leastsq
#        from scipy.optimize import leastsq

        Error = np.zeros( len(self.ROIS) )
        
        ROI_Data = []
        for roi in self.ROIS:
            x1,y1, x2,y2 = roi       
            dB   = self.ExpPatData[y1:y2,x1:x2].reshape( (x2-x1) * (y2-y1) )
            ROI_Data.append( dB / np.linalg.norm( dB ) )
        
        phi1, PHI, phi2 = [self.phi1_Val.value(), self.PHI_Val.value(), self.phi2_Val.value()]
                
        def CalcDot(B):
            #, varyList, phi1, PHI, phi2):
            params = dict(zip(varyList,B))

            Error = 0
            phi1, PHI, phi2 = [self.phi1_Val.value(), self.PHI_Val.value(), self.phi2_Val.value()]
            if self.Omega_Refine.isChecked(): self.EMsoft.genericFPar[1] = params['Omega_Refine']
            if self.PCX_Refine.isChecked():self.EMsoft.genericFPar[14] = params['PCX_Refine']
            if self.PCY_Refine.isChecked():self.EMsoft.genericFPar[15] = params['PCY_Refine']
            if self.Tilt_Refine.isChecked():self.EMsoft.genericFPar[17] = params['Tilt_Refine']
            if self.Distance_Refine.isChecked():self.EMsoft.genericFPar[18] = params['Distance_Refine']
            if self.Gamma_Refine.isChecked():self.EMsoft.genericFPar[21] = params['Gamma_Refine']
            
            if self.phi1_Refine.isChecked():phi1 = params['phi1_Refine']
            if self.PHI_Refine.isChecked():PHI = params['PHI_Refine']
            if self.phi2_Refine.isChecked():phi2 = params['phi2_Refine']
                
            self.EMsoft.UpdateOrient(phi1, PHI, phi2, self.IS_TSL.checkState() )
            self.EMsoft.CalculateEBSP()            
            
            for i in range(len(self.ROIS)):
                x1,y1, x2,y2 = self.ROIS[i]
                
                dA   = self.EMsoft.genericEBSDPatterns[x1:x2,y1:y2].reshape( (x2-x1) * (y2-y1) )
                dA   = dA / np.linalg.norm( dA )
                
                Error += (1. / np.dot( dA, ROI_Data[i]))**2
#                Error[i] = (1. / np.dot( dA, ROI_Data[i]))**2
            
            #print (Error)
            return Error
        
        
        p0 = []
        varyList = []
        
        if self.Distance_Refine.isChecked():
            varyList.append('Distance_Refine')
            p0.append(self.Distance_Val.value())
        if self.Omega_Refine.isChecked():
            varyList.append('Omega_Refine')
            p0.append(self.Omega_Val.value())        
        if self.PCX_Refine.isChecked():
            varyList.append('PCX_Refine')
            p0.append(self.PCX_Val.value())
        if self.PCY_Refine.isChecked():
            varyList.append('PCY_Refine')
            p0.append(self.PCY_Val.value())    
        if self.Tilt_Refine.isChecked():
            varyList.append('Tilt_Refine')
            p0.append(self.Tilt_Val.value())
        if self.Gamma_Refine.isChecked():
            varyList.append('Gamma_Refine')
            p0.append(self.Gamma_Val.value())
        if self.phi1_Refine.isChecked():
            varyList.append('phi1_Refine')
            p0.append(self.phi1_Val.value())
        if self.PHI_Refine.isChecked():
            varyList.append('PHI_Refine')
            p0.append(self.PHI_Val.value())
        if self.phi2_Refine.isChecked():
            varyList.append('phi2_Refine')
            p0.append(self.phi2_Val.value())     
            
        result = leastsq(CalcDot,p0 )
        #,args=(varyList, phi1, PHI, phi2) )
        #, max_nfev=np.int( self.MaxIts.text() ),  gtol=np.float( self.Error.text() ))  


#        print (result)        
                
        params = dict(zip(varyList,result.x))
#        params = dict(zip(varyList,result[0]))
        
        if self.Omega_Refine.isChecked(): 
            self.EMsoft.genericFPar[1] = params['Omega_Refine']
            self.Omega_Val.setValue( params['Omega_Refine'] )                   
        if self.PCX_Refine.isChecked():
            self.EMsoft.genericFPar[14] = params['PCX_Refine']
            self.PCX_Val.setValue( params['PCX_Refine'] )
        if self.PCY_Refine.isChecked():
            self.EMsoft.genericFPar[15] = params['PCY_Refine']
            self.PCY_Val.setValue( params['PCY_Refine'] )
        if self.Tilt_Refine.isChecked():
            self.EMsoft.genericFPar[17] = params['Tilt_Refine']
            self.Tilt_Val.setValue( params['Tilt_Refine'] )
        if self.Distance_Refine.isChecked():
            self.EMsoft.genericFPar[18] = params['Distance_Refine']
            self.Distance_Val.setValue( params['Distance_Refine'] )
        if self.Gamma_Refine.isChecked():
            self.EMsoft.genericFPar[21] = params['Gamma_Refine']
            self.Gamma_Val.setValue( params['Gamma_Refine'] )
                    
        phi1, PHI, phi2 = [self.phi1_Val.value(), self.PHI_Val.value(), self.phi2_Val.value()]
            
        if self.phi1_Refine.isChecked():
            phi1 = params['phi1_Refine']
            self.phi1_Val.setValue( params['phi1_Refine'] )            
        if self.PHI_Refine.isChecked():
            PHI = params['PHI_Refine']
            self.PHI_Val.setValue( params['PHI_Refine'] )
        if self.phi2_Refine.isChecked():
            phi2 = params['phi2_Refine']        
            self.phi2_Val.setValue( params['phi2_Refine'] )

        self.EMsoft.UpdateOrient(phi1, PHI, phi2, self.IS_TSL.checkState() )

        self.EMsoft.CalculateEBSP()
        self.EMsoft.Background()
        self.Plotter._update_canvas( self.EMsoft.genericEBSDPatterns )
            
def main():
    """
    This is the MAIN ENTRY POINT of our application.  The code at the end
    of the mainwindow.py script will not be executed, since this script is now
    our main program.   We have simply copied the code from mainwindow.py here
    since it was automatically generated by '''pyuic5'''.

    """
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainWindowUIClass()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

main()