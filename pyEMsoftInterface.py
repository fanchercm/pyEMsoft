from ctypes import c_size_t, c_char_p, c_void_p
import h5py
import numpy as np
from numpy.ctypeslib import ndpointer


class EMsoftInterface():
    
    def __init__(self, libPath):
        try:
            self.lib = np.ctypeslib.load_library("libEMsoftWrapperLib.so", libPath) 
        except:
            self.lib = np.ctypeslib.load_library("libEMsoftWrapperLib.dylib", libPath) 
            
        self.genericEBSDPatterns = None
        self.BackVal = None
        
    def setPointers(self):
        self.FParType    = ndpointer(dtype=np.float32, ndim=1, shape=self.genericFPar.shape, flags='F_CONTIGUOUS')
        self.iParType    = ndpointer(dtype=np.int32, ndim=1, shape=self.genericIPar.shape, flags='F_CONTIGUOUS')
        self.Accume_eType = ndpointer(dtype=np.int32, ndim=3, shape=(self.monteCarlo_dims[2], self.monteCarlo_dims[1], self.monteCarlo_dims[0]), flags='F_CONTIGUOUS')
        self.LPNHType    = ndpointer(dtype=np.float32, ndim=4, shape=(self.mLPNH_dims[3], self.mLPNH_dims[2], self.mLPNH_dims[1], self.mLPNH_dims[0]), flags='F_CONTIGUOUS')                    
        self.LPSHType    = ndpointer(dtype=np.float32, ndim=4, shape=(self.mLPSH_dims[3], self.mLPSH_dims[2], self.mLPSH_dims[1], self.mLPSH_dims[0]), flags='F_CONTIGUOUS')
        self.QUATSType   = ndpointer(dtype=np.float32, ndim=1, shape=( 4 ))
        self.EulerType   = ndpointer(dtype=np.float32, ndim=1, shape=( 3 ))

        self.lib.EMsoftCgetEBSDPatterns.restype = None
#        self.lib.__rotations_MOD_eu2qu.restype = None
#        self.lib.__rotations_MOD_eu2qu.argtypes = [self.EulerType, self.QUATSType]
#        self.lib.__rotations_MOD_qu2eu.restype = None
#        self.lib.__rotations_MOD_qu2eu.argtypes = [self.QUATSType, self.EulerType]

    def setInputs(self, Binning_Val, Omega_Val, PCX_Val, PCY_Val, Pixel_Size, Tilt_Val, Distance_Val, Beam_Current, Dwel, Gamma_Val):
        
        self.genericIPar = np.zeros(40, dtype=np.int32)
        
        self.genericIPar[0] =  np.int32( (self.numsx - 1) / 2 )
        self.genericIPar[8] = self.numset
        self.genericIPar[11] = ((self.incidentBeamVoltage - self.minEnergy) / self.energyBinSize) + 1
        self.genericIPar[16] = self.npx
        self.genericIPar[17] = 4
        self.genericIPar[18] = 1344          # Number of Pixels x
        self.genericIPar[19] = 1024          # Number of Pixels y
        self.genericIPar[20] = 1             # Number of Angles
        self.genericIPar[21] = Binning_Val             # Detector Binning
        self.genericIPar[22] = np.int( self.genericIPar[18] / self.genericIPar[21] )
        self.genericIPar[23] = np.int( self.genericIPar[19] / self.genericIPar[21] )
        self.genericIPar[24] = 0
        
        self.genericFPar = np.zeros(40, dtype=np.float32)
        self.genericFPar[0] = self.sig
        self.genericFPar[1] = Omega_Val
        self.genericFPar[14] = PCX_Val              #  // pattern center x component (in pixel units)
        self.genericFPar[15] = PCY_Val              # // pattern center y component (in pixel units)        
        self.genericFPar[16] = Pixel_Size                   # // pixel size (microns) on scintillator surface
        self.genericFPar[17] = Tilt_Val                  # // detector tilt angle (degrees) from horizontal (positive for detector looking upwards)
        self.genericFPar[18] = Distance_Val           # // sample-scintillator distance (microns)
        self.genericFPar[19] = Beam_Current                  # // beam current [nA]
        self.genericFPar[20] = Dwel                   # // beam dwell time per pattern [micro-seconds]
        self.genericFPar[21] = Gamma_Val        
        
        self.genericAccum_e = np.zeros(self.monteCarlo_dims[2] * self.monteCarlo_dims[1] * self.monteCarlo_dims[0], dtype=np.int32 )
        self.genericAccum_e = self.genericAccum_e.reshape(self.monteCarlo_dims[2],self.monteCarlo_dims[1],self.monteCarlo_dims[0], order='F' )
        for i in range( self.monteCarlo_dims[2] ):
            self.genericAccum_e[i,:,:] = self.monteCarloSquareData[:,:,i].T
        
        self.genericLPNH = np.zeros(self.mLPNH_dims[3] * self.mLPNH_dims[2] * self.mLPNH_dims[1] * self.mLPNH_dims[0], dtype=np.float32 )
        self.genericLPSH = np.zeros(self.mLPSH_dims[3] * self.mLPSH_dims[2] * self.mLPSH_dims[1] * self.mLPSH_dims[0], dtype=np.float32 )
        self.genericLPNH = self.genericLPNH.reshape(self.mLPNH_dims[3], self.mLPNH_dims[2], self.mLPNH_dims[1], self.mLPNH_dims[0], order='F')
        self.genericLPSH = self.genericLPSH.reshape(self.mLPSH_dims[3], self.mLPSH_dims[2], self.mLPSH_dims[1], self.mLPSH_dims[0], order='F')
        
        for i in range( self.mLPNH_dims[0] ):
            for j in range( self.mLPNH_dims[1] ):
                self.genericLPNH[:,:,j,i] = self.masterLPNHData[i,j,:,:]
                self.genericLPSH[:,:,j,i] = self.masterLPSHData[i,j,:,:]
                               
        self.setPointers()
        
        
#    def EMsoftEU2QU(self, EA):
#        quat = np.array([0,0,0,0], dtype=np.float32)        
#        self.lib.__rotations_MOD_eu2qu(EA, quat)
#        
#    
#    def EMsoftQU2EU(self, Q):
#        EA = np.array([0,0,0], dtype=np.float32)
#        self.lib.__rotations_MOD_qu2eu(Q, EA)
#        return EA
    
    def ReadMasterData(self, DataFiles):

        FileRead    = h5py.File(DataFiles, 'r')

        self.masterLPNHData = FileRead['EMData']['EBSDmaster']['mLPNH'][:]
        self.mLPNH_dims = FileRead['EMData']['EBSDmaster']['mLPNH'].shape
        self.masterLPSHData = FileRead['EMData']['EBSDmaster']['mLPSH'][:]
        self.mLPSH_dims = FileRead['EMData']['EBSDmaster']['mLPSH'].shape
        self.masterSPNHData = FileRead['EMData']['EBSDmaster']['masterSPNH'][:]
        self.masterSPNH_dims = FileRead['EMData']['EBSDmaster']['masterSPNH'].shape
        self.numset = FileRead['EMData']['EBSDmaster']['numset'][0]
        self.ekevs = FileRead['EMData']['EBSDmaster']['EkeVs'][0]
        self.numMPEnergyBins = FileRead['EMData']['EBSDmaster']['numEbins'][0]
                
        self.monteCarloSquareData = FileRead['EMData']['MCOpenCL']['accum_e'][:]
        self.monteCarlo_dims = FileRead['EMData']['MCOpenCL']['accum_e'][:].shape
        self.numDepthBins = FileRead['EMData']['MCOpenCL']['numzbins'][0]
        self.numMCEnergyBins = FileRead['EMData']['MCOpenCL']['numzbins'][0]
        
        self.numsx = FileRead['NMLparameters']['MCCLNameList']['numsx'][0]
        self.sig   = FileRead['NMLparameters']['MCCLNameList']['sig'][0]
        self.mcStructureFileName = FileRead['NMLparameters']['MCCLNameList']['xtalname'][0]
        self.incidentBeamVoltage = FileRead['NMLparameters']['MCCLNameList']['EkeV'][0]
        self.mcMode = FileRead['NMLparameters']['MCCLNameList']['MCmode'][0]
        self.omega = FileRead['NMLparameters']['MCCLNameList']['omega'][0]
        self.sigma = FileRead['NMLparameters']['MCCLNameList']['sig'][0]
        self.minEnergy = FileRead['NMLparameters']['MCCLNameList']['Ehistmin'][0]
        self.maxEnergy = FileRead['NMLparameters']['MCCLNameList']['EkeV'][0]
        self.energyBinSize = FileRead['NMLparameters']['MCCLNameList']['Ebinsize'][0]
        self.maxDepth = FileRead['NMLparameters']['MCCLNameList']['depthmax'][0]
        self.depthStep = FileRead['NMLparameters']['MCCLNameList']['depthstep'][0]
        self.totalNumIncidentEl = FileRead['NMLparameters']['MCCLNameList']['totnum_el'][0]

        self.npx = FileRead['NMLparameters']['EBSDMasterNameList']['npx'][0]
        
    def Background(self):
        import scipy.ndimage as ndimage

        X = np.arange( self.genericIPar[22] )
        
        NumAverage = 32 
        XFit = np.mean( X.reshape(-1,NumAverage),1)
        Background = np.array( [] )
    
        img = ndimage.gaussian_filter(self.genericEBSDPatterns[:,:,0], sigma=(64, 64), order=0)

        for i in range( self.genericIPar[23] ):
            z = np.poly1d( np.polyfit(XFit, np.mean( img[:,i].reshape(-1,NumAverage),1), 2) )
#            z = np.poly1d( np.polyfit(X, self.genericEBSDPatterns[i,:,0], 2) )

            Background = np.concatenate( (Background, z(X)) )

        self.BackVal = Background.reshape(self.genericIPar[23], self.genericIPar[22] ).T

        
    def UpdateOrient(self, phi1_Val, PHI_Val, phi2_Val, IS_TSL):
        EU = np.array([phi1_Val, PHI_Val, phi2_Val], dtype=np.float32)
        if not IS_TSL: EU[0]+=90.        
        self.UpdateQuat( EU  )
        
    def EU2QU(self, e):
        e = np.deg2rad( e )
        w,x,y,z = 0,1,2,3;    
        ee = np.array( [0., 0., 0.], dtype=np.float32 )
        res = np.array( [0., 0., 0., 0.], dtype=np.float32 )
        cPhi = 0.0
        cp = 0.0
        cm = 0.0
        sPhi = 0.0
        sp = 0.0
        sm = 0.0
        
        ee[0] = 0.5 * e[0];
        ee[1] = 0.5 * e[1];
        ee[2] = 0.5 * e[2];
        
        cPhi = np.cos(ee[1]);
        sPhi = np.sin(ee[1]);
        cm = np.cos(ee[0] - ee[2]);
        sm = np.sin(ee[0] - ee[2]);
        cp = np.cos(ee[0] + ee[2]);
        sp = np.sin(ee[0] + ee[2]);
        
        res[w] = cPhi * cp;
        res[x] = -1 * sPhi * cm;
        res[y] = -1 * sPhi * sm;
        res[z] = -1 * cPhi * sp;
    
        if (res[w] < 0.0):
            res[w] = -res[w];
            res[x] = -res[x];
            res[y] = -res[y];
            res[z] = -res[z];
    
        return res
    
    def QU2EU(self, qq):    
        
        res = np.zeros(3)
        w = 3
        x = 0
        y = 1
        z = 2
        
        q03 = qq[w] * qq[w] + qq[z] * qq[z]
        q12 = qq[x] * qq[x] + qq[y] * qq[y]
        chi = np.sqrt(q03 * q12)
        if chi == 0.0:
            if q12 == 0.0:
                Phi = 0.0;
                phi2 = 0.0;                
                phi1 = np.arctan2(-2.0 * qq[w] * qq[z], qq[w] * qq[w] - qq[z] * qq[z]);
            else:
                Phi = np.pi
                phi2 = 0.0;                
                phi1 = np.arctan2(2.0 * qq[x] * qq[y], qq[x] * qq[x] - qq[y] * qq[y]);
        else:
            Phi = np.arctan2( 2.0 * chi, q03 - q12 );
            chi = 1.0 / chi;
            phi1 = np.arctan2((-qq[w] * qq[y] + qq[x] * qq[z]) * chi , (-qq[w] * qq[x] - qq[y] * qq[z]) * chi );
            phi2 = np.arctan2( (qq[w] * qq[y] + qq[x] * qq[z]) * chi, (-qq[w] * qq[x] + qq[y] * qq[z]) * chi );

        res[0] = phi1
        res[1] = Phi
        res[2] = phi2


        if res[0] < 0.0:
            res[0] = (res[0] + 100.0 * np.pi) % (2.*np.pi)
        if res[1] < 0.0:
            res[1] = (res[1] + 100.0 * np.pi) % (2.*np.pi)
        if res[2] < 0.0:
            res[2] = (res[2] + 100.0 * np.pi) % (2.*np.pi)
   
        return res
      
    def UpdateQuat(self, EA):
        self.genericQuaternions = self.EU2QU( EA )

    def CalculateEBSP(self):
        self.genericIPar[22]        = np.int( self.genericIPar[18] / self.genericIPar[21] )
        self.genericIPar[23]        = np.int( self.genericIPar[19] / self.genericIPar[21] )
        self.genericEBSDPatterns    = np.zeros(self.genericIPar[22] * self.genericIPar[23] * self.genericIPar[20], dtype=np.float32).reshape( self.genericIPar[22], self.genericIPar[23], self.genericIPar[20], order='F' )
        self.EBSDType               = ndpointer(dtype=np.float32, ndim=3, shape=(self.genericIPar[22], self.genericIPar[23], self.genericIPar[20]), flags='F_CONTIGUOUS')        
        
        self.lib.EMsoftCgetEBSDPatterns.argtypes = [self.iParType, self.FParType, self.EBSDType, self.QUATSType, self.Accume_eType, self.LPNHType, self.LPSHType, c_void_p, c_size_t, c_char_p ]
        self.lib.EMsoftCgetEBSDPatterns(self.genericIPar, self.genericFPar, self.genericEBSDPatterns, self.genericQuaternions, self.genericAccum_e, self.genericLPNH, self.genericLPSH, None, c_size_t(0), c_char_p("".encode('utf-8')) )
        
        if type(self.BackVal) != type(None):
            self.genericEBSDPatterns[:,:,0] -= self.BackVal