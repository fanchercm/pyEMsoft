#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:27:31 2019

@author: hcf
"""


import numpy as np

from matplotlib.backends.qt_compat import QtWidgets

from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

from matplotlib.figure import Figure


class Qt5MplCanvas(FigureCanvas):
    """  A customized Qt widget for matplotlib figure.
    It can be used to replace GraphicsView of QtGui
    """
    def __init__(self, parent, width=4.5, height=4):
        """  Initialization
        """
        # from mpl_toolkits.axes_grid1 import host_subplot
        # import mpl_toolkits.axisartist as AA
        # import matplotlib.pyplot as plt

        # Instantiating matplotlib Figure
        
        self.fig = Figure(figsize=(width, height))
        self.fig.patch.set_facecolor('white')
#        self.fig.patch.set_facecolor('gray')

        self.axes = self.fig.add_subplot(111)  # return: matplotlib.axes.AxesSubplot
        self.fig.subplots_adjust(left=0.0)
        self.fig.subplots_adjust(right=1.0)
        self.fig.subplots_adjust(top=1.0)
        self.fig.subplots_adjust(bottom=0.0)
        self.axes.axis('off')

        
        # Initialize parent class and set parent
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        # Set size policy to be able to expanding and resizable with frame
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # Variables to manage all lines/subplot
        self._lineDict = {}
        self._lineIndex = 0

        # legend and color bar
        self._colorBar = None
        self._isLegendOn = False
        self._legendFontSize = 8


    def _update_canvas(self, img, ROIS=[]):
        """ Add an image by file
        """
#        import scipy.ndimage as ndimage

        #import matplotlib.image as mpimg

        self.clear_canvas()
        # set aspect to auto mode
        self.axes.set_aspect('auto')

        im = img[:,:,0]
#        im = ndimage.gaussian_filter(im, sigma=(8, 8), order=0)

        # lum_img = img[:,:,0]
        # FUTURE : refactor for image size, interpolation and origin
        self.axes.imshow(  np.fliplr( im[:,:].T ) ) 
        
        for rois in ROIS:            
            x1,y1, x2,y2 = rois
            self.axes.plot([x1,x2],[y1,y1], 'k')
            self.axes.plot([x1,x1],[y1,y2], 'k')
            self.axes.plot([x1,x2],[y2,y2], 'k')
            self.axes.plot([x2,x2],[y1,y2], 'k')

            
        self.axes.axis('off')
        self.draw()
    
    def addImage(self, img):
        """ Add an image by file
        """
        #import matplotlib.image as mpimg
        self.clear_canvas()

        # set aspect to auto mode
        self.axes.set_aspect('auto')

        #img = matplotlib.image.imread( imagefilename )
        self.axes.imshow(  img ) 
        
        # lum_img = img[:,:,0]
        # FUTURE : refactor for image size, interpolation and origin
        self.axes.axis('off')
        self.draw()
        
    def difference(self, SimPat, img):
        """ Add an image by file
        """
        #import matplotlib.image as mpimg
        self.clear_canvas()

        # set aspect to auto mode
        self.axes.set_aspect('auto')

#        img = matplotlib.image.imread( imagefilename )
        self.axes.imshow(  np.fliplr( SimPat[:,:,0].T ) - img ) 
        
        # lum_img = img[:,:,0]
        # FUTURE : refactor for image size, interpolation and origin
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.draw()        

            
    def clear_canvas(self):
        """ Clear data including lines and image from canvas
        """
        
        # clear image
        self.axes.cla()

# END-OF-CLASS (MplGraphicsView) 