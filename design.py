from asyncio.windows_events import NULL
from inspect import Attribute
import os, sys
from tkinter import dialog
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QAction,
    QSlider, QToolButton, QToolBar, QDockWidget, QMessageBox, QFileDialog, QGridLayout, 
    QScrollArea, QSizePolicy, QRubberBand , QInputDialog , QLineEdit)
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtGui import QIcon, QPixmap, QImage, QTransform, QPalette, qRgb, QColor
import math
import cv2
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from skimage.color import *
from PIL import Image
from Models import *
icon_path = "icons"
class imageLabel(QLabel):
    """Subclass of QLabel for displaying image"""
    def __init__(self, parent, image=None):
        super().__init__(parent)
        self.parent = parent 
        self.image = QImage()
        
        self.original_image = self.image
        

        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)

        # setBackgroundRole() will create a bg for the image
        #self.setBackgroundRole(QPalette.Base)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setScaledContents(True)

        # Load image
        self.setPixmap(QPixmap().fromImage(self.image))
        
        # self.setStyleSheet("background-color:black;")
        
        self.setAlignment(Qt.AlignCenter)
        self.o = ImageClasse()

    def openImage(self):
        """Load a new image into the """
        image_file, _ = QFileDialog.getOpenFileName(self, "Open Image", 
                "", "PNG Files (*.png);;JPG Files (*.jpeg *.jpg );;Bitmap Files (*.bmp);;\
                GIF Files (*.gif)")
        print(image_file)
        
        if image_file:
            # Reset values when opening an image
            self.parent.zoom_factor = 1
            self.path = image_file
            #self.parent.scroll_area.setVisible(True)
            self.parent.print_act.setEnabled(True)
            self.parent.updateActions()

            # Reset all sliders

            # Get image format
            image_format = self.image.format()
            self.image = QImage(image_file)
            
            self.original_image = self.image.copy()

            #pixmap = QPixmap(image_file)
            self.setPixmap(QPixmap().fromImage(self.image))
            #image_size = self.image_label.sizeHint()
            self.resize(self.pixmap().size())

            self.o.setImage(cv2.imread(self.path))
            #self.scroll_area.setMinimumSize(image_size)

            #self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), 
            #    Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif image_file == "":
            # User selected Cancel
            pass
        else:
            QMessageBox.information(self, "Error", 
                "Unable to open image.", QMessageBox.Ok)

    def Knn(self):
        res=self.o.Knn()

        # print("hello")
        dialog = QMessageBox(self)
        dialog.setWindowTitle("resultat de prediction par KNN")
        text = "la signature est predite comme class  : "
        text = text + res[0]
        dialog.setText(text)          
        dialog.show()

    def SVM(self):
        print("hello")
        dialog = QMessageBox(self)
        dialog.setWindowTitle("resultat de prediction par SVM")
        dialog.setText("la marque predite pour la voiture donnée en entrée est : AUDI")          
        dialog.show()

    def DT(self):
        print("hello")
        dialog = QMessageBox(self)
        dialog.setWindowTitle("resultat de prediction par Arbre de decision ")
        dialog.setText("la marque predite pour la voiture donnée en entrée est : AUDI")          
        dialog.show()
    

   
class PhotoEditorGUI(QMainWindow):
    
    def __init__(self):
        super().__init__()

        self.initializeUI()

        self.image = QImage()

        
       
       

    def initializeUI(self):
        self.setMinimumSize(300, 200)
        self.setWindowTitle("Machine Learning classement")
        self.showMaximized()

        
    

        self.zoom_factor = 1

        self.createMainLabel()
        # self.createEditingBar()
        self.createMenu()
        # self.createToolBar()

        self.show()

    def createMainLabel(self):
        """Create an instance of the imageLabel class and set it 
           as the main window's central widget."""
        self.image_label = imageLabel(self)
        self.image_label.resize(self.image_label.pixmap().size())

        self.scroll_area = QScrollArea()
        self.scroll_area.setBackgroundRole(QPalette.Dark)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        #self.scroll_area.setWidgetResizable(False)
        #scroll_area.setMinimumSize(800, 800)
        
        self.scroll_area.setWidget(self.image_label)
        #self.scroll_area.setVisible(False)

        self.setCentralWidget(self.scroll_area)

        #self.resize(QApplication.primaryScreen().availableSize() * 3 / 5)
     
  
    

    def createMenu(self):

        about_act = QAction('About', self)
        about_act.triggered.connect(self.aboutDialog)

        self.exit_act = QAction(QIcon(os.path.join(icon_path, "exit.png")), 'Quit Photo Editor', self)
        self.exit_act.setShortcut('Ctrl+Q')
        self.exit_act.triggered.connect(self.close)

        # Actions for File menu
        self.new_act = QAction(QIcon(os.path.join(icon_path, "new.png")), 'New...')

        self.open_act = QAction(QIcon(os.path.join(icon_path, "open.png")),'Open...', self)
        self.open_act.setShortcut('Ctrl+O')
        self.open_act.triggered.connect(self.image_label.openImage)

        self.print_act = QAction(QIcon(os.path.join(icon_path, "print.png")), "Print...", self)
        self.print_act.setShortcut('Ctrl+P')
        #self.print_act.triggered.connect(self.printImage)
        self.print_act.setEnabled(False)


        self.zoom_in_act = QAction(QIcon(os.path.join(icon_path, "zoom_in.png")), 'Zoom In', self)
        self.zoom_in_act.setShortcut('Ctrl++')
        self.zoom_in_act.triggered.connect(lambda: self.zoomOnImage(1.25))
        self.zoom_in_act.setEnabled(False)

        self.zoom_out_act = QAction(QIcon(os.path.join(icon_path, "zoom_out.png")), 'Zoom Out', self)
        self.zoom_out_act.setShortcut('Ctrl+-')
        self.zoom_out_act.triggered.connect(lambda: self.zoomOnImage(0.8))
        self.zoom_out_act.setEnabled(False)

        self.KNN = QAction(QIcon(os.path.join(icon_path, "zoom_out.png")), 'Methode KNN', self)
        self.KNN.triggered.connect( self.image_label.Knn)

        self.SVM = QAction(QIcon(os.path.join(icon_path, "zoom_out.png")), 'Methode SVM', self)
        self.SVM.triggered.connect( self.image_label.SVM)

        self.DT = QAction(QIcon(os.path.join(icon_path, "zoom_out.png")), 'Decision trees', self)
        self.DT.triggered.connect( self.image_label.DT)


        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)

        main_menu = menu_bar.addMenu('infos')
        main_menu.addAction(about_act)
        main_menu.addSeparator()
        main_menu.addAction(self.exit_act)

        

        # Create file menu and add actions
        file_menu = menu_bar.addMenu('File')
        file_menu.addAction(self.open_act)
        file_menu.addSeparator()

        edit_menu = menu_bar.addMenu('Edit')
        edit_menu.addAction(self.zoom_in_act)
        edit_menu.addAction(self.zoom_out_act)

        tool_menu = menu_bar.addMenu('tools')
        tool_menu.addAction(self.KNN)
        tool_menu.addAction(self.SVM)
        tool_menu.addAction(self.DT)




    
    def aboutDialog(self):
        QMessageBox.about(self, "About this application", 
            "Classement application \nVersion 0.1\n\nCreated by Achoauch Lamyae")
    
    def updateActions(self):
        """Update the values of menu and toolbar items when an image 
        is loaded."""
        self.zoom_in_act.setEnabled(True)
        self.zoom_out_act.setEnabled(True)
       
    def zoomOnImage(self, zoom_value):
        """Zoom in and zoom out."""
        self.zoom_factor *= zoom_value
        self.image_label.resize(self.zoom_factor * self.image_label.pixmap().size())

        self.adjustScrollBar(self.scroll_area.horizontalScrollBar(), zoom_value)
        self.adjustScrollBar(self.scroll_area.verticalScrollBar(), zoom_value)

        self.zoom_in_act.setEnabled(self.zoom_factor < 4.0)
        self.zoom_out_act.setEnabled(self.zoom_factor > 0.333)

    def normalSize(self):
        """View image with its normal dimensions."""
        self.image_label.adjustSize()
        self.zoom_factor = 1.0

    def adjustScrollBar(self, scroll_bar, value):
        """Adjust the scrollbar when zooming in or out."""
        scroll_bar.setValue(int(value * scroll_bar.value()) + ((value - 1) * scroll_bar.pageStep()/2))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_DontShowIconsInMenus, True)
    # app.setStyleSheet(style_sheet)
    window = PhotoEditorGUI()
    sys.exit(app.exec_())
