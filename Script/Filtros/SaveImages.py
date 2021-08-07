import cv2
import os
import re
import matplotlib.pyplot as plt
from Filtro3_Histogram import *
from os import system
from EdgeFilter import *
from Filtro1_GrayScale import *



system("cls")
##CARGAR SET DE IMAGENES
dirname = os.path.join(os.getcwd(), 'Imagenes/DEFECTUOSAS')
imgpath = dirname + os.sep 

images = []
directories = []
dircount = []
prevRoot=''
cant=0
i=0
print("leyendo imagenes de ",imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff|JPG)$", filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            image = cv2.resize(image, (64, 64), interpolation = cv2.INTER_AREA)
            # image = rgb2grascale(image)
            # image = Histogram(image)
            #image = Edges(image,0.5,1)
            cv2.imwrite(f'/Users/Johnj/Documents/GitHub/Anteproyecto-de-Grado-Clasificador-de-Papas-Da-adas-Vision-Artificial/ProcesamientoImagenes_Python_PruebasIniciales/Rescale/DEFECTUOSAS/{i}.jpg',image)
            i=i+1
print(f'Numero de Imagenes: {i}')