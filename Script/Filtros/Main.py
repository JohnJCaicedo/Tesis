from Input_Image import *
from EdgeFilter import *
from Filtro1_GrayScale import *
from Filtro2_RegionalMaxima import *
from Filtro3_Histogram import *
from skimage.transform import rescale, resize, downscale_local_mean
from Tiempo import *
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import skimage
import cv2
from skimage.feature import hog
from skimage import data, exposure
'''
Solo cambiar el número de la imagen
#Caracteristicas de la imagen
shape=imagen.shape     #Tamaño de la imagen
size=imagen.size       #Cantidad de pixeles
'''
# start_time = time()
papamala ="IMG_0785.JPG"
papabuena = "IMG_1223.JPG"
imagen=Input(papamala)

'''
FILTRO 1 RGB2GRAY 
-rgb2grascale(imagen)

FILTRO 2 REGIONAL MAXIMA
-RegionalMaxima(imagen,sgm,h)

FILTRO 3 HISTOGRAM EQUALIZATION
TIENE QUE SER UNA IMAGEN EN ESCALA DE GRISES
-Histogram

Canny Edge Filter
-im=imagen que se quiere analizar
-sgm1=Sigma deseado 1
-sgm2=Sigma deseado 2
'''
##############################################
grayscale = rgb2grascale(imagen)
image_rescaledgray = skimage.transform.rescale(grayscale, 0.1, anti_aliasing=False)
image_rescaledrgb = cv2.resize(imagen, (317,475), interpolation = cv2.INTER_AREA)
image = Histogram(image_rescaledgray)
#rint(grayscale.shape) 
#image_rescaled = resize(imagen, (100, 100,3))
#print(image_rescaled.shape)  

##############################################SEGMETNACION

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()































# plt.imshow(imagen)
# plt.title("Original")
# plt.show()

#TIEMPO DE EJECICIÖN
# elapsed_time = time() - start_time
# print("Elapsed time: %.10f seconds." % elapsed_time)