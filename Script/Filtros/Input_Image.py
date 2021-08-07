import os
from skimage import io
import matplotlib.pyplot as plt

     
def Input(numeroimagen):
    path = "/Users/Johnj/Documents/GitHub/Tesis/Tesis/Predicciones" 
    papa = os.path.join(path,"deportes",numeroimagen)
    imagen = io.imread(papa)
    ##VISUALIZACION
    # plt.imshow(imagen)
    # plt.title("Original")
    # plt.show()
    return imagen

