import numpy as np
import os
from natsort import natsorted, ns
from keras.preprocessing import image
from keras.models import load_model
from keras import backend as K
import matplotlib.pyplot as plt
K.clear_session()

#CARGAR MODELO
modelo = '/Users/Johnj/Documents/GitHub/Anteproyecto-de-Grado-Clasificador-de-Papas-Da-adas-Vision-Artificial/ProcesamientoImagenes_Python_PruebasIniciales/PruebasNuevas/Modelo/modelo.h5'
pesos_modelo = '/Users/Johnj/Documents/GitHub/Anteproyecto-de-Grado-Clasificador-de-Papas-Da-adas-Vision-Artificial/ProcesamientoImagenes_Python_PruebasIniciales/PruebasNuevas/Modelo/pesos.h5'
classifier = load_model(modelo)
classifier.load_weights(pesos_modelo)
target_size = (150,150)

#CARGAR IMAGENES PARA TESTEAR
#path = "/Users/Johnj/Documents/GitHub/Anteproyecto-de-Grado-Clasificador-de-Papas-Da-adas-Vision-Artificial/ProcesamientoImagenes_Python_PruebasIniciales/Imagenes/BUENAS" 
#path = "/Users/Johnj/Documents/GitHub/Anteproyecto-de-Grado-Clasificador-de-Papas-Da-adas-Vision-Artificial/ProcesamientoImagenes_Python_PruebasIniciales/Imagenes/DEFECTUOSAS" 
#path = "/Users/Johnj/Documents/GitHub/Anteproyecto-de-Grado-Clasificador-de-Papas-Da-adas-Vision-Artificial/ProcesamientoImagenes_Python_PruebasIniciales/Imagenes/Predict2" 
#path = "/Users/Johnj/Documents/GitHub/Anteproyecto-de-Grado-Clasificador-de-Papas-Da-adas-Vision-Artificial/ProcesamientoImagenes_Python_PruebasIniciales/Imagenes/Buenasnews" 
#path = '/Users/Johnj/Documents/GitHub/Fotos/Imagenes/Predict2'
#path = '/Users/Johnj/Documents/GitHub/Fotos/Imagenes/R12'
path = '/Users/Johnj/Downloads/PruebaDeportes'

list_files = os.listdir(path)
list_files = natsorted(list_files)
image_list = []
c1 = 0 
c2 = 0
for filename in list_files:
    papa = os.path.join(path,filename)
    test_image = image.load_img(papa, target_size = target_size)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    #print(result)
    if result[0] == 0:
        prediction = 'Buena/pastusa/Americano'
        c1 = c1+1
    if result[0] == 1:
        prediction = 'Mala/R12/Basquet'
        c2 = c2+1
    print(f'{filename}',prediction + '   ' + f'Porcentaje de: {result}')
print(f'Papas buenas:{c1}')
print(f'Papas malas:{c2}')


