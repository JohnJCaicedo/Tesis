# Prediction
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Input_Image import *

# names = ['Pastusa', 'R12']
names = ['americano', 'basket']
modelo = "/Users/Johnj/Documents/GitHub/Tesis/Tesis/Modelo/VGG16_modelo.h5"
pesos_modelo = '/Users/Johnj/Documents/GitHub/Tesis/Tesis/Modelo/VGG16_pesos.h5'

modelt = load_model(modelo)
modelt.load_weights(pesos_modelo)

width_shape = 150
height_shape = 150


imaget_path = '/Users/Johnj/Documents/GitHub/Tesis/Tesis/Predicciones/deportes/basquet4.jpg'
imaget=cv2.resize(cv2.imread(imaget_path), (width_shape, height_shape), interpolation = cv2.INTER_AREA)
xt = np.asarray(imaget)
xt=preprocess_input(xt)
xt = np.expand_dims(xt,axis=0)
preds = modelt.predict(xt)

print(names[np.argmax(preds)])
plt.imshow(cv2.cvtColor(np.asarray(imaget),cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()