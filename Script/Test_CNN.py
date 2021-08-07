# Prediction
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from skimage import io
import os
from natsort import natsorted, ns
from keras.preprocessing import image

# names = ['Pastusa', 'R12']
names = ['americano', 'basket']
modelo = "/Users/Johnj/Documents/GitHub/Tesis/Tesis/Modelo/CNN_modelo.h5"
pesos_modelo = '/Users/Johnj/Documents/GitHub/Tesis/Tesis/Modelo/CNN_pesos.h5'

cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)


def prediction(file,filename):
    x = load_img(file,target_size=(150, 150))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    predictions = cnn.predict(x)
    result = predictions[0]
    answer = np.argmax(result)
    if answer == 0:
      print(filename + '     ' + "pred: americano / pastusa")
    elif answer == 1:
      print(filename + '     ' + "pred: basket / R12")   
    return answer

path = '/Users/Johnj/Documents/GitHub/Tesis/Tesis/Predicciones/papas' 
list_files = os.listdir(path)
list_files = natsorted(list_files)
image_list = []
for filename in list_files:
    papa = os.path.join(path,filename)
    prediction(papa,filename)
