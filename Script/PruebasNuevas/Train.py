import sys
import os
from tensorflow import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import train_test_split

K.clear_session()
data_entrenamiento = "/Users/Johnj/Documents/GitHub/Anteproyecto-de-Grado-Clasificador-de-Papas-Da-adas-Vision-Artificial/ProcesamientoImagenes_Python_PruebasIniciales/dataset/TRAIN"
data_validacion = '/Users/Johnj/Documents/GitHub/Anteproyecto-de-Grado-Clasificador-de-Papas-Da-adas-Vision-Artificial/ProcesamientoImagenes_Python_PruebasIniciales/dataset/TEST'

# data_entrenamiento = '/Users/Johnj/Documents/Python Scripts/EjemlpoDeportes/sportimages - copia/Train' 
# data_validacion = '/Users/Johnj/Documents/Python Scripts/EjemlpoDeportes/sportimages - copia/Test'


batch_size = 64

filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
lr = 0.0005
target_size = (150,150)

##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest',
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2)


test_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest',
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2)

#CREACION DEL DATA SET

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=target_size,
    batch_size=32,
    class_mode='categorical',
    save_to_dir='/Users/Johnj/Documents/Python Scripts/EjemlpoDeportes/Augmented',
    save_prefix="img",
    save_format="jpg")

for i in range(len(entrenamiento_generador)):
    entrenamiento_generador.next()

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical')

#RED CNN

cnn = Sequential()
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(150, 150,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same", activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same", activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.1))
cnn.add(Dense(2, activation='softmax'))


cnn.compile(loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(lr=lr),
            metrics=['accuracy'])

##ENTRENAR RED VGG16
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'Modelo/pesos.hdf5',
    monitor="val_loss",
    verbose=1,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    options=None,
)

early = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=10,
    verbose=1,
    mode="auto",
)

batch_size = 100
# steps_per_epoch = entrenamiento_generador.n//batch_size
# validation_steps = validacion_generador.n//batch_size


history = cnn.fit(entrenamiento_generador,
                           steps_per_epoch = 10,
                           epochs = 20,
                           validation_data = validacion_generador,
                           validation_steps = 10,
                           verbose = 2,
                           callbacks = [checkpoint,early])

#Gráficas
acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(1,len(acc)+1,1)

plt.plot ( epochs,     acc, 'r--', label='Training acc'  )
plt.plot ( epochs, val_acc,  'b', label='Validation acc')
plt.title ('Training and validation accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')

plt.legend()
plt.figure()

plt.plot ( epochs,     loss, 'r--',label='Training loss')
plt.plot ( epochs, val_loss ,  'b',label='Validation loss'   )
plt.title ('Training and validation loss' )
plt.ylabel('acc')
plt.xlabel('epochs')

plt.legend()
plt.figure()


#Guardar Modelo
target_dir = '/Users/Johnj/Documents/GitHub/Anteproyecto-de-Grado-Clasificador-de-Papas-Da-adas-Vision-Artificial/ProcesamientoImagenes_Python_PruebasIniciales/PruebasNuevas/Modelo'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('/Users/Johnj/Documents/GitHub/Anteproyecto-de-Grado-Clasificador-de-Papas-Da-adas-Vision-Artificial/ProcesamientoImagenes_Python_PruebasIniciales/PruebasNuevas/Modelo/modelo.h5')
cnn.save_weights('/Users/Johnj/Documents/GitHub/Anteproyecto-de-Grado-Clasificador-de-Papas-Da-adas-Vision-Artificial/ProcesamientoImagenes_Python_PruebasIniciales/PruebasNuevas/Modelo/pesos.h5')


cnn.summary()
##ACCURACY y clases
clases = entrenamiento_generador.class_indices
print(clases)
test_lost, test_acc = cnn.evaluate_generator(validacion_generador)
print('Test ACcuracy:',test_acc)