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

K.clear_session()
#Imagenes Deportes
# data_entrenamiento = "/Users/Johnj/Documents/GitHub/Tesis/Tesis/sportimages/Train"
# data_validacion =  "/Users/Johnj/Documents/GitHub/Tesis/Tesis/sportimages/Test"
#Imagenes papas
data_entrenamiento = "/Users/Johnj/Documents/GitHub/Tesis/Tesis/dataset/Train"
data_validacion =  "/Users/Johnj/Documents/GitHub/Tesis/Tesis/dataset/Test"



longitud, altura = 150, 150
clases = 2

##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    #color_mode="grayscale",
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=20,
    class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=20,
    class_mode='categorical')

#RED CNN

cnn = tf.keras.Sequential()
cnn.add(Convolution2D(32, (3,3), padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Convolution2D(64, (3,3), padding ="same", activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))

cnn.summary()

cnn.compile(loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
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

batch_size = 32
steps_per_epoch = entrenamiento_generador.n//batch_size
validation_steps = validacion_generador.n//batch_size


history = cnn.fit(entrenamiento_generador,
                           steps_per_epoch = steps_per_epoch,
                           epochs = 100,
                           validation_data = validacion_generador,
                           validation_steps = validation_steps,
                           verbose = 2,
                           callbacks = [checkpoint,early])


##ACCURACY y clases
clases = entrenamiento_generador.class_indices
print(clases)
test_lost, test_acc = cnn.evaluate_generator(validacion_generador)
print('Test ACcuracy:',test_acc)


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
import os
target_dir = '/Users/Johnj/Documents/GitHub/Tesis/Tesis/Modelo'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('/Users/Johnj/Documents/GitHub/Tesis/Tesis/Modelo/CNN_modelo.h5')
cnn.save_weights('/Users/Johnj/Documents/GitHub/Tesis/Tesis/Modelo/CNN_pesos.h5')
