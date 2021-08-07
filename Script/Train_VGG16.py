from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from keras import backend as K


K.clear_session()

#COMPROBACION DATA AUGMENTATION
# path = 'Imagenes/Predict/1.jpeg' 
# img = image.load_img(path)
# data = img_to_array(img)
# samples = expand_dims(data,0)
# datagen = ImageDataGenerator(rotation_range=45)
# it = datagen.flow(samples, batch_size = 1)
# for i in range(6):
#     plt.subplot(230+1+i)
#     batch = it.next()
#     image = batch[0].astype('uint8')
#     plt.imshow(image)
# plt.show()



#DATA AUGMENTATION
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest',
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   validation_split=0.2)

validation_datagen = ImageDataGenerator(rescale = 1.0/255.)
test_datagen = ImageDataGenerator(rescale = 10./255.)

path1 = "/Users/Johnj/Documents/GitHub/Tesis/Tesis/sportimages/Train"
path2 = "/Users/Johnj/Documents/GitHub/Tesis/Tesis/sportimages/Test"
target_size = (150,150)

training_set = train_datagen.flow_from_directory(path1,
                                                 target_size = target_size,
                                                 batch_size = 20,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(path2,
                                            target_size = target_size,
                                            batch_size = 20,
                                            class_mode = 'binary')

##ARQUITECTURA VGG16 con FINE TUNNING
from tensorflow.keras.applications import VGG16


model = VGG16(input_shape = (150,150,3), include_top = False, weights = None)

model.trainable = True
set_trainable = False

for layer in model.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
    
    
modelFE = tf.keras.models.Sequential([
    model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

modelFE.summary()

modelFE.compile(optimizer = 'RMSprop' , loss = 'binary_crossentropy', metrics = ['accuracy'])

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
steps_per_epoch = training_set.n//batch_size
validation_steps = test_set.n//batch_size


history = modelFE.fit(training_set,
                           steps_per_epoch = 20,
                           epochs = 100,
                           validation_data = test_set,
                           validation_steps = 20,
                           verbose = 2,
                           callbacks = [checkpoint,early])

##ACCURACY y clases
clases = training_set.class_indices
print(clases)
test_lost, test_acc = modelFE.evaluate_generator(test_set)
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
modelFE.save('/Users/Johnj/Documents/GitHub/Tesis/Tesis/Modelo/VGG16_modelo.h5')
modelFE.save_weights('/Users/Johnj/Documents/GitHub/Tesis/Tesis/Modelo/VGG16_pesos.h5')
