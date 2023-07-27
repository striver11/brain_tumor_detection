import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.python.keras import regularizers
from django.conf import settings
import os

def StartTraining():
    train_dir = os.path.join(settings.MEDIA_ROOT, 'brain_tumor_dataset','train')
    test_dir = os.path.join(settings.MEDIA_ROOT, 'brain_tumor_dataset','test')


    img_size=48
    train_datagen = ImageDataGenerator(width_shift_range = 0.1,height_shift_range = 0.1,horizontal_flip = True,rescale = 1./255,validation_split = 0.2)

    train_generator = train_datagen.flow_from_directory(directory = train_dir,target_size = (img_size,img_size),batch_size = 64,color_mode = "grayscale",class_mode = "categorical",subset = "training")

    #preparing validation dataset
    validation_datagen = ImageDataGenerator(rescale = 1./255,validation_split = 0.2)

    validation_generator = validation_datagen.flow_from_directory( directory = test_dir,target_size = (img_size,img_size),batch_size = 64,color_mode = "grayscale",class_mode = "categorical",subset = "validation")

    from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax
    model= tf.keras.models.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48,1)))
    model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,(5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
        
    model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten()) 
    model.add(Dense(256,activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
        
    model.add(Dense(512,activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer = Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    epochs= 65
    batch_szie=60
    history = model.fit(x = train_generator,epochs = epochs,validation_data = validation_generator)
    acc = history.history['accuracy'][-1]
    loss = history.history['loss'][-1]
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train','Val'])
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train','Val'])
    plt.show()

    print('history:',acc)
    print('loss:',loss)
    return acc,loss
