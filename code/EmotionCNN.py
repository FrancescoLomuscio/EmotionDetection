# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:33:09 2020

@author: utenti
"""

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = '../res/data/data/train/'
test_path = '../res/data/data/test/'
xml_path = '../res/XML/'

emotions = []

num_train = 28709
num_test = 7178
batch_size = 64
num_epoch = 50

# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        test_path,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

model = Sequential()

model.add(Conv2D(32,kernel_size = (3,3),activation = 'relu',input_shape= (48,48,1)))
model.add(Conv2D(64,kernel_size = (3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

while(True):
    mode = input("Scegliere tra train e evaluate:\n")
    if mode == "train":
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
        model_info = model.fit(
                train_generator,
                steps_per_epoch=num_train // batch_size,
                epochs=num_epoch,
                validation_data=validation_generator,
                validation_steps=num_test // batch_size)
        model.save_weights('../res/model.h5')
        #plot_model_history(model_info)
    elif mode == "test":
        model.load_weights('../res/model.h5')
        
        emotions = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        capture = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(xml_path+'haarcascade_frontalface_default.xml')
        """ isOpened() controlla che la cattura sia sta inizializzata correttamente
        altrimenti si usa il metodo open() per forzare l'apertura """
        while(capture.isOpened()):
            """ ret e' True quando il frame viene letto correttamente """
            ret, frame = capture.read()
            if(ret):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                faces = face_cascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
                
                for (x,y,h,w) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    roi_gray = gray[y:y+h,x:x+w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    prediction = model.predict(cropped_img)
                    maxindex = int(np.argmax(prediction))
                
                    cv2.putText(frame,emotions[maxindex],(x+20,y-20),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('EmotionDetection',frame)
                k = cv2.waitKey(1)
                if k == 27:
                    break
                
        capture.release()
        cv2.destroyAllWindows()
    
