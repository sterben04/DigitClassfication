import os
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

################
path = 'myData'
################
images = []
classNo = []
myList = os.listdir(path)
noOfClasses = len(myList)

for x in range(0, noOfClasses):
    myPicList = os.listdir(path + '/' + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + '/' + str(x) + '/' + y)
        curImg = cv2.resize(curImg, (32, 32))
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")


images = np.array(images)
classNo = np.array(classNo)

X_train_full, X_test, y_train_full, y_test = train_test_split(images, classNo)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full)

noOfSamples = []
for x in range(0, noOfClasses):
    noOfSamples.append(len(np.where(y_train == x)[0]))


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_val = np.array(list(map(preProcessing, X_val)))

print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

print(X_train.shape)


y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_val = to_categorical(y_val, noOfClasses)

print(y_train.shape)

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(32, 32, 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = myModel()
print(model.summary())

history = model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=5,steps_per_epoch=1000)
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')


plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score=', score[0])
print('Test Accuracy=', score[1])
#
# pickle_out = open("model_trained.p","wb")
# pickle.dump(model,pickle_out)
# pickle_out.close()