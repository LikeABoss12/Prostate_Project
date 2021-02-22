#!/usr/bin/env python
# coding: utf-8

# In[1]:


# START OF PROJECT
from PIL import Image


# In[2]:


image=Image.open('/Users/sanath/Python_AI/Prostate_Project/AI_Research_Project/Grading_Data/Gleason_masks_train/mask_ZT111_4_A_1_12.png')


# In[3]:


image = image.convert("RGB")
image


# In[4]:


image.getpixel((100,100))


# In[5]:


# This function grades the image, assigning it a Gleason Score
def image_grading(image):
    red = 0
    green = 0
    blue = 0
    yellow = 0
    
    for y in range(0,image.size[0]):
        for x in range(0,image.size[1]):
            value = image.getpixel((y,x))
            if (value[0] == 0 and value[1] == 0 and value[2] == 255):
                blue+=1
            elif (value[0] == 255 and value[1] == 0 and value[2] == 0):
                red+=1
            elif (value[0] == 0 and value[1] == 255 and value[2] == 0):
                green+=1
            elif (value[0] == 255 and value[1] == 255 and value[2] == 0):
                yellow+=1
            elif (value[0] == 255 and value[1] == 255 and value[2] == 255):
                continue
    label_dict={'red':red,'green':green,'blue':blue,'yellow':yellow}
    vals=list(label_dict.values())
    keys=list(label_dict.keys())
    first_max=keys[vals.index(max(vals))]
    
    keys.remove(first_max)
    vals.remove(label_dict[first_max])
    second_max=keys[vals.index(max(vals))]
    gleason_lookup = {'red':5,'yellow':4,'blue':3,'green':0}
    
    # This code is for cells that have only one color
    if label_dict[second_max] > 0:        
        num1 = gleason_lookup[first_max]
        num2 = gleason_lookup[second_max]
        print(str(first_max.upper())+'and'+str(second_max.upper()))
        print(f"GLEASON SCORE: {num1+num2}")
        
    
    else:
        num1 = gleason_lookup[first_max]
        num2=0
        print(first_max.upper())
        print(f'GLEASON SCORE: {num1}')
        
    return num1+num2


# In[6]:


image_grading(image)
# TESTING


# In[7]:


import glob
import json
from PIL import Image
def image_finder(filepath): 
    image_status={}
    X=[]
    y=[]
    
    # STARTING ACTUAL WORK HERE
    files = glob.glob(filepath + '*.png')
    for file in files:
        print(f"Reading image {file}")
        image = Image.open(file)
        image = image.convert("RGB")
        score=image_grading(image)
        
        actualimage=file.split('/')[-1][5:-4]
        print("The filename is {}".format(actualimage))
        
        newimage=Image.open('/Users/sanath/Python_AI/Prostate_Project/AI_Research_Project/Grading_Data/Gleason_masks/'+actualimage+'.jpg')
        newimage=newimage.convert("RGB")

        if score <= 6:
            X.append(newimage)
            y.append(int(0))
            image_status[actualimage]=score
            
        elif score == 7:
            X.append(newimage)
            y.append(int(1))
            image_status[actualimage]=score
        elif score > 7:
            X.append(newimage)
            y.append(int(2))
            image_status[actualimage]=score
    
    with open("File_Grading.json",'w')as f:
        # UPLOADING IMAGES TO JSON FILE FOR FUTURE REFERENCE
        json.dump(image_status,f)
    return image_status


# In[44]:


# TESTING
image_finder('/Users/sanath/Python_AI/Prostate_Project/AI_Research_Project/Grading_Data/Gleason_masks_train')


# In[23]:


# TESTING
image_finder('/Users/sanath/Python_AI/Prostate_Project/AI_Research_Project/Grading_Data/Gleason_masks_train/')


# In[8]:


lowriskimages_y=[]
highriskimages_y=[]
mediumriskimages_y=[]
X = []
y = []
# the basic idea is to create 6 lists, 3 for X and 3 for y 
# X will have the image name itself, and y will have the label 0,1 or 2
# these for loops will make the X and y lists
for i in range(len(lowriskimage)):
    X.append(lowriskimages[i])
    y.append(int(0))
for i in range(len(mediumriskimages)):
    X.append(mediumriskimages[i])
    y.append(int(1)) 
for i in range(len(highriskimages)):
    X.append(highriskimages[i])
    y.append(int(2))


# In[2]:


filepath=('/Users/sanath/Python_AI/Prostate_Project/AI_Research_Project/Grading_Data/Gleason_masks/')


# In[3]:


import glob
for files in glob.glob(filepath+'*.png'):
    print(files)


# In[1]:


import json
with open('AI_Research_Project/Grading_Data/File_Grading.json')as f:
    file=json.load(f)


# In[5]:


len([*file.keys()])


# In[6]:


file['ZT111_4_A_8_13']


# In[7]:


keys=[*file.keys()]
values=[*file.values()]
# targetkey=values.index(keys[1])
# print(targetkey)
file[keys[1]]


# In[8]:


import cv2
import numpy as np
X = []
y = []
for i in range(0,len([*file.keys()])):
    filename=[*file.keys()][i]
    value=file[filename]
    image=cv2.imread('/Users/sanath/Python_AI/Prostate_Project/AI_Research_Project/Grading_Data/Gleason_masks/'+filename+'.jpg')
    image=cv2.resize(image,(128,128)).flatten() # This is the image resolution
    image=image.astype(np.float32)
    image=image/255.0
    X.append(image)
    score = file[filename]
    if score <= 6:
        y.append(int(0))
    if score == 7:
        y.append(int(1)) 
    if score > 7:
        y.append(int(2))
    print("File is flattened")


# In[9]:


X=np.array(X)
y=np.array(y)


# In[10]:


type(y)


# In[17]:


from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)


# In[83]:


y


# In[15]:


from sklearn.model_selection import train_test_split


# In[12]:


np.unique(y)


# In[13]:


X.shape


# In[85]:


y


# In[86]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,shuffle=True)


# In[19]:


X_train.shape


# In[20]:


print(X_train.shape)


# In[21]:


X.shape


# In[22]:


y_train


# In[23]:


import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(4096,input_shape=X_train.shape,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2048,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(3,activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()


# In[19]:


keras.callbacks.EarlyStopping(min_delta=0,patience=0,verbose=0,mode='auto')
history=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=40,batch_size=32)


# In[20]:


import matplotlib.pyplot as plt
fig=plt.figure()
plt.plot(history.history['val_loss'])
plt.plot(history.history['loss'])

plt.legend(history.history)
plt.show()
plt.savefig('loss_20epochs.jpg')


# In[21]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()
plt.savefig('accuracy_20epochs.jpg')


# In[22]:


model.save('accuracy_20epochs.h5')


# In[59]:


X.shape


# In[23]:


model.predict(X_test)


# In[64]:


y_test


# In[65]:


X_train.shape


# In[78]:


y_train.shape


# In[77]:


y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)


# In[84]:


y_train.shape


# In[35]:


1536+387


# In[24]:


import tensorflow as tf


# In[68]:


print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
print(y_train[0])


# In[25]:


cnnmodel=tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3),input_shape=(128,128,3),activation='relu'),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     tf.keras.layers.Dropout(0.15),
                                     #Second Convolution Layer
                                     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     tf.keras.layers.Dropout(0.1),
                                     #Third Convolution Layer
                                     tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     tf.keras.layers.Dropout(0.1),
                                     #Flattening Layer
                                     tf.keras.layers.Flatten(),
                                     # Dense 1
                                     tf.keras.layers.Dense(512,activation='relu'),
                                     tf.keras.layers.Dense(256,activation='relu'),   
                                     #Dense2/Output Layer
                                     tf.keras.layers.Dense(128,activation='relu'),
                                     tf.keras.layers.Dense(3,activation='sigmoid') 
                                     ])


# In[59]:


from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
y_train=lb.fit_transform(y_train)
y_test=lb.fit_transform(y_test)


# In[46]:


from keras.optimizers import SGD
opt=SGD(lr=0.001)
cnnmodel.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['acc'])


# In[47]:


cnnmodel.summary()


# In[33]:


early_stopping=keras.callbacks.EarlyStopping(min_delta=0,patience=3,verbose=0,mode='auto')
cnnhistory=cnnmodel.fit(X_train.reshape(-1,128,128,3),y_train,validation_data=(X_test.reshape(-1,128,128,3),y_test),epochs=30,validation_split=0.2,shuffle=True,callbacks=[early_stopping],batch_size=48)


# In[97]:


import matplotlib.pyplot as plt
plt.plot(cnnhistory.history['acc'])
plt.plot(cnnhistory.history['val_acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy %')
plt.legend({'Accuracy':0,'ValidationAccuracy':1})
plt.title('Training Accuracy vs Validation Accuracy')
plt.savefig('cnn_accuracy_earlystopping.jpg')


# In[84]:


plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.savefig('cnn_loss.jpg')


# In[82]:


cnnmodel.save('cnnmodel.h5')


# In[48]:


import keras
import cv2
import numpy as np

class ProstateDiagnosis(object):
    def __init__(self):
        self.model=keras.models.load_model('cnnmodel.h5')
    
    def read_image(self,imagename):
        image=cv2.imread(imagename)
        image=cv2.resize(image,(128,128)).flatten() # This is the image resolution
        image=image.astype(np.float32)
        image=image/255.0
        self.image=image.reshape(-1,128,128,3)
    
    def predict(self):
        prediction=self.model.predict(self.image)
        target_class=(np.argmax(prediction))
        if target_class==0:
            print("The specimen is at Benign Risk")
            self.risk=target_class
        if target_class==1:
            print("The specimen is at Moderate Risk")
            self.risk=target_class
        if target_class==2:
            print("The specimen is at High Risk")
            self.risk=target_class


# In[49]:


pd=ProstateDiagnosis()


# In[50]:


import os
os.getcwd()


# In[51]:


pd.read_image('AI_Research_Project/Grading_Data/ZT199_1_A/ZT199_1_A_2_2.jpg')


# In[52]:


pd.predict()


# In[57]:


pd.image.shape


# In[29]:


pip freeze


# In[69]:


print(X_train.shape,y_train.shape)


# In[70]:


print(X_test.shape, y_test.shape)


# In[62]:


641-(641*.2)


# # One Hot Encoding

# In[79]:


from keras.utils.np_utils import to_categorical


# In[80]:


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


# In[81]:


y_test


# In[90]:


cnnmodel=tf.keras.models.Sequential([tf.keras.layers.Conv2D(64,(3,3),input_shape=(128,128,3),padding='valid',strides=(2,2),activation='relu'),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     tf.keras.layers.Dropout(0.15),
                                     #Second Convolution Layer
                                     tf.keras.layers.Conv2D(128,(3,3),strides=(1,1),activation='relu',activity_regularizer=tf.keras.regularizers.l2(0.01)),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     tf.keras.layers.Dropout(0.1),
                                     #Third Convolution Layer
                                     tf.keras.layers.Conv2D(256,(3,3),strides=(1,1),activation='relu',activity_regularizer=tf.keras.regularizers.l2(0.01)),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     tf.keras.layers.Dropout(0.1),
#                                      Fourth Convolution Layer
                                     tf.keras.layers.Conv2D(512,(3,3),strides=(1,1),activation='relu',activity_regularizer=tf.keras.regularizers.l2(0.01)),
                                     tf.keras.layers.MaxPool2D(2,2),
                                     tf.keras.layers.Dropout(0.1),
                                     #Flattening Layer
                                     tf.keras.layers.Flatten(),
                                     # Dense 1
                                     tf.keras.layers.Dense(512,activation='relu',activity_regularizer=tf.keras.regularizers.l2(0.01)),
                                     tf.keras.layers.Dropout(0.5),
                                     tf.keras.layers.Dense(256,activation='sigmoid',activity_regularizer=tf.keras.regularizers.l1_l2(0.01)),   
                                     #Dense2/Output Layer
                                     tf.keras.layers.Dense(128,activation='sigmoid'),
                                     tf.keras.layers.Dense(32,activation='sigmoid'),
                                     tf.keras.layers.Dense(3,activation='sigmoid',activity_regularizer=tf.keras.regularizers.l1_l2(0.01)) 
                                     ])


# In[91]:


from keras.optimizers import SGD
opt=SGD(lr=0.003)
cnnmodel.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])


# In[87]:


from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
y_train=lb.fit_transform(y_train)
y_test=lb.fit_transform(y_test)


# In[89]:


y_train


# In[92]:


cnnmodel.summary()


# In[93]:


early_stopping=keras.callbacks.EarlyStopping(min_delta=0,patience=5,verbose=0,mode='auto')
cnnhistory=cnnmodel.fit(X_train.reshape(-1,128,128,3),y_train,validation_data=(X_test.reshape(-1,128,128,3),y_test),epochs=30,shuffle=True,callbacks=[early_stopping],batch_size=32)


# In[94]:


cnnmodel.save('cnnmodel_earlycallback')


# In[97]:


import matplotlib.pyplot as plt
plt.plot(cnnhistory.history['acc'])
plt.plot(cnnhistory.history['val_acc'])
plt.savefig('accuracy_rmsprop.jpg')


# In[98]:


plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.show()
plt.savefig('loss_rmsprop.jpg')


# In[102]:


# Saving the model here
tf.keras.models.save_model(cnnmodel,'updatedmodel')

