#!/usr/bin/env python
# coding: utf-8

# In[14]:


# These libraries can help diagnose and grade the images
from keras.models import load_model
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class ProstateDiagnosis():
    def __init__(self):
        self.model = load_model('updatedmodel')

    def read_image(self, imagename): #This function is mainly for preprocessing purposes
        self.image = cv2.imread(imagename)
        self.image = cv2.resize(self.image, (128, 128)).flatten()
        self.image = self.image.astype(np.float32)
        self.image = self.image / 255.0  #Normalizing data 
        self.image = self.image.reshape(-1, 128, 128, 3)
        return self.image

    def predict(self):
        prediction = self.model.predict(self.image)
        target_class = (np.argmax(prediction))
        if target_class == 0:
            print("The specimen is at Benign Risk")
            self.riskclass="Benign"
            self.risk = target_class
        if target_class == 1:
            print("The specimen is at Moderate Risk")
            self.riskclass="Moderate"
            self.risk = target_class
        if target_class == 2:
            print("The specimen is at High Risk")
            self.riskclass="High"
            self.risk = target_class
        return self.risk


# In[15]:


pd = ProstateDiagnosis()
image=pd.read_image('AI_Research_Project/Grading_Data/ZT199_1_A/ZT199_1_A_1_11.jpg')

risk=pd.predict()

plt.imshow(image.reshape(128,128,3))
plt.title(f"The patient is at a {pd.riskclass} risk")
plt.show()


# In[ ]:




