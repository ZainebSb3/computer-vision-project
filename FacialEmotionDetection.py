from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
import matplotlib as plt
import glob
import cv2
import time


import os
import seaborn as sns
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, Input,Average,concatenate
from keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
print(os.listdir("/content/drive/MyDrive/images"))
['test', 'train']
SIZE=224
train_images=[]
train_labels=[]
for directory_path in glob.glob("/content/drive/MyDrive/images/Train/*"):
  label=directory_path.split("\\")[-1]
  print(label)
  for img_path in glob.glob(os.path.join(directory_path,"*.tiff")):
      print(img_path)
      img=cv2.imread(img_path,cv2.IMREAD_COLOR)
      img=cv2.resize(img,(SIZE,SIZE))
      img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
      train_images.append(img)
      train_labels.append(label)
train_images=np.array(train_images)
train_labels=np.array(train_labels)
test_images=[]
test_labels=[]
for directory_path in glob.glob("/content/drive/MyDrive/images/Test/*"):
  label=directory_path.split("\\")[-1]
  print(label)
  for img_path in glob.glob(os.path.join(directory_path,"*.tiff")):
      print(img_path)
      img=cv2.imread(img_path,cv2.IMREAD_COLOR)
      img=cv2.resize(img,(SIZE,SIZE))
      img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
      test_images.append(img)
      test_labels.append(label)
test_images=np.array(test_images)
test_labels=np.array(test_labels)
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
le.fit(train_labels)
train_labels_encoded=le.transform(train_labels)
le.fit(test_labels)
test_labels_encoded=le.transform(test_labels)
x_train,y_train ,x_test,y_test= train_images, train_labels_encoded,test_images,test_labels_encoded
x_train = x_train/255.0
x_test = x_test/255.0
VGG_model=VGG16(weights='imagenet', include_top=False, input_shape=(SIZE,SIZE,3))
VGG_model.summary()
x=VGG_model.output
x=GlobalAveragePooling2D()(x)
model=Model(inputs=VGG_model.input,outputs=x)
for layer in model.layers:
  layer.trainable=False
model.summary()
feature_extractor=model.predict(x_train)
features = feature_extractor.reshape(feature_extractor.shape[0],-1)
x_for_training=feature_extractor
x_test_feature=model.predict(x_test)
x_test_feature=x_test_feature.reshape(x_test_feature.shape[0],-1)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
from sklearn.svm import SVC
from scipy.stats import loguniform
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import metrics
!pip install scikit-elm
clf1=MLPClassifier(hidden_layer_sizes=(86) , max_iter = 3000 , activation ='relu' , alpha = 0.0001 , solver = 'adam' , shuffle =True)
clf1.fit(x_for_training, y_train)
prediction=clf1.predict(x_test_feature)
prediction = le.inverse_transform(prediction)
print("accuracy = ", metrics.accuracy_score(test_labels,prediction))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split ,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
from sklearn.svm import SVC
from scipy.stats import loguniform
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import metrics
clf1 = MLPClassifier(hidden_layer_sizes=(60,), max_iter = 2000,activation = 'logistic',alpha=0.0001, solver = 'adam',shuffle=True)
!pip install scikit-elm

