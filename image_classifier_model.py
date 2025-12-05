import os
import shutil
import numpy as np
import cv2
import pywt
import pandas as pd
import seaborn as sn
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



cropped_path = r"C:\Users\mshai\Documents\face_image_ML_project\images_dataset\cropped"
celebrity_filename_dict = {}


def w2d(img, mode='haar', level=1):
    
    imArray = img
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    imArray =  np.float32(imArray)  
    imArray /= 255;
    coeffs=pywt.wavedec2(imArray, mode, level=level)
    
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  
    
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)
    
    return imArray_H



for root, dirs, files in  os.walk(cropped_path):
    celebrity_name = root.split("\\")[-1]
    if celebrity_name != "cropped":
        celebrity_filename_dict[celebrity_name] = []
        for file in files:
            celebrity_filename_dict[celebrity_name].append(os.path.join(root, file))
            
 
class_dict = {}
count = 0
for celebrity_name in celebrity_filename_dict:
    class_dict[celebrity_name] = count
    count +=1
    
    
X, y = [], []
for celebrity_name, training_files in celebrity_filename_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img,'db1',5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        X.append(combined_img)
        y.append(class_dict[celebrity_name])        

X = np.array(X).reshape(len(X), 4096).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=10))])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))

report = classification_report(y_test, pipe.predict(X_test))
print(report)


model_params = {
    'svm':{
        'model': SVC(gamma='auto', probability=True),
        'params':{
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf', 'linear']
            }
        },
    
    'random_forest':{
        'model': RandomForestClassifier(),
        'params':{
            'randomforestclassifier__n_estimators': [1,5,10]
            }
        },
    'logistic_regression':{
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params':{
            'logisticregression__C': [1,5, 10]
            }
        }
    }

scores = []
best_estimators = {}

for model_name, model_param in model_params.items():
    pipe = make_pipeline(StandardScaler(), model_param['model'])
    clf = GridSearchCV(pipe, model_param['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_param': clf.best_params_
        })
    best_estimators[model_name] = clf.best_estimator_


df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_param'])

print(df['best_score'])

print(best_estimators['svm'].score(X_test, y_test))
print(best_estimators['random_forest'].score(X_test, y_test))
print(best_estimators['logistic_regression'].score(X_test, y_test))

best_clf = best_estimators['logistic_regression']

cm = confusion_matrix(y_test, best_clf.predict(X_test))
plt.figure(figsize=(10,10))
sn.heatmap(cm, annot=True)
plt.xlabel('y_predicted')
plt.ylabel('y_Truth')
plt.show()

with open("face_recognition_model.pickle", "wb") as f:
    pickle.dump(best_clf, f)
    
with open ("class_dict.json", "w") as file:
    file.write(json.dumps(class_dict))




