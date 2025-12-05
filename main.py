import streamlit as st
import json 
import pickle
import os
import numpy as np
import cv2
import tempfile
import pywt
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
cropped_dataset_path = os.path.join(dir_path, "images_dataset/cropped")

    
def load_ML_model():
    with open(os.path.join(dir_path, 'face_recognition_model.pickle'), 'rb') as f:
        return pickle.load(f)
    
def load_class_dict():
    with open(os.path.join(dir_path, 'class_dict.json'), 'rb') as f:
        return json.load(f)
    
celebrity_class_dict = load_class_dict()
model = load_ML_model()

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

def get_cropped_face(image_path):
    face_cascade = cv2.CascadeClassifier(os.path.join(dir_path, "opencv/haarcascades/haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(os.path.join(dir_path, "opencv/haarcascades/haarcascade_eye.xml"))
    
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >=2:
            return roi_color

def classify_image(uploaded_image):  

    cropped_img = get_cropped_face(uploaded_image)
    img = cropped_img
    scalled_raw_img = cv2.resize(img, (32, 32))
    img_har = w2d(img,'db1',5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
    final_image = np.array(combined_img).reshape(1, (32*32*3 + 32*32)).astype(float)
    
    result = {'class': model.predict(final_image)[0],
              'class_probability': np.around(model.predict_proba(final_image)*100,2).tolist()[0]}
    
    return result
    
st.title("Sports Person Image Classifier")

image_list = {}
for root, dirs, files in  os.walk(cropped_dataset_path):
    for celeb_name in celebrity_class_dict:
        if celeb_name in root:
            celeb_image_file = os.listdir(root)[0]
            image_list[celeb_name] = os.path.join(root, celeb_image_file)
          
cols = st.columns(len(image_list))
count = 0
for celeb_name in image_list:
    with cols[count]:
        st.image(image_list[celeb_name], caption=celeb_name, width=100)
        count+=1
        
uploaded_image = st.file_uploader( "Drag and drop an Image to Identify:", type=['jpg', 'png'])

if st.button("Identify"):
    if uploaded_image:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_image.name)
        with open(path, "wb") as f:
                f.write(uploaded_image.getvalue())
                
        result = classify_image(path)
        if result:
            for key_celeb_name in celebrity_class_dict:
                if celebrity_class_dict[key_celeb_name]==result['class']:
                    st.subheader("Identified As:")
                    st.image(image_list[key_celeb_name], caption=key_celeb_name, width=100)
                    prob_data = pd.DataFrame({"Predicted Probability":result['class_probability']}, index=celebrity_class_dict)
                    st.table(prob_data)
                    st.success(key_celeb_name)
        


    

