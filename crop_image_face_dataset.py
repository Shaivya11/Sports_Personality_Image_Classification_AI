import os
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt

cropped_path = r"C:\Users\mshai\Documents\face_image_ML_project\images_dataset\cropped"
celebrity_filename_dict = {}

def get_cropped_face(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >=2:
            return roi_color
    
    

img = cv2.imread("C:/Users/mshai/Documents/face_image_ML_project/images_dataset/maria_sharapova/sharapova-hits-the-practice-courts-and-met-ball-kids.jpg")
print(img.shape)
plt.imshow(img)

grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(grayimg.shape)
plt.imshow(grayimg, cmap='gray')

face_cascade = cv2.CascadeClassifier(r"C:\Users\mshai\Documents\face_image_ML_project\opencv\haarcascades\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"C:\Users\mshai\Documents\face_image_ML_project\opencv\haarcascades\haarcascade_eye.xml")

if os.path.exists(cropped_path):
    shutil.rmtree(cropped_path)
os.mkdir(cropped_path)

cv2.destroyAllWindows()
for root, dirs, files in  os.walk(r"C:\Users\mshai\Documents\face_image_ML_project\images_dataset"):
    celebrity_name = root.split("\\")[-1]
    if celebrity_name != "cropped":
        celebrity_filename_dict[celebrity_name] = []
    count = 1
    
    for file in files:
        image_file_path = os.path.join(root, file)
        roi_color = get_cropped_face(image_file_path)
        if roi_color is not None:
            cropped_folder = os.path.join(cropped_path, celebrity_name)
            if not os.path.exists(cropped_folder):
                os.mkdir(cropped_folder)
            print("Generationg image to folder: ", cropped_folder)
            cropped_filename = celebrity_name + str(count) + '.png'
            cropped_filepath = os.path.join(cropped_folder, cropped_filename)
            cv2.imwrite(cropped_filepath, roi_color)
            celebrity_filename_dict[celebrity_name].append(cropped_filepath)
            count+=1

print(celebrity_filename_dict)
