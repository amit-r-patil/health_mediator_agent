import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from imutils import paths
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm.notebook import tqdm
from imutils import paths
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Dropout,Conv2D,BatchNormalization,Activation

import pickle





def predict(model, images):
    modelInputImages = []
    
    for image in images:
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,(480,480))
        modelInputImages.append(image)
    
    modelInputImages = np.asarray(modelInputImages).astype('float32')
    
    predictions = model.predict(modelInputImages)
    
    return predictions






def predictReportType(model, imageList):
    images = []
    for image in imageList:
        if image.endswith('.png') :
            image_pixels = cv2.imread(image)
            image_pixels = cv2.cvtColor(image_pixels, cv2.COLOR_BGR2GRAY)
            image_pixels = cv2.resize(image_pixels, (480,480))
            image_pixels = image_pixels.reshape(480, 480, 1)
            image_pixels = image_pixels/255.


            if len(image_pixels.shape) == 3 :
                images.append(image_pixels)

        else :
            image_pixels = plt.imread(image)
            image_pixels = cv2.cvtColor(image_pixels, cv2.COLOR_RGB2GRAY)
            image_pixels = cv2.resize(image_pixels, (480,480))
            image_pixels = image_pixels.reshape(480, 480, 1)
            image_pixels = image_pixels/255.


            if len(image_pixels.shape) == 3 :
                images.append(image_pixels)
            
    images = np.array(images)
            
    predictions = model.predict(images)
    
    return predictions







from flask import Flask, request
import json
from PIL import Image
import cv2





predReprtTypeLabels = ['blood', 'xray']






app = Flask(__name__)

@app.route("/detect_report_type/", methods=['POST'])
def report_type_prediction_model():
    model = keras.models.load_model('models/report_type_classifier_model.h5')
    
    files = request.files.to_dict(flat=False) ## files is a list containing two images.
    results = []
    print(files)
    inputFiles = []
    for i, file in enumerate(files):
        #file.save(f'image-{i}.jpg')
        filedata = files[file]
        #img = Image.open(filedata[0].stream)
        inputFile = file
        filedata[0].save(inputFile)
        inputFiles.append(inputFile)
        #img = cv2.imread('sample_image_'+str(i)+'.jpg')
        
    resultsPreds = predictReportType(model, inputFiles)
    
    results = {}
    for i in range(len(resultsPreds)) :
        predid = resultsPreds[i].argmax()
        classname = predReprtTypeLabels[int(predid)]
        results[inputFiles[i]] = classname
        
    return json.dumps(results)
    

@app.route("/model/", methods=['POST'])
def fracture_prediction_model():
    model = keras.models.load_model('models/fracture_detection_model.h5')
    
    files = request.files.to_dict(flat=False) ## files is a list containing two images.
    results = []
    print(files)
    inputFiles = []
    for i, file in enumerate(files):
        #file.save(f'image-{i}.jpg')
        filedata = files[file]
        #img = Image.open(filedata[0].stream)
        inputFile = file
        filedata[0].save(inputFile)
        inputFiles.append(inputFile)
        #img = cv2.imread('sample_image_'+str(i)+'.jpg')
        
    resultsPreds = predict(model, inputFiles)
    
    results = {}
    for i in range(len(resultsPreds)) :
        if resultsPreds[i] > 0.4 :
            resultValue = 'fracture detected'
        else :
            resultValue = 'fracture not detected'
        results[inputFiles[i]] = resultValue
    return json.dumps(results)


@app.route("/predict_diabetes/", methods=['POST'])
def predict_diabetes():
    data = json.loads(request.data)
    preg = data.get('Pregnancies', 0)
    glucose = data.get('Glucose', 0)
    bp = data.get('BP', 100)
    skinThickness = data.get('SkinThickness', 0)
    insulin = data.get('Insulin', 0)
    bmi = data.get('BMI', 0)
    dpf = data.get('DPF', 0)
    age = data.get('Age', 0)
    
    modelInput = [[preg, glucose, bp, skinThickness, insulin, bmi, dpf, age]]
    diabetesPredictor = pickle.load(open('models/diabetes_prediction_model.pkl', 'rb'))
    result = diabetesPredictor.predict(modelInput)
    if result[0] == '0':
        return {"result": "negative"}
    else :
        return {"result": "positive"}
    


@app.route("/hello", methods=['GET'])
def hello():
    return "Hello...123"


if __name__ == '__main__':
    #from werkzeug.serving import run_simple
    #run_simple('localhost', 9000, app)
    app.run(host="0.0.0.0", port=9000)