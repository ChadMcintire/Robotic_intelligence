from fer import Video
from fer import FER
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def run_fer(video):
    face_detector = FER(mtcnn=True)
    input_video = Video(video)
    processing_data = input_video.analyze(face_detector, display=False)

# run_fer("./Q7_vids/Video_Two.mp4")

def process_data(data_file):
    x = []
    angry = []
    disgust = []
    fear = []
    happy = []
    neutral = []
    sad = []
    surprise = []
    file = open(data_file)
    i = 0
    for item in file:
        if(not item.startswith("angry")):
            x.append(i)
            i+=1
            item = item.split(',')
            angry.append(float(item[0]))
            disgust.append(float(item[5]))
            fear.append(float(item[6]))
            happy.append(float(item[7]))
            neutral.append(float(item[8]))
            sad.append(float(item[9]))
            surprise.append(float(item[10].replace("\n", "")))
    print(angry)
    a, = plt.plot(x, angry, label="angry")
    b, = plt.plot(x, disgust, label="disgust")
    c, = plt.plot(x, fear, label="fear")
    d, = plt.plot(x, happy, label="happy")
    e, = plt.plot(x, neutral, label="neutral")
    f, = plt.plot(x, sad, label="sad")
    g, = plt.plot(x, surprise, label="surprise")
    plt.legend(handles=[a,b,c,d,e,f,g])
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.show()

process_data("./data_2.csv")