from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import random

import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC


main = tkinter.Tk()
main.title("DETECTION AND CLASSIFICATION OF FRUIT DISEASES")  # designing main screen
main.geometry("1300x1200")

fruits_disease = ['Black spot', 'Canker', 'Greening', 'healthy', 'scab']

global filename
global classifier
global X, Y
global X_train, X_test, y_train, y_test


def uploadFruitDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n");


def Preprocessing():
    global X, Y
    X, Y = [],[]

    for i in range(len(fruits_disease)):
        for root, dirs, directory in os.walk('FruitDataset/'+fruits_disease[i]):
            for j in range(len(directory)):
                img = cv2.imread('FruitDataset/'+fruits_disease[i]+"/"+directory[j])
                img = cv2.resize(img,(128,128))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pixel_vals = img.reshape((-1,3))
                pixel_vals = np.float32(pixel_vals)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
                retval, labels, centers = cv2.kmeans(pixel_vals, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                centers = np.uint8(centers)
                segmented_data = centers[labels.flatten()]
                X.append(segmented_data.ravel())
                Y.append(i)
                print('FruitDataset/'+fruits_disease[i]+"/"+directory[j]+" "+str(X[j].shape))
    np.save("features/features.txt.npy", X)
    np.save("features/labels.txt.npy", Y)

    X = np.load("features/features.txt.npy")
    Y = np.load("features/labels.txt.npy")
    text.insert(END, "Total preprocess images are : " + str(X.shape[0]) + "\n\n")



def close():
    main.destroy()


font = ('times', 16, 'bold')
title = Label(main, text='DETECTION AND CLASSIFICATION OF FRUIT DISEASES USING IMAGE PROCESSING')
title.config(bg='white', fg='Black')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Fruits Dataset", command=uploadFruitDataset)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)

processButton = Button(main, text="Image Preprocessing & KMEANS Segmentation", command=Preprocessing)
processButton.place(x=250, y=550)
processButton.config(font=font1)

featuresButton = Button(main, text="Features Extraction", command=featuresExtraction)
featuresButton.place(x=650, y=550)
featuresButton.config(font=font1)

svmButton = Button(main, text="Train SVM Classifier", command=svmClassifier)
svmButton.place(x=50, y=600)
svmButton.config(font=font1)

classifyButton = Button(main, text="Upload Test Image & Classification", command=Classification)
classifyButton.place(x=250, y=600)
classifyButton.config(font=font1)

main.config(bg='white')
main.mainloop()
