import numpy as np
import matplotlib.pyplot as plt
import os   
import pandas as pd
import sys

from skimage import feature, transform
from skimage.color import rgb2gray

# classification required packages

from sklearn.svm import SVC  
from sklearn.model_selection import  train_test_split  
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from Utilities import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
    

args=[]
def train():   
    Datasetimages=pd.read_csv('..\\notebooks\\gicsd_labels.csv', sep = ",", skipinitialspace=True)
    Datasetimages.LABEL = pd.factorize(Datasetimages.LABEL)[0].astype(np.uint16)
    y = Datasetimages['LABEL'].tolist()
    FVdf= Datasetimages.where(Datasetimages['LABEL']==0)
    PVdf= Datasetimages.where(Datasetimages['LABEL']==1)
    NVdf= Datasetimages.where(Datasetimages['LABEL']==2)
    print("There are {} FULL_VISIBILITY images, {} PARTIAL_VISIBILITY images, {} NO_VISIBILITY images in the dataset".format(FVdf['IMAGE_FILENAME'].count(),PVdf['IMAGE_FILENAME'].count(),NVdf['IMAGE_FILENAME'].count()))
    x=[]
    x = np.load("..\\data\\dataset.npy",allow_pickle=True)
    print("[INFO] Loading images into variable ==> x")
    newlabelarray=np.array(y)
    training_classes, training_counts = np.unique(newlabelarray, return_counts=True)
    maxSamples = max(training_counts)
       
    print("The dataset is imbalanced. Each of the {} classes should have {} samples to make it a balanced dataset".format(len(training_classes), max(training_counts)))
    
    # Calculate the number of images to add for each class to have balanced data
    delta = [maxSamples-count for count in training_counts]
    
    X_train_balanced = []
    y_train_balanced  = []
    
    print("Augmenting the image dataset to combat class imbalance problem")
    n_classes = len(np.unique(newlabelarray))
    for c in range(n_classes):
        x_samples, y_samples = getImagesByClass(x, y, c)
        x_samples_balanced = x_samples
        y_samples_balanced = y_samples
        for i in range(int(maxSamples/len(x_samples))-1):
            x_samples_balanced = np.append(x_samples_balanced, transform_samples(x_samples), axis=0)
            y_samples_balanced = np.append(y_samples_balanced, y_samples, axis=0)
    
        diff = maxSamples - len(x_samples_balanced)
        if(diff > 0):
            x_samples_balanced = np.append(x_samples_balanced, transform_samples(x_samples[0:diff]), axis=0)
            y_samples_balanced = np.append(y_samples_balanced, y_samples[0:diff], axis=0)
        x_samples_balanced=list(x_samples_balanced)
        y_samples_balanced=list(y_samples_balanced)
        X_train_balanced.append(x_samples_balanced)
        y_train_balanced.append(y_samples_balanced)
        print("Class {} has {} samples/labels instead of having only {}. {} Samples added.".format(c, len(x_samples_balanced), len(x_samples), delta[c]))
       
    newX = [y for x in X_train_balanced for y in x]
    newY=  [y for x in y_train_balanced for y in x]
    
    
    hogfeat = []
    print("Extracting HOG features from the balanced dataset")
    if os.path.isfile("..\\artifacts\\HoG\\HoGfeatures.npy") :
        hogfeat = np.load("..\\artifacts\\HoG\\HoGfeatures.npy")
        print("HoG features are loaded from HoGfeatures.npy to variable ==> hogfeat")
    else:
        for i in range(0,len(newX)):
            if i > 0 and i % 200 == 0:
                print("[INFO] processed {}/{}".format(i, len(newX)))
            I = newX[i]
            grayim = rgb2gray(I)
            grayim = transform.resize(grayim,(64,64))
            (H_4x4, hogImage) = feature.hog(grayim, orientations=9, pixels_per_cell=(4, 4),
                cells_per_block=(2, 2), transform_sqrt=True, visualize=True)
            hogfeat.append(H_4x4)
        np.save("..\\artifacts\\HoG\\HoGfeatures.npy", hogfeat)
           
    Xhog = np.array(hogfeat).astype("float")
    newY = np.array(newY).astype("float")
    
    Tfeatures=np.transpose(Xhog)
    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(Tfeatures)
    Tscaled_X=np.transpose(scaled_X)
    features, labels = shuffle(Tscaled_X, newY, random_state=1)
    print("Splitting the dataset into 75% training and 25% testing dataset")
    (trainData, testData, trainLabels, testLabels) = train_test_split(features,
        labels, test_size=0.25, random_state=42)
    print("training data points: {}".format(len(trainLabels)))
    print("testing data points: {}".format(len(testLabels)))
    print("Training the HOG features using Linear Support vector classifier")
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(trainData,trainLabels)
    print("[INFO] Succefully trained the classsifier. \n Saving the classifier for further use")
    joblib.dump(svclassifier, '..\\artifacts\\svclassifier.pkl') 
    print("[INFO] Classifier Saved")
    print("accuracy on training data: {}".format(svclassifier.score(trainData,trainLabels)))
    print("accuracy on testing data: {}".format(svclassifier.score(testData,testLabels)))
    predictions = svclassifier.predict(testData)
    print("Showing a final classification report demonstrating the accuracy of the classifier")
    print("Evaluation on Testing dataset")
    print(classification_report(testLabels, predictions)) 
    print("Confusion Matrix-SVM Linear Kernel")
    print("class 0 -  FULL_VISIBILITY\nclass 1 -  PARTIAL_VISIBILITY\nclass 2 -  NO_VISIBILITY")
    cm=confusion_matrix(testLabels, predictions)
    df_cm = pd.DataFrame(cm, index = [i for i in range(0,3)], columns = [i for i in range(0,3)])
    print(df_cm)
    return

def predict():
    if os.path.isfile("..\\artifacts\\svclassifier.pkl"):
        print("[INFO] loading classifier: SVC trained on HoG features...")
        svcprob = joblib.load("..\\artifacts\\svclassifier.pkl")
        print("[INFO] Classifer is loaded as instance ::svc::")
    
    test1 = plt.imread(sys.argv[2])
    grayim = rgb2gray(test1)
    grayim = transform.resize(grayim,(64,64))
        #plt.imshow(grayim)
    (t1_feat, hogImage) = feature.hog(grayim, orientations=9, pixels_per_cell=(4, 4),
            cells_per_block=(2, 2), transform_sqrt=True, visualize=True)
    t1_featrs=t1_feat.reshape(-1,1)
    scalerfinal = MinMaxScaler()
    t1_featsc = scalerfinal.fit_transform(t1_featrs)
    tt1_featsc=np.transpose(t1_featsc)
    t1_predict = svcprob.predict(tt1_featsc)
    print("class 0 -  FULL_VISIBILITY\nclass 1 -  PARTIAL_VISIBILITY\nclass 2 -  NO_VISIBILITY")
    print("==========")
    print("predicted:{}\n".format(t1_predict[0]))
    print("The image belongs to the class: {}".format(t1_predict))
    print("==========")
    return

def main():
    # construct the argument parser and parse the arguments
    if len(sys.argv) < 2:
        print("You must set argument!!!")
        print("Usage: \nTrain the dataset: python main.py --train" )
        print("Prediction: python main.py --predict  ../images/GICSD_50_7_213.png")
    elif len(sys.argv) == 2:
        train()
    else:
        predict()
    
    
if __name__ == '__main__':
    main()