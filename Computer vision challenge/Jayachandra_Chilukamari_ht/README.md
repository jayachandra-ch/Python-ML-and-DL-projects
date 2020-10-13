***


# Classifying the visibility of ID cards in photos

The folder images inside data contains several different types of ID documents taken in different conditions and backgrounds. The goal is to use the images stored in this folder and to design an algorithm that identifies the visibility of the card on the photo (FULL_VISIBILITY, PARTIAL_VISIBILITY, NO_VISIBILITY).

## Data

Inside the data folder you can find the following:

### 1) Folder images
A folder containing the challenge images.

### 2) gicsd_labels.csv
A CSV file mapping each challenge image with its correct label.
	- **IMAGE_FILENAME**: The filename of each image.
	- **LABEL**: The label of each image, which can be one of these values: FULL_VISIBILITY, PARTIAL_VISIBILITY or NO_VISIBILITY. 


## Dependencies

This work requires:

Python 3.6 and the following libraries installed:

* [Jupyter](http://jupyter.org/)
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)
* [scikit-learn](http://scikit-learn.org/)
* [Open CV](http://opencv.org/)
* [Scikit-image](https://scikit-image.org/)
* [Seaborn](https://seaborn.pydata.org/)

These libraries can be installed using pip. To install the latest version of a package, use: pip install "package name"

Alternatively, install Anaconda distribution. This will install all the required libraries for this challenge including Jupyter notebooks. 

## Run Instructions

- Change to the code directory in the unzipped folder. 

For training the model use the below command-line.

python main.py --train

For predicting an image from the dataset use the below command-line

python main.py --predict ../data/GICSD_2_3_113.png

Sample output of training and testing an image is shown below.

![alt text](./data/Output.png)

## Project files

- The complete training pipeline is explained in the notebook in the form of a story. Please go through the notebook (Machine learning steps.html file) provided under the notebooks directory to get a complete understanding of the machine learning steps used for model training.

- The data directory has the entire image dataset in numpy array file format with file name dataset.npy. Raw images are not provided in the directory. Three sample images one per each class are provided to test the classifier.

- The artifacts directory has pickled version of the classifier with file name svclassifier.pkl

- The misc (research papers) directory has the research papers that are used as the basis for this challenge.


## Approach

Initially exploratory analysis is carried by loading the dataset information from CSV file into a Pandas dataframe. The dataset distribution is visually analysed and Pandas queries are used to count the number of images per each class in the dataset. The class imbalance problem is identified and artifical image synthesis is implemented by translating, scaling and warping the images present in the dataset to obtain a balanced dataset. Taking into account of the complexity of the challenge, advanced feature engineering is implemented by converting the images to gray scale and extracting HOG features from the balanced dataset. A linear SVM is trained on the extracted features. The performance of the classifier is evaluated using simple accuracy metric as the dataset is balanced using image augmentation and also advanced metrics like Precision, recall and F-measure. The confusion matrix is also shown in the notebook to analyse the performance of the classifier on each individual classes. Finally, the trained classifier is used to predict the class of the input image.

- The image classification accuracy using HOG descriptor by a linear SVM is hampered due to noisy images. 

- The test accuracy is decreased as noise made it harder to separate the classes.


## Future Work

- Use feature concatenation technique. Extract different types of features such as local binary pattern (LBPs), HOG and concatenate them and train the model and evaluate the performance.
- Experiment with denoising techniques like gaussian, median and Non local means filter and evaluate the classification performance. 
- Use different machine learning classifiers and spot check the performance. For example, use random forest, decision trees, gradient boosting technique, etc. 
- Use Convolution Neural Network (CNN) for training as these are the state-of-the-art for image classification. Start with transfer learning techniques and then finally train from scratch and evaluate the peformance. 
- The current approach only uses train and test dataset. In the future, use train, validation and test dataset and use grid search to find the best parameters on the validation dataset.
- Use a bigger dataset.
- Plot the learning curves and interpret the bias and variance.
- Define the human-level performance.
- Use manual error analysis and create eye box validation dataset and also black box validation dataset and then do model selection and hyperparameter tuning. 
