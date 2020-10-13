import numpy as np
import cv2

def random_translate(img):
    rows,cols,_ = img.shape
    
    # allow translation up to px pixels in x and y directions
    px = 2
    dx,dy = np.random.randint(-px,px,2)

    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    #dst = dst[:,:,np.newaxis]
    
    return dst


def random_scaling(img):   
    rows,cols,_ = img.shape

    # transform limits
    px = np.random.randint(-2,2)

    # ending locations
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])

    # starting locations (4 corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(rows,cols))
    
    #dst = dst[:,:,np.newaxis]
    
    return dst


def random_warp(img):
    
    rows,cols,_ = img.shape

    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.06   # this coefficient determines the degree of warping
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.06

    # 3 starting points for transform, 1/4 way from edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4

    pts1 = np.float32([[y1,x1],
                       [y2,x1],
                       [y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
                       [y2+rndy[1],x1+rndx[1]],
                       [y1+rndy[2],x2+rndx[2]]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))
    
    
    #dst = dst[:,:,np.newaxis]
    
    return dst


# utility function to obtain images by class
def getImagesByClass(X, y, class_id):
    
    '''
    Find the images in X with labels equal to class_id.
    Input:
        X: all the images
        y: all the labels of X
        class_id: the id of class 
    Return:
        x_images: the images of the class with id class_id
        y_images: the labels of x_images
    '''
    newXbyclass=[]
    newYbyclass=[]
    for countimg in range(len(X)):
        if (float(y[countimg])==class_id): 
            newXbyclass.append(X[countimg])
            newYbyclass.append(y[countimg])
    return newXbyclass, newYbyclass



# Funtion to apply scaling, warping and translation simultaneously
def transform_samples(x_samples):
    x_samples_transformed = []
    for img in x_samples:
        x_samples_transformed.append(random_translate(random_warp(random_scaling(img))))
    return x_samples_transformed
