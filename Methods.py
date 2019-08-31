import numpy as np
import os
import cv2
from tqdm import tqdm
import random
from skimage.feature import hog
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier       
from sklearn import svm 
from sklearn.neural_network import MLPClassifier
##\mainpage Download different classes from Open Images Dataset V4
#https://github.com/EscVM/OIDv4_ToolKit?fbclid=IwAR2Msqh8tzkpdNDC4f1LDv4u43iuBRR9FLyFxDXqHJfXhkvPPyJ2zRNbchg

##creates and prepares training and testing data for image classification\n 
#creates path for labels of images and turns labels into indexes depends on their order (ex. cat=0,dog=1)\n
#https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/?completed=/introduction-deep-learning-python-tensorflow-keras/
#@param DATADIR string: the path of training and testing dataset
#@param CATEGORIES list: list of the categories (ex. cat,dog etc.)
#@param data list: training and testing data
#@param images list: training and testing images  
#@param labelIndexes list: labels of images
#@param IMG_SIZE int: size of images for resizing 
def create_data(DATADIR,CATEGORIES,data,images,labelIndexes, IMG_SIZE):
    for category in CATEGORIES:   
        indexes = []
        path = os.path.join(DATADIR,category)  # create path for labels
        indexes = CATEGORIES.index(category)  # for the classification. 0=label1 1=label2 2=label3   

        for img in tqdm(os.listdir(path)):  # iterate over each image per their labels 
            try:
                allImages_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                resizedImages_array = cv2.resize(allImages_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                data.append([resizedImages_array, indexes])  # add this to our data
            except: 
                pass
            
    random.shuffle(data) # to shuffle the data   
    
    for feature,labelIndex in data:
        images.append(feature)
        labelIndexes.append(labelIndex)

##classify\n
#https://scikit-learn.org/stable/supervised_learning.html#supervised-learning        
#@param clf classifier: the classification method  
#@param train_data array: training images
#@param test_data array: testing images
#@param train_label list: labels of training images
#@param test_label list: labels of testing images        
def Methods(clf,train_data, test_data, train_label, test_label):  
    clf.fit(train_data, train_label) 
    
    accuracy = clf.score(test_data, test_label)
    print(accuracy*100)         

##extracts feautres of images with using Histogram of oriented gradients method\n
#https://www.kaggle.com/olaniyan/h-o-g-for-image-classification\n
#https://www.kaggle.com/manikg/training-svm-classifier-with-hog-features\n
#https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html     
#@param images list: training and testing images
#@param hog_features list: extracted features of images       
def HOG(images,hog_features): 
    hog_images = []
    for i in range(len(images)):
        fd, hog_image = hog(images[i], orientations=15, pixels_per_cell=(5, 5),cells_per_block=(2, 2), visualize=True, block_norm="L1", transform_sqrt=False)
        hog_images.append(hog_image)
        hog_features.append(fd) # !! our training datas  
 
##preprocess for images\n
#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html\n
#https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html#sphx-glr-auto-examples-neural-networks-plot-mnist-filters-py\n
#https://www.kaggle.com/ahmethamzaemra/mlpclassifier-example\n
#https://www.python-course.eu/neural_networks_with_scikit.php\n
#https://stackoverflow.com/questions/34972142/sklearn-logistic-regression-valueerror-found-array-with-dim-3-estimator-expec        
#@param imagesMLP list: training and testing images              
def PreprocessDataForMLP(imagesMLP): 
    imagesMLP = [x / 255.0 for x in imagesMLP]
    # Reshape
    imagesMLP = np.array(imagesMLP)
    nsamples, nx, ny = imagesMLP.shape
    imagesMLP = imagesMLP.reshape((nsamples,nx*ny))
    return imagesMLP

##preprocess for images\n
#https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/    
#@param imagesTensorflow list: training and testing images
#@param labelsTensorflow list: labels of training and testing images
#@param categoriesLen int: how many categories(labels) we have (ex. cat,dog --> 2 label and lenght is 2)
#@param IMG_SIZE int: size of images for resizing
#@return imagesTensorflow array: training and testing images
#@return labelsTensorflow array: labels of training and testing images     
def PreprocessDataForTensorflow(imagesTensorflow,labelsTensorflow, categoriesLen, IMG_SIZE):
    #preprocess the data
    imagesTensorflow = [x / 255.0 for x in imagesTensorflow]
    # Reshape
    imagesTensorflow = np.array(imagesTensorflow).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    # convert to one-hot-encoding
    labelsTensorflow = to_categorical(labelsTensorflow, num_classes = categoriesLen) # 1 0 -> cat  0 1 -> dog 
    return imagesTensorflow,labelsTensorflow 

##tensorflow-keras model for training and making classification\n
#https://keras.io/layers/convolutional/\n
#https://www.tensorflow.org/tutorials/keras/basic_classification\n
#https://www.kaggle.com/kanncaa1/convolutional-neural-network-cnn-tutorial\n
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html\n
#https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/\n
#https://pythonprogramming.net/convolutional-neural-network-deep-learning-python-tensorflow-keras/?completed=/loading-custom-data-deep-learning-python-tensorflow-keras/\n    
#https://www.pyimagesearch.com/2018/12/31/keras-conv2d-and-convolutional-layers/\n
#https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf\n
#https://tech.trustpilot.com/forward-and-backward-propagation-5dc3c49c9a05
#https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc\n
#@param trainImages array: training images
#@param trainLabels array: labels of training images
#@param testImages array: testing images
#@param testLabels array: labels of testing images
#@param categoriesLen int: how many categories(labels) we have (ex. cat,dog --> 2 label and lenght is 2)
#@param IMG_SIZE int: size of images for resizing        
def TF(trainImages,trainLabels,testImages,testLabels, categoriesLen, IMG_SIZE):
    #build the model
    #set up the layers
    model = Sequential()
    
    model.add(Conv2D(filters = 32, kernel_size = (7,7),padding = 'Same', 
                     activation ='relu', input_shape = (IMG_SIZE,IMG_SIZE,1)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 
                     activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', 
                     activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(categoriesLen, activation = "softmax"))
    
    # Define the optimizer
    #Adam optimizer: Change the learning rate
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        
    # Compile the model
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    
    model.fit(trainImages, trainLabels, epochs=50, batch_size=64)
    
    test_loss, test_acc = model.evaluate(testImages, testLabels)
    
    print('Test accuracy:', test_acc)    
    
# %% 
    
IMG_SIZE = 128 #size of images for resizing

CATEGORIES = ["Train"]
categoriesLen = 0
CATEGORIES2 = ["Test"]
categoriesLen2 = 0

gotError = False
while(CATEGORIES != CATEGORIES2):
    dataSetPath = (input("Path of Dataset: "))
    
    DATADIR = dataSetPath + "\\Train"  # train images
    CATEGORIES = os.listdir(DATADIR)
    categoriesLen = len(CATEGORIES)
    train_data = []
    train_images = []
    train_labelIndexes = []
    
    DATADIR2 = dataSetPath + "\\Test" # test images
    CATEGORIES2 = os.listdir(DATADIR2)
    categoriesLen2 = len(CATEGORIES2)
    test_data = []
    test_images = []
    test_labelIndexes = []
    
    if(categoriesLen != categoriesLen2 or CATEGORIES != CATEGORIES2):
        print("Train and Test labels must be the same..!")

create_data(DATADIR,CATEGORIES,train_data,train_images,train_labelIndexes, IMG_SIZE)
create_data(DATADIR2,CATEGORIES2,test_data,test_images,test_labelIndexes, IMG_SIZE)

# %%
           
x = "0"
hogIsCalculated = False
while(x != "q"):
    exceptionHandler=0
    try:        
        x = (input("1 - kNN\n2 - SVM\n3 - RandomForestClassifier\n4 - NaiveBayes\n5 - MLP\n6 - Tensorflow\n--> (Enter q for exit) - Enter a number:"))

        if(x=="1" or x=="2" or x=="3" or x=="4"):
            if(hogIsCalculated==False):
                print("Calculating Train Hog Features...")
                train_hog_features = []
                HOG(train_images,train_hog_features)
                print("Calculating Train Hog Features is successfully done!\nCalculating Test Hog Features...")
                test_hog_features = []
                HOG(test_images,test_hog_features)
                print("Calculating Test Hog Features is successfully done!")
                hogIsCalculated = True
            if(x=="1"):
                clf = neighbors.KNeighborsClassifier()
                Methods(clf,train_hog_features, test_hog_features, train_labelIndexes, test_labelIndexes)
                print("kNN method is used!")
            elif(x=="2"):
                clf = svm.SVC()
                Methods(clf,train_hog_features, test_hog_features, train_labelIndexes, test_labelIndexes)
                print("SVM method is used!")
            elif(x=="3"):
                clf = RandomForestClassifier()
                Methods(clf,train_hog_features, test_hog_features, train_labelIndexes, test_labelIndexes)
                print('RandomForestClassifier method is used!')
            elif(x=="4"):
                clf = GaussianNB()
                Methods(clf,train_hog_features, test_hog_features, train_labelIndexes, test_labelIndexes)
                print('Naive Bayes method is used! ')
        else:
            if(x=="5"):
                train_imagesMLP = []
                test_imagesMLP = []
                train_imagesMLP = PreprocessDataForMLP(train_images)
                test_imagesMLP = PreprocessDataForMLP(test_images)
                clf = MLPClassifier(hidden_layer_sizes=(100,100,100), activation='relu', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=10, shuffle=True, random_state=20, tol=0.000001, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
                Methods(clf,train_imagesMLP,test_imagesMLP,train_labelIndexes, test_labelIndexes)
            elif(x=="6"):
                train_imagesTensorflow = []
                train_labelsTensorflow = []
                test_imagesTensorflow = []
                test_labelsTensorflow = []
                train_imagesTensorflow,train_labelsTensorflow = PreprocessDataForTensorflow(train_images,train_labelIndexes, categoriesLen, IMG_SIZE)
                test_imagesTensorflow,test_labelsTensorflow = PreprocessDataForTensorflow(test_images,test_labelIndexes, categoriesLen, IMG_SIZE)
                TF(train_imagesTensorflow,train_labelsTensorflow,test_imagesTensorflow,test_labelsTensorflow, categoriesLen, IMG_SIZE)
                
            elif(x=="q"):
                break
            else:
                print("Invalid Input!")     

    except Exception as e:
        if(exceptionHandler==0):
            print(str(e))
            
# %%                
