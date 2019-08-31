# ImageClassification / Methods.py
## How to use the program:

* After running the program, give the path of the dataset.
* The dataset must have 2 folder named Train and Test.

$ tree --Images
          
    .
    ├── Train                       
    │   ├── Airplane                
    │   ├── Cat                    
    │   ├── Dog                     
    │   └── Others     
    └── Test                   
        ├── Airplane                
        ├── Cat                     
        ├── Dog                     
        └── Others                  
    
* After giving the path, choose the methods for image classification.
* For exit , enter q.

## Pseudocode:

* User enters the path when the app runs.
* Checks the train and test labels are same or not.
* If they are the same, create variables to hold images from loaded images and their labels and triggers create_data function. If not turn back to  “1”.
  * Loads the images from the path that is given by user.
  * Resizes the loaded images as specific size.
  * Merges resized images and their labels in a single array.
  * Shuffles the array.
  * Seperates the images and labels from the array. So the main data is created.
* Asks user to choose what method to be used.
* If the input is between 1-4 (kNN-SVM-RandomForest-NaiveBayes) 
  * Triggers HOG function to calculate hog features.
  * HOG function takes the images and extract the features of images.
  * After HOG is done, classifier will be created.
    * For 1 (kNN) Classifier method is -> neighbors.KNeighborsClassifier()
    * For 2 (SVM) Classifier method is -> svm.SVC()
    * For 3 (RandomForest) Classifier method is -> RandomForestClassifier()
    * For 4 (NaiveBayes) Classifier method is -> GaussianNB()
  * After classifier is identified, Method function is triggered to train the model and calculate the accuracy of it.
    * Takes the training and testing data.
    * Trains the data with using identified classifier.
    * Calculates the accuracy.
* If the input is 5 (MLP)
  * Triggers PreprocessDataForMLP for reshaping the images to be used in MLP classifer.
  * After PreprocessDataForMLP is done, classifier will be created.
    * For 5 (MLP) Classifier method is -> MLPClassifier()
  * After classifier is identified, Method function is triggered to train the model and calculate the accuracy of it. 
* If the input is 6 (Tensorflow-Keras)
  * Triggers PreprocessDataForTensorflow for reshaping the images and converting the labels to one-hot-encoding in order to be used in TF function.
  * After PreprocessDataForTensorflow is done, TF function is triggered.
    * Creates the model.
    * Sets up the layers.
    * Compiles the model.
    * Trains the data.
    * Evaluates the accuracy.
* Return to “4”.

# Images From:
* Open Images Dataset V4-V5

# How to download Open Images V4
## [Open Images V4 Image Downloading Tool](https://github.com/EscVM/OIDv4_ToolKit?fbclid=IwAR2Msqh8tzkpdNDC4f1LDv4u43iuBRR9FLyFxDXqHJfXhkvPPyJ2zRNbchg)
