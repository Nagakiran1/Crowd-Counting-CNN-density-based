# Crowd-Counting
Project is to count the number group of people in an image and to count number of persons associated with each group. Here two enhanced Convolutional models have used in building effective crowd counting architecture.

[**Analysis and Model building** Crowd counting image data source and analysis associated with the data step by step code in Google colab notebook](https://colab.research.google.com/github/Nagakiran1/Crowd-Counting/blob/main/CrowdCouting.ipynb)


[**Testing** :- Click on the link to Test the model in Google Colab environment](https://colab.research.google.com/github/Nagakiran1/Crowd-Counting/blob/main/Testing_CrowdCountingModel.ipynb)

Objectives in Building Crowd Counting model:-
```

            Objectives to find :-

              1)       Total number of group of people in the image frame.,

              2)      Number of person within each group of an image.




```
Here 2 individual Convolutional models have used in extracting the number of groups and number of persons associated with an image respectively.

Here we have used Embeddings based mechanism in extracting the crowd groups associated with an image and People count based convolution network in counting the persons associated with each group.

[image data source for Crowd image datasets](https://github.com/gjy3035/Awesome-Crowd-Counting/blob/master/src/Datasets.md)

Steps involved in Building Crowd Counting model:-
```

            Steps in Crowd Counting model :-

              1)      Preprocess image data and Train a Group Detection Convolutional Neural Network in extracting the count of persons in image,

              2)      Extract image groups associated with each image from Inbetween layer output of Convolutional Neural network,

              3)      Applying Image processing techniques of Contour formation, filtering contours, gausian blur, median blur etc. to segregate group of poeple in image individually,
              
              4)      Train a Person Detection Convolutional Neural Network in extracting the count of persons in image from each group detected,
              
              5)      Segregate the count of persons each group and show case in an image with rectangle.



```

[Crowd counting image data source and analysis associated with the data step by step code in Google colab notebook](https://colab.research.google.com/github/Nagakiran1/Crowd-Counting/blob/main/CrowdCouting.ipynb)

 - to download the pre-trained Models [Pretrained Models](https://github.com/Nagakiran1/Crowd-Counting/blob/main/DownloadData.py)
 - to download sample [labelled character Images](https://github.com/Nagakiran1/Crowd-Counting/blob/main/DownloadData.py)


![alt text](https://github.com/Nagakiran1/Crowd-Counting/blob/main/Capture1.PNG)


**1) Preprocessing Image data and trainig :scissors: from Image :**

 - Select any survelience data of images representing crowd 
 
 -  ***Remove noise and background***
             Model based background removal approach have take at here in removing the background common area of images, followed by applying morphological transformation to suppress the gaps of image noise, further to morphological tranformation Gassian blur and other image jprocessing techniques have used in synthesizing the image dataset.
             
             Contours can be explained simply as a curve joining all the continuous points (along the boundary). The contours are a useful tool for shape analysis and object detection and recognition. Here Contours explained in differentiating each individual character in an image with using [contour dilation](https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html) technique.
             


- ***Pre-processing***
1) The raw data depending on the data acquisition type is subjected to a number of preliminary processing steps to make it usable in the descriptive stages of character analysis. The image resulting from scanning process may contain certain amount of noise

2) Smoothing implies both filling and thinning. Filling eliminates small breaks, gaps and holes in digitized characters while thinning reduces width of line.

            (a) noise reduction

            (b) normalization of the data and

            (c) compression in the amount of information to be retained.
            
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt text](https://github.com/Nagakiran1/Crowd-Counting/blob/main/Models/img1.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt text](https://github.com/Nagakiran1/Crowd-Counting/blob/main/Models/img2.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt text](https://github.com/Nagakiran1/Crowd-Counting/blob/main/Models/img3.png)


 
 
 
**2) Build a ConvNet Model  :scissors:(Character Recognition Model):**


  Convolution Network of 8 layers with 2\*4 layers residual feedbacks used in remembering the Patterns  :scissors: of the Individual Character Images.
  
 
  ![alt text](https://github.com/Nagakiran1/Receipt_Image_Classification-/blob/master/ConvNet1.png)
 
- [x] 1st Model will train on the Individual Crowd Image to quantify number of persons in an image.
- [ ] 2nd Model is same model with last before layer as predictor which will Calculate a Embedding of specified Flatten Neurons ( The Predicted flatten Values will have Feature Information of Persons cluster Images ).
            
  - Convolution Last before layer Embedding Output is considered as Pattern Feature of Image.
  
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt text](https://github.com/Nagakiran1/Crowd-Counting/blob/main/Models/img4.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt text](https://github.com/Nagakiran1/Crowd-Counting/blob/main/Models/img5.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt text](https://github.com/Nagakiran1/Crowd-Counting/blob/main/Models/img6.png)
  

**3) Segregate data for the second model from clusters for groups:**

Crowd groups extracted from 1st model would be segregated with step by step image processing techniques of Contour formation, Gassian blur etc..

\

- !) Once after training the Group Detection model on labelled number of persons data, load the pre trained model in Extracting the groups by Image processing techniques from 5th Layer of trained model predictions.
- !!) Apply Image processing techniques to synthesize the Group detection model outputs.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt text](https://github.com/Nagakiran1/Crowd-Counting/blob/main/Models/img7.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt text](https://github.com/Nagakiran1/Crowd-Counting/blob/main/Models/img8.png)





            
**4) Test and Consolidate Predictions of Crowd Detection model :**

Consolidate predicitons involves, assigning specific ID to each word related contour with the line associated with the word in image, Consolidating all predictions in a sorted series of specific word related contour and letters associated word.
![alt text](https://github.com/Nagakiran1/Crowd-Counting/blob/main/Models/img9.png)
![alt text](https://github.com/Nagakiran1/Crowd-Counting/blob/main/Models/img10.png)
![alt text](https://github.com/Nagakiran1/Crowd-Counting/blob/main/Models/img11.png)

- Predict each character image and label it with the prediction associated with the Optical character recognition technique.
- Fix the word associated with the prediction with the use of word contour and line through line related contour and consolidate all together.


/play rimshot
