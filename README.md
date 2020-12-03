# Crowd-Counting
Project is to count the number group of people in an image and to count number of persons associated with each group. Here two enhanced Convolutional models have used in building effective crowd counting architecture.

Crowd counting and Analysis have a plethora of real-world applications such as planning emergency evacuations in case of fire outbreaks, calamitous events, etc. and making informed decisions on the basis of the number of people such as water, food planning, detecting congestion etc

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

              1)      Train a Group Detection Convolutional Neural Network in extracting the count of persons in image,

              2)      Extract image groups associated with each image from Inbetween layer output of Convolutional Neural network,

              3)      Applying Image processing techniques of Contour formation, filtering contours, gausian blur, median blur etc. to segregate group of images individually,
              
              4)      Train a Person Detection Convolutional Neural Network in extracting the count of persons in image,
              
              5)      Segregate the count of persons each group and show case in an image with rectangle.



```

[Crowd counting image data source and analysis associated with the data step by step code in Google colab notebook](https://colab.research.google.com/github/Nagakiran1/Crowd-Counting/blob/main/CrowdCouting.ipynb)

 - to download the pre-trained Models [Pretrained Models](https://github.com/Nagakiran1/Crowd-Counting/blob/main/DownloadData.py)
 - to download sample [labelled character Images](https://github.com/Nagakiran1/Crowd-Counting/blob/main/DownloadData.py)


![alt text](https://github.com/Nagakiran1/Crowd-Counting/blob/main/Capture.PNG)


**1) Optical Scanning :scissors: from Image :**

 - Select any document or letter of having text information 
 ![alt text](https://github.com/Nagakiran1/4-simple-steps-in-Builiding-OCR/blob/master/sample.jpg) 
 -  ***Extract Character boundaries***
             Contours can be explained simply as a curve joining all the continuous points (along the boundary). The contours are a useful tool for shape analysis and object detection and recognition. Here Contours explained in differentiating each individual character in an image with using [contour dilation](https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html) technique.
             Create a boundary to each character in an image with using [OpenCV Contours](https://docs.opencv.org/3.3.0/dd/d49/tutorial_py_contour_features.html) method. 
             Character recognition with the use ofOpenCV contours method

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt text](https://github.com/Nagakiran1/4-simple-steps-in-Builiding-OCR/blob/master/Countours.PNG)


            
***Naming Convention followed***
the extracted Text characters should be labelled with the Original character associated with it.

Here the Naming convention followed for the letters is last letter of file name should be the name associated with the character.
             
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt text](https://github.com/Nagakiran1/4-simple-steps-in-Builiding-OCR/blob/master/character%20Labelling.PNG)
 


- ***Pre-processing***
1) The raw data depending on the data acquisition type is subjected to a number of preliminary processing steps to make it usable in the descriptive stages of character analysis. The image resulting from scanning process may contain certain amount of noise

2) Smoothing implies both filling and thinning. Filling eliminates small breaks, gaps and holes in digitized characters while thinning reduces width of line.

            (a) noise reduction

            (b) normalization of the data and

            (c) compression in the amount of information to be retained.

            

 
 
 
**2) Build a ConvNet Model  :scissors:(Character Recognition Model):**


  Convolution Network of 8 layers with 2\*4 layers residual feedbacks used in remembering the Patterns  :scissors: of the Individual Character Images.
  
 
  ![alt text](https://github.com/Nagakiran1/Receipt_Image_Classification-/blob/master/ConvNet1.png)
 
- [x] 1st Model will train on the Individual Character Images with direct Classification to predict the Images with softmax Classification of Character Categories.
- [ ] 2nd Model is same model with last before layer as predictor which will Calculate a Embedding of specified Flatten Neurons ( The Predicted flatten Values will have Feature Information of Receipt Images ).
            
  - Convolution Last before layer Embedding Output is considered as Pattern Feature of Image.

**3) Load Trained ConvNet OCR model:**

Optical Character recognition last step involves preprocessing of image into specific word related contours and letter contours, followed by prediction and consolidating according to letter and word related contours in an image.


once affter training the model loading the pre-trained Optical character recognition model.

- !) Once after training the OCR model on labelled names data, load the pre trained model in recognising the specific character. .
- !!) Predict each character image and label it with the prediction associated with the Optical character recognition technique.


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![alt text](https://github.com/Nagakiran1/4-simple-steps-in-Builiding-OCR/blob/master/OCR%20workflow.PNG)




            
**4) Test and Consolidate Predictions of OCR :**

Consolidate predicitons involves, assigning specific ID to each word related contour with the line associated with the word in image, Consolidating all predictions in a sorted series of specific word related contour and letters associated word.
![alt text](https://github.com/Nagakiran1/4-simple-steps-in-Builiding-OCR/blob/master/Word_contour.PNG)

- Predict each character image and label it with the prediction associated with the Optical character recognition technique.
- Fix the word associated with the prediction with the use of word contour and line through line related contour and consolidate all together.


/play rimshot
