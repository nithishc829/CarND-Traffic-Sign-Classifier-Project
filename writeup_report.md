# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/visualization1.jpg "Visualization"
[image2]: ./images/traffic_sign_color.jpg "Color_Traffic_Sign"
[image3]: ./images/traffic_sign_gray.jpg  " Gray_Traffic_Signs"
[image4]: ./images/traffic_sign_normalized.jpg "Normalized Image"
[image5]: ./german_traffic_images/14.jpg "Random Noise"
[image6]: ./german_traffic_images/1.jpg "Traffic Sign 1"
[image7]: ./german_traffic_images/36.jpg "Traffic Sign 2"
[image8]: ./german_traffic_images/15.jpg "Traffic Sign 3"
[image9]: ./german_traffic_images/22.jpg "Traffic Sign 4"
[image10]: ./german_traffic_images/22.jpg "Traffic Sign 5"

[image11]: ./data_augmentation/1_augmented.jpg  "Original "
[image12]: ./data_augmentation/2_augmented.jpg  "Flip along x-axis "
[image13]: ./data_augmentation/3_augmented.jpg  "Flip along y-axis "
[image14]: ./data_augmentation/4_augmented.jpg  "Flip along x-axis and y-axis "
[image15]: ./data_augmentation/5_augmented.jpg  "Affine 1 "
[image16]: ./data_augmentation/6_augmented.jpg  "Affine 2 "
[image17]: ./data_augmentation/7_augmented.jpg  "Affine 3 "
[image18]: ./data_augmentation/8_augmented.jpg  "Affine 4 "
[image19]: ./data_augmentation/9_augmented.jpg  "Salt and Pepper "



[image20]: ./data_visualization/8_1_visual.jpg 
[image21]: ./data_visualization/10_1_visual.jpg 
[image22]: ./data_visualization/12_1_visual.jpg 
[image23]: ./data_visualization/13_1_visual.jpg 
[image24]: ./data_visualization/16_1_visual.jpg 
[image25]: ./data_visualization/25_1_visual.jpg 
[image26]: ./data_visualization/31_1_visual.jpg 





## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

Ans:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data.

![alt text][image20]
![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]
![alt text][image25]

More image samples for visualization are present in the `data_visualization` folder.

![alt text][image1]

From the above graph it is clear that we do not have sufficient samples for all the classes.We can also see that some of the classes have less than 200 images. 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because looking at several images in the data set we can clearly see that we can still recognize the traffic signs even without color information.

Here is an example of a traffic sign image before and after grayscaling.

Color traffic sign

![alt text][image2] 

Gray scale traffic sign

![alt text][image3]

As a last step, I normalized the image data because it helps in training better.With normalization we can ensure that the data we are dealing with are all of same variance.Also removes noise.

![alt text][image4]

I decided to generate additional data because the training set contatins some imbalance i.e the traffic sign per class is not uniform meaning it can be biased towards some classes. Its a good idea to generate some extra samples for traffic sign classes that contain less images to train.



To add more data to the the data set, I used the following techniques

Here is an example of an original image and an augmented image:

  
* Original data
* ![alt text][image11]

* I used gaussian blur to blur the image
* ![alt text][image12]
* I also used random affine transformation to the image
* ![alt text][image15] ![alt text][image16] ![alt text][image17]
  ![alt text][image18]
* I also used opencv flip function to flip image left and right top and bottom
* ![alt text][image13] ![alt text][image14] 
* Finally I added salt and pepper noise to the images randomly
* ![alt text][image19]








#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|      Layer      |                 Description                 |
| :-------------: | :-----------------------------------------: |
|      Input      |             32x32x1 Gray image              |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x5  |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 24x24x15 |
|      Relu       |                                             |
|   Max pooling   |        2x2 stride,  outputs 12x12x15        |
| Convolution 3x3 | 1x1 stride, valid padding, outputs 10x10x30 |
| Convolution 3x3 |  1x1 stride, valid padding, outputs 8x8x48  |
|      Relu       |                                             |
|   Max pooling   |         2x2 stride,  outputs 4x4x48         |
| Convolution 3x3 |  1x1 stride, valid padding, outputs 2x2x64  |
|      Relu       |                                             |
| Fully Connected |                 outputs 128                 |
|      Relu       |                                             |
|     Dropout     |               keep_prob =0.5                |
| Fully Connected |                 outputs 80                  |
|      Relu       |                                             |
|     Dropout     |               keep_prob =0.5                |
| Fully Connected |                 outputs 43                  |
|      Relu       |                                             |

 I initially started my experiments using the LeNet was getting resonably good output of accuracy 91% just before I augmented the data. I wanted to try out similar model of my own. I used the above configuration as my model. 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.


I trained the model using an Adam optimizer , learning rate of 0.0001 , dropout rate of 0.5 and batch size of 128.

* I initially started off with learning rate of 0.01 because of which my validation accuracy was too low. So started to reduce the learning rate and found that 0.0001 was okay.
* I assumed that batch size of 128 was a good number since by default on LeNet we were getting an accuracy of 89%. I did'nt bother changing it.
* Drop out played a major role in accuracy improvement for me. When I initially used my model I was not getting no where close to 80% accuracy. I added dropouts for fully connected layers suddenly I got an accuracy boost of 10% may be because the my model was overfitting and dropouts helped me lot.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 92.01 %
* validation set accuracy of 95.4%
* test set accuracy of 92.6%


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  
  Intially I started off by simply having 2 convolution layer and only one fully connected layer to connect to ouput layers. The model was under performing less than 2% validation accuracy. I thought the model is not deep enough to capture the detail infromation present in Image.
* What were some problems with the initial architecture?
  
  With the model of my own I was not getting even 10% validation accuracy. The problem I thought that model is not deep enough. So added few more convolution layres and also added one more fully connected layer the model was better and got validation accuracy of 73%.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting ?

    * Even with adding more convolution layers the model was not performing great. In order to increase validation accuracy I removed the strides in convlution instead added max pooling and that too only after every 2 convolution layer. It helped a lot.
    * I also made an experiments by removing activations in convolution layers and also having activations after each convolution layers model performance reduced I don't know why.
    * So I added activation only after each max pooling which permormed well.
    * I also ran the model with and without pre-processing I could see 3 % gap when compared to model with pre-processing. Hence I used the pre-processing.
  
  
* Which parameters were tuned? How were they adjusted and why?
    * I had to adjust the learning rate multiple times. Initally started with high learning rate which caused lot of accuracy drop. I ran the model with different learning rate to see which performs well and set that to 0.0001. Later I found that learning rate was not affecting my model much.
    * Dropout played a major role in accuracy improvement of my model from 88% to 95% accuracy. I just added these dropout connections after fully connected layers and model was way better. May be my model was overfitting to the data with dropouts this was minimized.
  
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    * These are some of my observations so far
    * Model pre-processing helps choose if color information is required or not. I tried with GRAY scale image and also LAB L channel images found GRAY to be better.
    * Initially to start with always to keep the learning rate very low close to 0.00001 it may take more epochs but model will be more correct.
    * Number of convolutions at inital states can start of with some number and each stage keep alomst doubling the convolutions with reducing the width and height of image as we go deeper.Once the model reaches required accuracy we still come back and change model to keep minimum convlutions required however you have to retrain the model.
    * More information is present in convolution layer stages so it is critical to decide if we want to be used dropouts there. I used dropouts only for the classification part of the ConvNet i.e fully connected layers to get the required accuracy.
    * Data augmentation well helps to get more data to training set but should keep in mind what kind of augmentation is really useful. Like flipping of data may sometimes lead to misclassification and hence drop in accuracy.


 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

* ![alt text][image5]
* ![alt text][image6] 
* ![alt text][image7] 
* ![alt text][image8] 
* ![alt text][image9]

* The stop sign is just blurred version of the image so it should be easy to classify
* The 30 km speed limit traffic sign is difficult to classify because whe we feed image to model the traffic sign was covering the entire spacial dimensions which is not the case here.
* All the images were of different resolutions, when I resized some traffic imges lost lot of information causing my model to predict wrong output results.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction for 5 images:

|        Image         |      Prediction      |
| :------------------: | :------------------: |
|      Stop Sign       |      Stop sign       |
|     No-Vehicles      |   Go straight left   |
| Go Straight or Right | Go Straight or Right |
|       50 km/h        |       20 km/h        |
|      Bumpy Road      |    Slippery Road     |


* The model was able to correctly guess 2 of the 9 traffic signs, which gives an accuracy of 22.22%. 
* This compared to the accuracy on the test set of very low because the images downloded contained lot of noise and when rescaled to image size of 32x32 lost lots of information.

The cell for this is present in `Load and Output the Images` section of notebook

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the cell `Predict the Sign Type for Each Image`  of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability |      Prediction      | Classfied Correctly |
| :---------: | :------------------: | :-----------------: |
|     .99     |      Stop sign       |         Yes         |
|    .268     |     No-Vehicles      |         No          |
|    .937     | Go Straight or Right |         Yes         |
|     .80     |       50 km/h        |         No          |
|    .463     |      Bumpy Road      |         No          |


* For the first image the probablility is very high and is correctly classified.
* For the second image, the No vehichle sign was misclassified as left sign because probably the noise at the center of image causing image to look like left sign image with low conficence score.
* The third image is properly classified.
* The 50Km/h sign is misclassified with high confidence probably because it looks like 20Km/h image in the upside down version of it.
  May be because since in data augmentation I have used fliping of image as one of augmentation technique.
* This image contained lot of noise hence the prediction neither high nor low .

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


