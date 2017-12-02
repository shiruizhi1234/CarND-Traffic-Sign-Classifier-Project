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
[barplot]:bar_plot.png "bar plot"
[before]:./examples/before.png "before"
[after]:./examples/after.png "after"
[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/1.jpg "Traffic Sign 1"
[image5]: ./examples/2.jpg "Traffic Sign 2"
[image6]: ./examples/3.jpg "Traffic Sign 3"
[image7]: ./examples/4.jpg "Traffic Sign 4"
[image8]: ./examples/5.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/shiruizhi1234/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][barplot]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to YUV and then only take the Y space to get rid of the brightness effect.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][before]
![alt text][after]


As a last step, I normalized the image data because it can get better optimization result.

I decided to generate additional data because the data is distributed unevenly. Some labels only have 400 data. 

To add more data to the the data set, I write some functions to randomly shift the image. Because these transformations will not change the label. 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the I shift the original image randomly by 1 or 2 pixels. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				   |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16|
| RELU     				|
| Max pooling		    |
| flatten			    |	
| Fully connected		|        									    |
| Dropout               |   prob 0.5                                    |
| Fully connected	    |        									    |
| Dropout               |												|
| Fully connected	    |												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used cross entropy as the loss function and adam as the optimization algorithm. And I set the learning rate 0.001, epochs 25 and batch size 128.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99%
* validation set accuracy of 95%? 
* test set accuracy of 92%?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
 
 Lenet, because it is the architecture that was used in the reference paper.

* What were some problems with the initial architecture?
 
 Overfitting

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
 
 Since the training accuracy is high and the validation accuracy is low, I think this is due to overfittion. So I add dropout layer to the model.

* Which parameters were tuned? How were they adjusted and why?

 I tuned the number of epochs to train the model. And I find that increase the epochs to 20 - 25 will improve the performance.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

 Because this is a image classification problem and convolution net can help capture local information. Dropout helps because it reduces the modeling power and force the model capture the rule instead of memorizing the data.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images might be difficult to classify because they are larger than 32 x 32 and resizing may make the center unclear and loss some information.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Snow      		    | Children crossing                             | 
| Road Work     	    | Vehicles over 3.5 metric tons prohibited      |
| No Passing			| Speed limit (120km/h)							|
| Pedestrians	  		| Bicycles crossing					 				|
| Speed limit (30km/h)	| Speed limit (30km/h)                            |


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This is actually werid. I think the reason may be that the image I found is not 32x32 size and resizing may cause some problems.
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the fifth image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (30km/h)   							| 
| .0005     				| Speed limit (70km/h) 						|
| .00004					| Speed limit (20km/h)     |
| .00006	      			| Speed limit (50km/h)    					 				|
| .00006				    | Roundabout mandatory|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
The first layer is detecting a triangle shape or round shape object.


