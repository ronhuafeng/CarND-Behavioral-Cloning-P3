# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./image2.jpg "center image"
[image3]: ./image3.jpg "right border image"
[image4]: ./image4.jpg "left border mage"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to read training datasets
* model-nvidia.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code to extract and augment training datasets.
The model-nvidia.py contains the code for training and saving the convolution neural network. The files show the pipeline I used for training and validating the model, and they contain comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 or 5x5 filter sizes and depths between 32 and 128 (model-nvidia.py lines 20-34) 

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras BatchNormalization layer (model-nvidia.py line 24). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model-nvidia.py lines 40/44/48). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model-nvidia.py line 118-124). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model-nvidia.py lines 88/91).

#### 4. Appropriate training data

I use three kinds of datasets:
- Training data that keeps the vehicle driving on the road. 
- Training data that keeps the vehicle driving along the right border of the road. 
- Training data that keeps the vehicle driving along the left border of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a convolution neural network model similar to the Nvidia model mentioned in the course.

At first, I only use training data that keeps the vehicle driving on the road.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that though my first model had a low mean squared error on the training set and low mean squared error on the validation set, the simulated result is not good (the car doesn't know how to turn corners).

After several trials to construct different kinds training data, I found the car will learn to turn corners if I drive it near the border when passing a corner. So I constructed two other kinds of training data:

- Training data that keeps the vehicle driving along the right border of the road. 
- Training data that keeps the vehicle driving along the left border of the road. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model-nvidia.py lines 20-55) consisted of a convolution neural network with the following layers and layer sizes:

- crop layer to cut edges (60, 24), (0, 0))
- BatchNormalization layer
- GaussianNoise(0.01) layer
- conv2D 5x5 filters 24 strides 2x2, activation function 'elu' 
- conv2D 5x5 filters 36 strides 2x2, activation function 'elu' 
- conv2D 2x2 filters 48 strides 2x2, activation function 'elu' 
- conv2D 2x2 filters 64 strides 2x2, activation function 'elu' 
- Flatten layer
- fc layer size 1164, activation function 'elu', with dropout 0.5
- fc layer size 100, activation function 'elu', with dropout 0.5
- fc layer size 50, activation function 'elu', with dropout 0.5
- fc layer size 10, activation function 'elu', with dropout 0.5
- fc layer size 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle driving along the left side and right side of the road so that the vehicle would learn to recover if it's close the border.

The image shows a car near the right border:

![alt text][image3]

The image shows a car near the right border:

![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles.

After the collection process, I had 50,000 number of data points. I then preprocessed this data by fliping and filtering by steering angles.

The fucntion I use to filter steering angles:

model-nvidia.py line 137 to 147

```python
# only consider steering angles that moves the car from right border to middle road
def filterRightBorder(ang):
    return (ang < -0.03), ang

# only consider steering angles that moves the car from left border to middle road
def filterLeftBorder(ang):
    return (ang > 0.000), ang

# only consider steering angles that actually changes the direction of the car
def filterAboveZeroAngle(ang):
    return (abs(ang) > 0.005), ang
```

These kind of filter function will return a (bool, float) tuple, which the first element means the the data points corresponding to this angle will be use to train the model and the second element is the angle.

I finally randomly shuffled the data set and put 1% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 
The ideal number of epochs was 3 to 10 as evidenced by the loss of training dataset and validation dataset.
I used an adam optimizer so that manually training the learning rate wasn't necessary.

Because I use different kinds of datasets, I train the model several rounds.

The first round is use dataset of car driving near the center. Then I train the generated model by transfer learning use datasets of car driving near the right border and the left border alternatively.

| round | dataset      | epochs |
| ----- | ------------ | ------ |
| 1     | center       | 7      |
| 2     | right border | 10     |
| 3     | right border | 10     |
| 4     | left border  | 10     |
| 5     | right border | 5      |
| 6     | right border | 5      |
| 7     | right border | 3      |
