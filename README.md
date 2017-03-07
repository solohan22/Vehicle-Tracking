## Vehicle Detection Project

The goal of this project is to perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier. 

The labeled data for vehicle and non-vehicle examples to train the classifier come from a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself. 

[//]: # (Image References)
[image1]: ./output_images/data.png
[image2]: ./output_images/HOG.png
[image3]: ./output_images/sliding-window.png
[image4]: ./output_images/detected-window.png
[image5]: ./output_images/heatmap.png
[video1]: ./project_video.mp4


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images in cell 2. I defined a function data_look(car_list, notcar_list) for exploring the data. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) in functions get_hog_features() and extract_features(), specifically in the similar codes as in Udacity lessons:

```Python
 if color_space != 'RGB':
    if color_space == 'HSV':
      feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
      feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
      feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
      feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
      feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
```

In cell 4, I tested the HOG feature using the HOG parameters `orient = 8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried different sets of parameters and finalized with colorspace='YCrCb', orientations=8, pixels_per_cell=(8, 8) and cells_per_block=(2, 2) and hog_channel = All. I had a few different experimentations as follows

| Color Space   | Orientations  | Pixels_per_cell| Cells_per_block| HOG channel| Accuracy |
| ------------- |:-------------:| -----:| -------------: |:-------------:| -----------:|
| N/A    | 9| 8 | 2     | 0 | 0.92 | 
| YUV    | 8| 8 | 2     | 0 | 0.96 | 
| YCrCb | 8| 7 | 2     | 0 | 0.99 | 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In cell 5, I trained a linear SVM using the vehicle and non-vhicle dataset provided by Udacity. Specifically, I extract the features of car and non-car data
```
car_features = extract_features(cars, color_space=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
notcar_features = extract_features(notcars, color_space=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
```
And then scale and normalize the data
```
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
```
Split the data for training and testing
```
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
```
Traing the classifier
```
svc = LinearSVC()
svc.fit(X_train, y_train)
```
And lastly test it on testing set
```
train_acc = svc.score(X_train,y_train)
test_acc = svc.score(X_test,y_test)
```
Here are some statistics of the SVM classifier:

```
100.33 seconds to extract HOG features...
Feature vector length: 9312
31.79 seconds to train SVC...
Test Accuracy of SVC:  0.9878
SVC predictions:  [ 0.  0.  0.  0.  0.  0.  1.  0.  1.  0.]
Actual 10 labels:  [ 0.  0.  0.  1.  0.  0.  1.  0.  1.  0.]
0.04211 seconds to predict 10 labels with SVC
```

I also tested a neural network classifier which is similar to https://github.com/HTuennermann/Vehicle-Detection-and-Tracking, but the training and predicting is much slower (at least one order of magnitude) than SVM, therefore is not realistic for such a task. 


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented sliding window search mianly in get_boxes() and slide_window() functions. The former is as follows and the latter is similar to the lesson code. 

```Python
def get_boxes(image):
    #Create a list to append scan window coordinates
    boxes = []
    boxes1 = slide_window(image, x_start_stop=[599, None], y_start_stop=[379, 499], xy_window=(64,64), xy_overlap=(0.75, 0.5))
    boxes2 = slide_window(image, x_start_stop=[449, None], y_start_stop=[399, 549], xy_window=(128,80), xy_overlap=(0.75, 0.5))
    boxes3 = slide_window(image, x_start_stop=[299, None], y_start_stop=[419, 699], xy_window=(250,160), xy_overlap=(0.75, 0.5))
    
    for i in range (29):
        chosen1 = np.random.randint(0, len(boxes1))
        boxes.append(boxes1[chosen1]) 
    for i in range (25):
        chosen2 = np.random.randint(0, len(boxes2))
        boxes.append(boxes2[chosen2])           
    for i in range (6):
        chosen3 = np.random.randint(0, len(boxes3))
        boxes.append(boxes3[chosen3])      
    return boxes

```
![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb full-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provides the following example results:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
You could directly download the [video](./result.mp4) in this repo or watch it [online](https://drive.google.com/open?id=0B8g4mCBBmkoacGpjdTFHNUtqbmM)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The video processing is done in function process_video(), in which I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap using add_heat() and then thresholded that map to identify vehicle positions using apply_threshold(). I then identify individual blobs in the heatmap using label() function. I made bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image5]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially, I was all about to use a Deep Neural Netowrk to do the training and predictinng, but the result turns out to be far from efficient. It seems that the deep learning approach serves as a black box and can be easily implemented and tested, but the efficiency for a real-time tracking task like this is really an issue. It might be due to the actual implementation using Keras and Python, I am wondering it could be improved using C++. 

The major diffculty I faced in this project is to tweak this large amount of parameters, it is a bit confusing without any prior experience. Thus I started with similar parameters as those in the lesssons and try to tweak them one at a time and finally got it work. The pipeline tends to work well on the video provided, and it could be improved by adding more training and testing data, maybe using the new Udacity data in csv file, and do better filtering over frames to reduce the jittering. 


