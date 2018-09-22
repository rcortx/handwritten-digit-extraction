# Reading a handwritten number from an image of a form

### Problem Statement
Read a hand written number on different types of physical forms. Number may be of arbitary denomination (length), is not OCR friendly (is not enclosed within printed boxes - check below) and may contain noise.
cost of misclassification is 100 times than that of classifying a number as UNKNOWN

Relevant Metrics:
- *Correct Classifications:* the predicted number matches the expected number exactly
- *UNKNOWNs:* predicted number could not be determined reliably
- *Misclassifications:* algorithm predicted number XYZ but the expected number was different.

The *cost of a misclassification is 100x the cost of UNKNOWN* (heavily prefer UNKNOWN to mispredictions)

### Solution
*Performance:* 95%+ number reading accuracy while correctly detecting majority of misclassifications as UNKNOWNS.

### Note: All code has been moved to *aggregated_pipeline.py* except CNN architecture and training code which is in *data_exploration.ipynb*

#### *data_exploration.ipynb* contains data exploration, visualizations and experimentations with same code but is dirty and un-refactored

### Framework features (Reusable code): 

#### Unsupervised Peak Clustering algorithm to remove horizontal/vertical noise from image. Yields pretty good results.

Run in Jupyter notebook from same directory
```
import cv2
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from aggregated_pipeline import image_augmentation_pipeline, filter_horizontal_noise_yaxis_projection, filter_vertical_noise_xaxis_projection
img = cv2.imread("regions/1602300000_ae61d5.png")
label = "1602300000"
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh, blur_img = image_augmentation_pipeline(gray)
masked = np.copy(thresh)
filter_horizontal_noise_yaxis_projection(blur_img, masked)
filter_vertical_noise_xaxis_projection(blur_img, masked)
fig, (ax1, ax2) = plt.subplots(2)
ax1.imshow(masked, cmap="gray")
ax2.imshow(thresh, cmap="gray")
```

#### MergedBox and Box classes as wrappers around opencv Bounding Boxes
Easy to use and extensive functionality classes that support tree like hierarchy, merging, splitting, etc

Run from CLI in same directory
```
from aggregated_pipeline import MergedBox, Box
x, y, w, h = 10, 10, 10, 10
x2, y2, w2, h2 = 5, 5, 10, 10
box = Box([x, y, w, h])
print(box[2], box.area())
mbox1 = MergedBox([x2, y2, w2, h2])
mbox2 = MergedBox(box)
mbox3 = MergedBox([mbox1, mbox2]) # can't contain 

mbox_left, mbox_right = mbox3.recursive_tree_split(x=7)
print(mbox_left)
print(mbox_right)

mbox_merged = mbox_left + mbox_right
print(mbox_merged, mbox3)

# Note: only compares superbox for equality checks for now, not constituent boxes
is_equal = mbox_merged == mbox3 
print("Parent and addition box equal? ", is_equal)

print(mbox_left.overlap(mbox_right))
```

### Pipeline

#### 0. Noise Removal: Unsupervised Peak Clustering
    performing this via finding and merging peaks on x/y projections with provided threshold and removing noise patterns

![alt text][noise]

#### 1. Unsupervised digit bounding box extraction: 
    - First, all contours are detected in an image using opencv and encapsulated in MergedBox/Box classes
    - Performing clustering via merging and splitting Boxes with appropriate heuristics
    - Noticing that boxes will form a tree like hierarchy in an image, merging and splitting until optimum segmentation is achieved
        features were engineered which have high predictivity for finding merge and split candidates. 
    - For now, thresholds are hardcoded and there's scope of using clustering algorithms like kmeans or hierarchial clustering like agglomerative
        over these features
    - We split boxes which may have multiple digits conjoined like '00' by primary detecting and clustering peaks on the image x-axis projection
    - Then separated digit images are extracted (centered and of the same size) for training 

![alt text][dig_sep]

#### 2. Digit Classifier: (CNN)
    Trained only on accurate results from digit extraction algorithm
    97% accuracy on test set, 94% accurate on whole dataset

##### Class Balance histogram:

![alt text][class_balance]

#### 3. Number Reader:
    Since cost of misclassification is 100 times than that of classifying a number as UNKNOWN, probability thresholding is used to ensure classifier is certain about it's prediction for the entire number.
    95%+ accurate - while rejecting UNKNOWNs (for unsure classifications) (approx = 0.97^9) but correctly labels (94%) individual digits

There's a lot of scope for improvement in terms of: 

#### - Improving this Architecture: 
1. Improving CNN model architecture
2. Augmenting data to increase classification accuracy
3. MNIST pretrained models deep learning (transfer learning)
4. Adding two new output classes: digit-boundary class and double-digit class and generating training data for both
5. Adding dual digit detection classes to detect conjoined digits i.e. (00) and generating data by merging two individual digits
6. Improving Bounding box clustering by leveraging the engineered features with a clustering algorithm like DBSCAN, agglomerative
7. Using trained classifier to generate digit boundary scores (on areas of low threshold probabilities)
8. Merge and Split routines can be improved by tuning thresholds further or adding more filters
9. Training a classifier to detect incorrect classifications by feeding training data engineered from probability scores, maximum classwise probability, mean and std deviation, class balance history, past mistake history, raw image data, other engineered features, etc to generate a probability of a digit/number classification being correct. (This should be able to detect things like 1 being likely to be mistaken as 0, 5 being likely to be mistaken as 2, so if probability scores of 5 and 2 are comparable, this is an uncertain classification)

#### - Changing Architecture:
1. Using moving window classifier to detect digits boundaries and digits
2. Using LSTM with Connectionist Temporal Classification (CTC)
3. Autoencoders: encoder/decoder-type approach
4. Stroke completion algorithm to complete faded/unclear digits
5. Investigate YOLO and region proposals, single shot algorithms

#### Some messy research notes:
https://docs.google.com/spreadsheets/d/19vBYosoy1mu7PmBCad_bZ8mePfNMGFI1C2KPwpiBASM/edit?usp=sharing

#### Setup:
```
git clone https://github.com/rbcorx/handwritten-digit-extraction
``` 
Env: python 3.5.2 (setup Anaconda: strongly recommended)
```
cd handwritten-digit-extraction

`pip install -r requirements.txt`
OR
`pip install -r requirements_compressed.txt`
```
##### Install OpenCV (Using Anaconda)

```
conda install opencv
```

##### Use: (Run `pythonw` on MAC as matplotlib requires python to be installed as a framework)
```
>>> from aggregated_pipeline import NumberReader
>>> nr = NumberReader()
```
##### return label after applying probability threshold, 'UNKNOWN' for uncertain
```
>>> nr.read_from_filename("regions/20111823_6884a0.png")
'20111823'
```
##### return uncertain label instead of 'UNKNOWN' for inspection (how many digits were predicted correctly)
```
>>> nr.read_from_filename("regions/20111823_6884a0.png", thresh=False)
'20111823'
```

### Images:

#### Peak Clustering algorithm
![alt text][peak_cluster_y]

![alt text][peak_cluster_x]

#### First 30 images bounding boxes
![alt text][first_thirty_bounding_box]


[noise]: research_output_images/noise_removal.png "Noise Removal Results"
[peak_cluster_y]: research_output_images/pre_post_peak_cluster_compare.png "Peak Clustering X Results"
[peak_cluster_x]: research_output_images/x_project_pre_post_peak_cluster_compare.png "Peak Clustering Y Results"
[first_thirty_bounding_box]: research_output_images/first30_segmented_digits_error_digit_x2_new.png "First Thirty Images Bounding boxes"
[dig_sep]: research_output_images/separated_digits.png "Separated Digits Results"
[class_balance]: research_output_images/class_balance.png "Class Balance Results"

