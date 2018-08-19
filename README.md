# Reading a handwritten number from an image

### Note: All code has been moved to `aggregated_pipeline.py` for faster review

#### `data_exploration.ipynb` contains data exploration, visualizations and experimentations with same code but is dirty and un-refactored

### Noteworthy reusable code: 

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
print(mbox_left.overlap(mbox_right))
```

### Pipeline
Currently, I have stitched together a pipeline as follows:

#### 0. Noise Removal: Unsupervised Peak Clustering
    performing this via finding and merging peaks on x/y projections with provided threshold and removing noise patterns


#### 1. Unsupervised digit bounding box extraction: 
    80% accurate on 5998 image dataset (detecting bounding boxes equal to number of digits in label)
    - performing this via merging and splitting Boxes with appropriate heuristics
    - Noticing that boxes will form a tree like heirarchy in an image, merging and splitting until optimum segmentation is achieved
        features were engineered which have high predictivity for finding merge and split candidates. 
    For now, thresholds are hardcoded and there's scope of using clustering algorithms like kmeans or hierarchial clustering like agglomerative
        over these features

#### 2. Digit Classifier: 
    Trained only on accuracte results from digit extraction algorithm
    (basic LogisticRegression with PCA(n=100)) 83% accuracy on test set, 71% accurate on whole dataset

#### 3. Number Reader:
    20% accurate (approx = 0.84^9) but correctly labels (70-80% of) individual digits
    Have used probability threshold for incorrect predictions to classify them as 'UNKNOWN' which is configurable

There's a lot of scope for improvement in terms of: 

#### - Improving this Architecture: 
1. better classifier model like CNNs
2. augmenting data to increase classification accuracy
3. MNIST pretrained models deep learning (transfer learning)
4. adding two new output classes: digit-boundary class and double-digit class and generating training data for both
5. adding dual digit detection classes to detect conjoined digits i.e. (00) and generating data by merging two individual digits
6. improving Bounding box clustering by leveraging the engineered features with a clustering algorithm like DBSCAN, agglomerative
7. Using trained classifier to generate digit boundary scores (on areas of low threshold probabilities)

#### - Changing Architecture:
1. Using moving window classifier to detect digits boundaries and digits
2. Using LSTM with Connectionist Temporal Classification (CTC)
3. Autoencoders: encoder/decoder-type approach
4. Stroke completion algorithm to complete faded/unclear digits

#### Some messy research notes:
https://docs.google.com/spreadsheets/d/19vBYosoy1mu7PmBCad_bZ8mePfNMGFI1C2KPwpiBASM/edit?usp=sharing

#### Setup:
```
git clone https://github.com/rbcorx/handwritten-digit-extraction
``` 
Env: python 3.5.2
```
cd handwritten-digit-extraction

`pip install -r requirements.txt`
OR
`pip install -r requirements_compressed.txt`
```
##### Use:
```
>>> from aggregated_pipeline import NumberReader
>>> nr = NumberReader()
```
##### return label after applying probability threshold, 'UNKNOWN' for uncertain
```
>>> nr.read_from_filename("regions/20111823_6884a0.png")
'UNKNOWN'
```
##### return uncertain label instead of 'UNKNOWN' for inspection (how many digits were predicted correctly)
```
>>> nr.read_from_filename("regions/20111823_6884a0.png", thresh=False)
'20111223'
```
