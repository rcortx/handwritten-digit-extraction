"""
Setup Env: python 3.5.2
`pip install -r requirements.txt`
OR
`pip install -r requirements_compressed.txt`

Use:

>>> from aggregated_pipeline import NumberReader
>>> nr = NumberReader()

# return label after applying probability threshold, 'UNKNOWN' for uncertain
>>> nr.read_from_filename("regions/20111823_6884a0.png")
'20111823'

# return uncertain label instead of 'UNKNOWN' for inspection (how many digits were predicted correctly)
>>> nr.read_from_filename("regions/20111823_6884a0.png", thresh=False)
'20111823'
"""

import os
import glob
import math
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.signal import find_peaks, peak_widths, peak_prominences
import cv2
from sklearn import svm, linear_model, preprocessing, metrics, model_selection, decomposition
from tensorflow import keras

DIR = "regions"
dig_classes = 10


def timeit(func, label=None, verbose=True):
    """timing decorator"""
    def func_caller(*args, **kwargs):
        t1 = time.time()
        ret = func(*args, **kwargs)
        t2 = time.time()
        verbose_cur = verbose
        if args:
            # checking for internal object verbose setting
            try: 
                verbose_cur = args[0].verbose
            except AttributeError as e:
                pass
        if verbose_cur:
            # use label if available or else function name
            print("time taken for {}: {:.2f}s".format(label if label else func.__name__.upper(), t2-t1))
        return ret
    return func_caller


class CNNDigitClassifier(object):
    """Classifies separated digit of a number using a CNN classifier."""
    
    MODEL_FN = "keras_model_saved_20.h5"
    
    def __init__(self, trained_model=None):
        if not trained_model:
            self.load()
        else: self.model = trained_model
            
    def load(self):
        self.model = keras.models.load_model(CNNDigitClassifier.MODEL_FN)
    
    def preprocess_data(self, X):
        n_samples = len(X)
        reshaped_X = []
        for x in X:
            reshaped_X.append(np.reshape(x, (MergedBox.MAX_HEIGHT, MergedBox.MAX_WIDTH, 1)))
        X = np.array(reshaped_X)
        return X

    def predict(self, X):
        probas = self.predict_probabilities(X)
        return [np.argmax(p) for p in probas]
    
    def predict_probabilities(self, X):
        """returns predicted probabilities instead of classes"""
        X = self.preprocess_data(X)
        return self.model.predict(X)
 

class DigitClassifier(object):
    """Classifies separated digit of a number
    Encapsulates digit image preprocessing, transformation (PCA) and classifier
    Currently using basic classifier: Logistic Regression with PCA(top 100)
    """
    
    PCA_N_COMPS = 100
    CACHE_FILE = "pipeline.pkl" # saves trained transformer and classifier to this file
    
    def __init__(self, load_from_cache=False, transformer=None, classifier=None, verbose=True):
        self.transformer = transformer if transformer else decomposition.PCA(n_components=DigitClassifier.PCA_N_COMPS) 
        self.classifier = classifier if classifier else linear_model.LogisticRegression()
        self.predict_ready = False
        self.verbose = verbose
    
    def log(self, msg):
        if self.verbose:
            print(msg)
    
    def save(self):
        with open(DigitClassifier.CACHE_FILE, 'wb') as fd:
            pickle.dump([self.transformer, self.classifier, ], fd)
            self.log("Classifier saved to disk.")
            
    def load(self):
        if self.predict_ready:
            self.log("Classifier is pre-trained and ready to predict.")
            return
        if os.path.isfile(DigitClassifier.CACHE_FILE):
            with open(DigitClassifier.CACHE_FILE, 'rb') as fd:
                self.transformer, self.classifier = pickle.load(fd)
                self.log("classifier loaded from cache file.")
                self.predict_ready = True
                return True
        self.log("Classifier cache file unavailable.")
        return False
    
    def preprocess_data(self, X):
        n_samples = len(X)
        X = np.array(X, dtype=np.uint8)
        X = X.reshape(n_samples, -1)
        return X
    
    def preprocess_transform_data(self, X):
        """preprocess and transform, without fitting transformer"""
        X = self.preprocess_data(X)
        X = self.transformer.transform(X)
        return X
    
    def preprocess_labels(self, y):
        """conversion to `int`"""
        y = list(map(int, y))
        return y
    
    @timeit
    def train(self, X, y):
        """train on whole dataset"""
        X = self.preprocess_data(X)
        X = self.transformer.fit_transform(X)
        y = self.preprocess_labels(y)
        self.classifier.fit(X, y)
        self.save()
        self.predict_ready = True
    
    def predict(self, X):
        if not self.predict_ready:
            self.load()
        X = self.preprocess_transform_data(X)
        return self.classifier.predict(X)
    
    def predict_probabilities(self, X):
        """returns predicted probabilities instead of classes"""
        if not self.predict_ready:
            self.load()
        X = self.preprocess_transform_data(X)
        return self.classifier.predict_proba(X)
    
    def predict_single(self, x):
        if not self.predict_ready:
            self.load()
        # TODO: validate
        X = self.preprocess_transform_data([x, ])
        return self.classifier.predict(X)
    
    @timeit
    def evaluate(self, X, y):
        """For testing different transformers and classifiers. 
        Ouputs classification report and confusion matrix"""
        X = self.preprocess_data(X)
        y = self.preprocess_labels(y)
        predicted = self.classifier.predict(X)
        self.log("Classification report for classifier %s:\n%s\n"
              % (self.classifier, metrics.classification_report(y, predicted)))
        self.log("Confusion matrix:\n%s" % metrics.confusion_matrix(y, predicted))
        
    @timeit
    def train_test(self, X, y):
        """train and evaluate convenience function"""
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, shuffle=True, stratify=y)
        self.train(X_train, y_train)
        self.evaluate(X_test, y_test)


class NumberReader(object):
    """
    Number Reading class

    Thresholds computed using mean, max values and std dev 
    of incorrect/accuracte digit classifications
    """
    THRESH_99 = 0.948026661481
    THRESH_CORRECT_PROB_MU = 0.805628253442
    THRESH_INCORRECT_PROB_MU = 0.597240586805
    THRESH_INCORRECT_TEST = 0.5
    UNKNOWN_LABEL = "UNKNOWN"
    
    def __init__(self, digit_classifier=None):
        self.digit_classifier = digit_classifier if digit_classifier else CNNDigitClassifier()
    
    def read_from_grayscale(self, image_gray):
        """image_gray has to be in grayscale"""
        digit_images = get_separated_digits(image_gray)
        predictions = self.digit_classifier.predict(digit_images)
        return "".join(map(str, predictions))
    
    def read_from_grayscale_with_proba(self, image_gray):
        digit_images = get_separated_digits(image_gray)
        probs = self.digit_classifier.predict_probabilities(digit_images)
        predictions = self.digit_classifier.predict(digit_images)
        # TODO: optimize: get class from probabilities
        # predictions = list(map(lambda x: np.argmax(x), probs))
        return "".join(map(str, predictions)), probs
    
    def read_from_grayscale_with_thresh(self, image_gray):
        digit_images = get_separated_digits(image_gray)
        probs = self.digit_classifier.predict_probabilities(digit_images)
        predictions = list(map(lambda x: np.argmax(x), probs))
        mxp = list(map(lambda x: x[np.argmax(x)], probs))
        
        if all(map(lambda x: x>NumberReader.THRESH_INCORRECT_TEST, mxp)):
            res = "".join(map(str, predictions))
        else:
            res = NumberReader.UNKNOWN_LABEL
            
        return res
    
    def read_from_image(self, image, thresh=True):
        """image has to be in BGR format"""
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if thresh:
            return self.read_from_grayscale_with_thresh(image_gray)
        return self.read_from_grayscale(image_gray)
        
    def read_from_filename(self, filename, thresh=True):
        if not os.path.isfile(filename):
            print("File doesn't exist!")
            raise ValueError("Incorrect filename.")
        image = cv2.imread(filename) # BGR format
        return self.read_from_image(image, thresh)

    def evaluate_thresh(self, images_gray, labels):
        """evaluates number errors, unknowns and correct labels with threshold"""
        count = 0
        incorrect = []
        correct = []
        unknowns = []
        for i, (img_g, label) in enumerate(zip(images_gray, labels)):
            predicted = self.read_from_grayscale_with_thresh(img_g)
            if predicted == label:
                correct.append((i, label))
            elif predicted == NumberReader.UNKNOWN_LABEL:
                unknowns.append((i, label, predicted))
            else:
                incorrect.append((i, label, predicted))
            count += 1
        print("incorrect: {} - {}%; count: {}".format(len(incorrect), len(incorrect)/float(count)*100, count))
        print("unknowns: {} - {}%; count: {}".format(len(unknowns), len(unknowns)/float(count)*100, count))
        print("correct: {} - {}%; count: {}".format(len(correct), len(correct)/float(count)*100, count))

    def generate_prediction_probability_stats(self, images_gray, labels, return_proba=False):
        """generates digit probability score stats for incorrect and correct labels
        Used to compute viable thresholds

        TODO: optimize: restructure loops for efficient computation
        """
        incorrect = []
        correct = []
        count = 0
        for i, (img_g, label) in enumerate(zip(images_gray, labels)):
            predicted, probabilities = self.read_from_grayscale_with_proba(img_g)
            if predicted == label or predicted == NumberReader.UNKNOWN_LABEL:
                correct.append((i, label, probabilities))
            else:
                incorrect.append((i, label, predicted, probabilities))
            count += 1

        if return_proba:
            return correct, incorrect
            
        correct_digit_count = 0
        incorrect_digit_count = 0
        incorrect_p_stats = []
        for i, label, pred, prob in incorrect:
            label = list(map(int, list(label)))
            for l, p in zip(label, prob):
                mxp = p[np.argmax(p)]
                mu = np.mean(p)
                sigma = np.std(p)
                if l != np.argmax(p):
                    incorrect_digit_count += 1
                    incorrect_p_stats.append((mxp, mu, sigma))
                else: 
                    correct_digit_count += 1
                    correct_p_stats.append((mxp, mu, sigma))

        correct_p_stats = []
        for i, label, prob in correct:
            for p in prob:
                mxp = p[np.argmax(p)]
                mu = np.mean(p)
                sigma = np.std(p)
                correct_p_stats.append((mxp, mu, sigma))
                
        for st in [incorrect_p_stats, correct_p_stats]:
            mx, mu, sigma = list(zip(*st))
            print(len(mx), len(mu), len(sigma))
            print("mx: ", np.mean(mx), np.std(mx))
            print("mu: ", np.mean(mu), np.std(mu))
            print("sigma: ", np.mean(sigma), np.std(sigma))   


class Box(object):
    """Flexible Wrapper around a cv2 bounding box (x, y, w, h)"""

    OVERLAP_X_EXTEND = 0 # pixels to virtually extend width to consider overlap
    
    def __init__(self, box):
        if type(box) is Box:
            self.box = box.box
        else:
            self.box = box

    def __str__(self):
        return "Box(x:{}, y:{}, w:{}, h:{})".format(*self.box)

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        """for sorting and box comparisions, comparing by x coordinate"""
        return self.box[0] < Box.get_box_obj(other).box[0]

    def __eq__(self, other):
        """for box equality comparisions, comparing all box coordinates"""
        return all(map(lambda x, y: x==y, self.box, Box.get_box_obj(other).box))

    def __add__(self, other):
        """other can be instance of MergedBox/Box or list of box coordinates
        Note: destroys constituent boxes!
        """
        return self.merge(Box.get_box_obj(other))

    def __getitem__(self, key):
        return self.box[key]
    
    @staticmethod
    def merge_boxes(box1, box2):
        """merge boxes by taking max/min bounds to encompass largest area"""
        x1, y1, w1, h1 = box1
        xe1 = x1+w1
        ye1 = y1+h1
        x2, y2, w2, h2 = box2
        xe2 = x2+w2
        ye2 = y2+h2
        xn, yn, xen, yen = min(x1, x2), min(y1, y2), max(xe1, xe2), max(ye1, ye2)
        return xn, yn, xen-xn, yen-yn

    def merge(self, b_obj):
        # Note: destroys constituent boxes!
        # TODO: which .box? parent vs child
        return Box(Box.merge_boxes(self.box, b_obj.box))

    def diff(self, b_obj):
        # Note: destroys constituent boxes!
        # TODO: implement
        pass

    def split(self, x, pixel_gap=0):
        """Split this box by provided x coordinate, if x lies in between box bounds, 
                and return two new `Box` instances
        
        `pixel_gap`: artificial gap introduced between the two new boxes, equally divided into both

        Returns 2 `Box` instances in sorted order by x coordinate

        Note: if x lies on Box boundaries, box is not split and the 
            `x` equivalent boundary side (left/right) of the return values is `None`
        """
        if self[0] < x < self[2] + self[0]:
            coor_l = list(self[:])
            coor_r = list(self[:])
            
            coor_l[2] = x - coor_l[0] - pixel_gap//2 - pixel_gap%2
            if coor_l[2] < 0:
                coor_l[2] = 0
            
            coor_r[0] = x + pixel_gap//2
            coor_r[2] = (self[0] + coor_r[2]) - coor_r[0]
            if coor_r[2] < 0:
                coor_r[2] = 0
                coor_r[0] = (self[0] + self[2])

            return Box(coor_l), Box(coor_r)
        elif self[0] == x:
            return None, self
        elif self[2] + self[0] == x:
            return self, None     
        else:
            raise ValueError("`Box` can't be split by unenclosed `x` value")

    def area(self):
        return self.box[-1] * self.box[-2]
    
    @staticmethod
    def overlap_calc(b1, b2, thresh=True):
        """horizontal overlap, negative implies there is distance between boxes"""
        
        if (b1[0] < b2[0]):
            prev = b1
            nx = b2
        else:
            prev = b2
            nx = b1
        x1, y1, w1, h1 = prev
        x2, y2, w2, h2 = nx    
        res = x1 + w1 + (Box.OVERLAP_X_EXTEND if thresh else 0) - x2
        # if x2 < x1:
        #     # implies b2 is behind b1
        #     return -res
        return res
    
    @staticmethod
    def get_box_obj(b_obj):
        # Note: implicit conversion of MergedBox instance to Box by taking superbox
        if type(b_obj) is MergedBox:
            box = b_obj.box
        elif type(b_obj) is Box:
            box = b_obj
        else:
            # assuming b_obj is a tuple of box coordinates
            box = Box(b_obj)
        return box

    def overlap(self, b_obj, thresh=True):
        """overlap between box objects"""
        b1 = self.box
        b2 = Box.get_box_obj(b_obj).box
        return Box.overlap_calc(b1, b2, thresh)


class MergedBox(object):
    """
    MergedBox class to facilitate merging and holding multiple bounding box coordinates
    
    Hierarchial Tree structure: may hold multiple Box/MergedBox objects
    
    class design has been focused on seamless interoperability beween Box and MergedBox 
        instances in terms of add/merge, area and overlap computations
    """
    
    OVERLAP_X_THRESH = 0.5 # fraction of width overlap required to consider merging

    MAX_WIDTH = 70 # max acceptable digit width
    MAX_HEIGHT = 33 # max acceptable digit height 

    def __str__(self):
        return "MergedBox(x:{}, y:{}, w:{}, h:{})".format(*self.box.box)

    def __repr__(self):
        return self.__str__()
    
    def __lt__(self, other):
        """for sorting and box comparisions, comparing by x coordinate
        `other` may be instance of Box/MergedBox
        """
        return self.box < MergedBox.get_mbox_obj(other).box

    def __eq__(self, other):
        """for box equality comparisions, superbox of `self` == `other`
        """
        return self.box == MergedBox.get_mbox_obj(other).box

    def __getitem__(self, key):
        """Access internal boxes directly"""
        return self.boxes[key]

    def __iter__(self):
        """Provides iteraton across sorted sequence of internal boxes"""
        self.sort()
        return self.boxes

    @property
    def box(self):
        # optimized via caching results, check correctness, add tests
        if self.cached:
            return self._res
        
        self._res = self.internal_merge()
        self.cached = True
        return self._res
    
    @box.setter
    def box(self, boxes):
        # boxes has to be list of objects which may be instances of Box/MergedBox
        self.cached = False
        self.boxes = boxes
        
    def __init__(self, boxes=[]):
        """boxes can be MergedBox/Box instance, list of Box/MergedBox instances, 
        list of tuples of box coordinates, or single tuple of box coordinates"""
        # TODO: box label encapsulation, update in init args, add, sub
        # TODO: add support to hold peaks
        
        if type(boxes) is MergedBox:
            boxes = boxes.boxes
        elif type(boxes) is Box:
            boxes = [boxes, ]
        elif boxes:
            if ((type(boxes[0]) is not tuple) and (type(boxes[0]) is not list)):
                # if boxes is non empty and first element of boxes is non iterable, implies: boxes is single
                # tuple of coordinates or boxes is list of Box/MergedBox objects            
                if type(boxes[0]) not in [MergedBox, Box]:
                    # implies boxes is single list of coordinates
                    boxes = [Box(boxes), ]
            else:
                # implies boxes is a nested list of coordinates 
                boxes = list(map(Box, boxes))
        
            # deprecated.
            # raise TypeError("argument `boxes` has to be list of tuples of box coordinates")
        
        self.cached = False # marks if self.box is computed
        self.boxes = boxes 
    
    @staticmethod
    def get_box_list(b_obj):
        """returns list of boxes according to b_obj provided"""
        if type(b_obj) is MergedBox:
            boxes = b_obj.boxes
        elif type(b_obj) is Box:
            boxes = [b_obj, ]
        else:
            pass
            # deprecated.
            # assuming b_obj is a tuple of box coordinates
            # boxes = [Box(b_obj), ]
        return boxes

    @staticmethod
    def get_mbox_obj(b_obj):
        """converts b_obj to MergedBox instance"""
        if type(b_obj) is MergedBox:
            return b_obj
        else:
            return MergedBox(b_obj)

    def __add__(self, b_obj):
        """b_obj can be instance of MergedBox/Box

        # TODO: modify add behaviour: 
        """
        # return MergedBox(self.boxes + MergedBox.get_box_list(b_obj))
        # self.boxes.append(b_obj)
        return MergedBox(self.boxes + [b_obj, ])

    def __sub__(self, b_obj):
        """b_obj can be instance of MergedBox/Box or list of box coordinates"""
        # assuming common box elements are same array objects
        
        # check if b_obj is in list and pop and return 
        # if b_obj in self.boxes:
        #     self.boxes.remove(b_obj)
        #     return self

        boxes = list(self.boxes)
        if b_obj in boxes:
            boxes.remove(b_obj)
            return MergedBox(boxes)
        return MergedBox(list(set(self.boxes).difference(MergedBox.get_box_list(b_obj))))

    def internal_merge(self):
        # TODO: optimize by batch max/min
        if not self.boxes:
            return Box([0, 0, 0, 0])
        cur = self.boxes[0]
        if type(cur) is MergedBox:
            cur = cur.box
        for i in range(1, len(self.boxes)):
            nx = self.boxes[i]
            if type(nx) is MergedBox:
                nx = nx.box
            cur = cur.merge(nx)
        return cur

    def merge(self, b_obj):
        # TODO: which .box? parent vs child
        return self + b_obj
    
    def varea(self):
        """virtual area -> of the superbox encompassing all constituent boxes, is an overestimation and may include noise"""
        return self.box.area()
    
    def area(self):
        """actual area of Box by adding constituent box areas; `self.area` <= `self.varea`"""
        return sum(map(lambda b: b.area(), self.boxes))
    
    def sort(self):
        # TODO: implement caching
        # sort all internal boxes as per `x` start
        self.boxes.sort()
    
    def overlap(self, b_obj, thresh=True):
        """overlap between MergedBox objects
        simple horizontal overlap logic: upgrade to vertical overlap or 
            fragmented `true` overlap as MergedBox constitutents may be discontinous

        `b_obj` can be MergedBox/Box instance or coordinate list as Box.overlap takes care of this
        """
        return self.box.overlap(b_obj.box, thresh)
    
    def best_overlap(self, b1_obj, b2_obj, thresh=True):
        """useful func for merging box with adjacent boxes in sorted list"""
        if self.overlap(b1_obj, thresh) > self.overlap(b2_obj, thresh):
            return b1_obj
        return b2_obj
    
    def ioverlaps(self):
        """calculates array of overlaps between sorted internal boxes"""
        self.sort()
        overlaps = []
        if not self.boxes:
            return overlaps
        bp = self.boxes[0]
        for b in range(1, len(self.boxes)):
            overlaps.append(bp.overlap(self.boxes[b]))
            bp = self.boxes[b]
        return overlaps
    
    def can_merge(self, b_obj):
        """returns whether merge possible with b_obj after applying overlap threshold

        # TODO: add param threshold fraction / absolute
        """
        overlap = self.overlap(b_obj)
        # if overlap > width threshold of current box or candidate box: perform merge
        if (overlap >= self.box[2] * MergedBox.OVERLAP_X_THRESH) or (overlap >= b_obj.box[2] * MergedBox.OVERLAP_X_THRESH):
            return True
        return False

    def flatten(self):
        """flattens the tree heirarchy (Nested `MergedBox`(s) instances in `boxes` member) and outputs the leaves as box coordinates `[x, y, w, h]`"""
        flat = []
        for box in self.boxes:
            if type(box) is MergedBox:
                flat.extend(box.flatten())
            else:
                flat.append(box)
        return flat

    def recursive_tree_split(self, x):
        """Splits the entire tree by x value into two distinct trees

        Doesn't create new boxes unless x lies in between the bounds of a leaf node
        This methods preserves Heirarchial information
        """
        self.sort()
        split_i, bl, br = None, None, None # split index, box_left, box_right

        for i, el in enumerate(self.boxes):
            # comparing by x coordinate only, taking first containing unit
            if el.box[0] <= x <= el.box[0] + el.box[2]:
                if type(el) is MergedBox:
                    bl, br = el.recursive_tree_split(x)
                else:
                    # Found atomic split location!
                    # This is an instance of Box class
                    bl, br = el.split(x, pixel_gap=1)
                split_i = i
                break
            elif el.box[0] < x:
                # handling disjoint fragmented boxes member case
                bl = i
            elif el.box[0] > x and (br is None):
                # handling disjoint fragmented boxes member case
                br = i

        if split_i is not None or (bl is not None and br is not None):
            if split_i is not None:
                # skipping ith element is left or right as it has been split into two
                boxes_l = self.boxes[:split_i] + ([bl, ] if bl else [])
                boxes_r = ([br, ] if br else []) + self.boxes[split_i+1:]
            else:
                boxes_l = self.boxes[:bl+1]
                boxes_r = self.boxes[br:]
                # TODO: remove debug output
                if br > bl + 1:
                    print("\n \n @Tree Split Anomally Detected! \n \n")
            left, right = [(MergedBox(boxes) if boxes else None) for boxes in [boxes_l, boxes_r]]
            return left, right
        else:
            raise ValueError("`MergedBox` can't be split by unenclosed `x` value")

    def get_split_scores(self, peaks=None, project=None, peak_widths_l=None, peak_base_heights=None):
        """
        returns split scores for this Box.
        Used to split a bounding box which could be holding multiple digits

        Feature engineered stats include: (of constituent boxes)
        z_score_widths, z_score_heights, distances, rel_heights, peak_counts, peak_imp, heights

        # TODO: restructure and improve data contract
        """
        flat = self.flatten()
        flat.sort()
        z_score_widths, z_score_heights = [stats.zscore(param) for param in zip(*list(map(lambda x: (x[2], x[3]), flat)))]
        heights = list(map(lambda x: x[3], flat))
        widths = list(map(lambda x: x[2], flat))
        h_mean, h_std = np.mean(heights), np.std(heights)
        w_mean, w_std = np.mean(widths), np.std(widths)
        distances = []
        rel_heights = []
        peak_counts = []
        peak_imp = []
        # TODO: push import to top
        # TODO: replace other z score calculations by stats.zscore
        if flat:
            for i in range(1, len(flat)):
                # distance is -ve of overlap
                distances.append(-flat[i-1].overlap(flat[i]))
                rel_heights.append(heights[i-1]/float(heights[i]) if heights[i-1] < heights[i] else heights[i]/float(heights[i-1]))
            
            if peaks:
                p_i = 0
                peaks_in_boxes = []
                for f in flat:
                    (x, y, w, h) = f
                    peaks_in_boxes.append([])
                    while p_i < len(peaks) and peaks[p_i] <= x+w:
                        # check apply higher threshold? use higher gaussian kernel?
                        peak_imp_score = peak_widths_l[p_i] * (project[peaks[p_i]] - peak_base_heights[p_i])
                        if x <= peaks[p_i] <= x+w:
                            peaks_in_boxes[-1].append((peaks[p_i], peak_imp_score))
                        else:
                            # peak wasted!
                            pass
                        p_i += 1
                peak_counts = [len(x) for x in peaks_in_boxes]
                peak_imp = [(int(sum(list(zip(*x))[1])/1.) if x else 0) for x in peaks_in_boxes]

        hrep = "Height(Mean: {}, std: {})".format(h_mean, h_std)
        wrep = "Width(Mean: {}, std: {})".format(w_mean, w_std)
        
        # caching results: box report
        self.brep = z_score_widths, z_score_heights, distances, rel_heights, peaks_in_boxes, peak_counts, peak_imp, heights, flat

        return hrep, wrep, z_score_widths, z_score_heights, distances, rel_heights, peak_counts, peak_imp, heights

    def split_node(self, discard_noise=True):
        """divides the box into two boxes by sensible overlap heuristic (minimize overlap?)"""
        dist_thresh = 2
        height_thresh = 0.75
        noise_thresh = 5
        split = False
        x_split = None
        if self.brep:
            # getting box reports previously computed
            z_score_widths, z_score_heights, distances, rel_heights, peaks_in_boxes, peak_counts, peak_imp, heights, flat = self.brep
            if distances:
                bst = np.argmax(distances)
                if bst > dist_thresh:
                    if rel_heights[bst] > height_thresh:
                        x_split = flat[bst].box[0] + flat[bst].box[2]
                        split = True
        if not split:
            # TODO: attempt peakwise split
            # just splitting by half for now
            x_split = self.box[0] + self.box[2]//2

        if x_split is not None:
            b1, b2 = self.recursive_tree_split(x_split)
            if discard_noise:
                if b1.box[2] < noise_thresh: 
                    b1 = None
                elif b2.box[2] < noise_thresh:
                    b2 = None
            return b1, b2
        return None
    
    def distance(self, other):
        """opposite of overlap: convenience function"""
        return -self.overlap(other)

    def cut_from_image(self, image, mx_width=None, mx_height=None, discard_extra_crop=False, resize_no_aspect_maintain=False):
        """Cut flattened Box constituents from provided image to generate same size digit training data
        
        Only cuts detected contours with no extra area and minimizes noise added
        
        Returns newly created np array of shape (MergedBox.MAX_WIDTH, MergedBox.MAX_HEIGHT)
        """
        if not mx_width:
            mx_width = MergedBox.MAX_WIDTH
        if not mx_height:
            mx_height = MergedBox.MAX_HEIGHT
        
        
        x_start = self.box[0]
        y_start = self.box[1]
        width = self.box[2]
        height = self.box[3]
        constituent_boxes = self.flatten()
    
        # Note: image height and width are reversed in numpy as height corresponds to num rows: axis 0
        cut_img = np.zeros(shape=(mx_height, mx_width), dtype=np.uint8)
        # dealing with anomally: single digit width > MAX_WIDTH
        # TODO: center cut/resize appropriately
        # if width > mx_width:
        #     # print("box width anomally! Truncating box")
        #     constituent_boxes = [Box([x_start, y_start, mx_width, height]), ]
        #     width = mx_width
        # # dealing with anomally: single digit height > MAX_height
        # if height > mx_height:
        #     # print("box width anomally! Truncating box")
        #     constituent_boxes = [Box([x_start, y_start, width, mx_height]), ]
        #     height = mx_height

        digit_img = self.get_from_image(image)
        # TODO: keep aspect ratio
        #     try not cutting the image
        #     perform maximum required image compression
        # NOTE: this introduces size imbalance in the data
        #     alternative: resize bounding box so that it's limits are 
        #    can CNN be trained to be digit size independent repr.?
        #    difference between smaller 1 and larger 1?
        #    Is this the reason for CNN performance issues and misclassifications?
        
        if resize_no_aspect_maintain:
            # return image after resizing to target size without maintaining aspect ratio
            return cv2.resize(digit_img, (mx_width, mx_height))

        if discard_extra_crop:
            # crop image if image larger than target size
            if width > mx_width:
                # print("box width anomally! Truncating box")
                width = mx_width
            # dealing with anomally: single digit height > MAX_height
            if height > mx_height:
                # print("box width anomally! Truncating box")
                height = mx_height
        else:
            # resize image while maintaining aspect ratio if image larger than target size
            height_ratio = mx_height / float(height)
            width_ratio = mx_width / float(width)

            if width > mx_width or height > mx_height:
                target_ratio = min(height_ratio, width_ratio)
                # resize image without maintaining aspect ratio:
                # digit_img = cv2.resize(digit_img, (mx_width, mx_height))
                
                # resize image while maintaining aspect ration:
                digit_img = cv2.resize(digit_img, (0, 0), fx=target_ratio, fy=target_ratio)
                height, width = digit_img.shape

        # centering training data
        w_offset = (mx_width - width) // 2
        h_offset = (mx_height - height) // 2
        cut_img[h_offset:height+h_offset, w_offset:width+w_offset] = digit_img[0:height, 0:width]

        # for box in constituent_boxes:
        #     (x, y, w, h) = box.box
        #     cut_img[y-y_start+h_offset:y+h-y_start+h_offset, x-x_start+w_offset:x+w-x_start+w_offset] = digit_img[y:y+h, x:x+w]

        return cut_img

    def get_from_image(self, image):
        x_start = self.box[0]
        y_start = self.box[1]
        (x, y, w, h) = self.box
        constituent_boxes = self.flatten()
        cut_img = np.zeros(shape=(h, w), dtype=np.uint8)
        for box in constituent_boxes:
            (x, y, w, h) = box.box
            cut_img[y-y_start:y+h-y_start, x-x_start:x+w-x_start] = image[y:y+h, x:x+w]
        return cut_img

    def peaks(self):
        pass
    
    def box_encloses(self, x):
        pass

    def area_thresh(self, area_thresh):
        """applies area threshold on itself and returns Bool
            # TODO: figure out whether to do on superbox or true area?
        """
        pass


# generating number labels
def get_label(f_name):
    return f_name.split("/")[1].split("_")[0]


def get_digit_map(labels):
    # digit map marking every digit in image_n, digit_pos tuple
    digit_map = {i:[] for i in range(dig_classes)}

    for i, digits in enumerate(labels):
        for j, dig in enumerate(digits):
            digit_map[j].append((i, j))
    return digit_map


def get_grayscale_images_labels():
    """loads images, converts to grayscale and extracts labels"""
    f_names = glob.glob(os.path.join(DIR, "*.png")) 
    labels = list(map(get_label, f_names))

    # checking number lengths
    # digit_lengths = set(map(len, labels))
    images = list(map(cv2.imread, f_names)) # BGR format
    images_gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]

    return images_gray, labels


def gen_subplot_group(n, cols=2, size=1.0, colsize=1.0, yield_group=1):
    """generates index, axes tuple from figure subplots
        generates `yield_group` axes at once
    """
    rows = n//cols
    row_size_mul = 1.4 * size
    col_size_mul = 6 * colsize
    fig = plt.figure(figsize=(int(cols*col_size_mul), int(rows*row_size_mul)))
    group = []
    for i in range(1, rows*cols+1):
        ax = fig.add_subplot(rows, cols, i)
        group.append(ax)
        if len(group) == yield_group:
            yield ((i//yield_group)-1, group)
            group = []


def draw_images(images, labels, gray=False, to_rgb=True, n=10, gauss_blur_kernel=(3, 3)):
    """
    draws `n` images in graymap/converting to RGB from BGR
    gray and to_rgb are mutually exclusive
    
    `gauss_blur_kernel`: (x, y) use Gaussian blur with this kernel size to remove noise 
        with a Gaussian filter. x & y must be odd.
    """
    for i, axes in gen_subplot_group(n, size=2.0):
        ax = axes[0]
        args = [cv2.GaussianBlur(images[i], gauss_blur_kernel, 0), ] # cv2.GaussianBlur(gray,(3,3), 0)
        args = [cv2.threshold(images[i], 200,255,cv2.THRESH_BINARY_INV)[1], ]
        args = [cv2.GaussianBlur(args[0], gauss_blur_kernel, 0), ]
        args[0] = cv2.resize(args[0], (0, 0), fx=2.0, fy=3.0)
        # check inverting image instead of thesh
        kwargs = {}
        if gray: 
            kwargs["cmap"] = "gray"
        elif to_rgb: 
            args[0] = args[0][:, :, ::-1]
        plt.imshow(*args, **kwargs)
        ax.set_xlabel(labels[i] + " : {}".format(len(labels[i])))
        
        
def cluster_peaks(x, peaks, peak_prom, width_results, max_width_only=False, threshold=0.5):
    """
    unsupervised clustering of peaks
    merges overlapping peaks after applying prominence threshold
    suppresses local minima in signals

    Args:
        `peaks`: peak indices
        `peak_prom`: peak prominences
        `width_results`: peak width results from scipy.signal.peak_widths
        `max_width_only`: if True, returns only maximum width peak after merging

    Creates `peak_stats`: format: [(p_i, p_prom, w, w_h, st, en), ...]
            tuples of (peak_indices, peak_prominences, peak_width, peak_base_height, start, end

    Returns:
        `peak_stats` in separated format: 
            peak_indices, peak_prominences, peak_width, peak_base_height, start, end

    # TODO: optimize by using merged candidate bool map instead of delete from array
            optimize by changing presort according to start, end to find candidates faster
            optimize for noise filtering: filter peaks that are on edges
            optimize data contract binding: remove unneccary use of zip/map and improve data interoperability
    """
    # changing zipped tupples to list for mutability
    peak_stats = list(map(lambda x: list(x), zip(peaks, peak_prom, *width_results)))
    peak_stats.sort(key=lambda x: x[2], reverse=True) # sort peaks by width
    # peaks should be merged with a parent peak of minimum width, hence sorting is required
    # TODO: recheck efficient deletes creating dict array for O1 deletes
    # peak_dict = {i: peak_stat for i, peak_stat in enumerate(peak_stats)}
    count = len(peak_stats)

    def remove_cand_from_cur(cur, cand):
        """Removes overlapping but unmergable candidate peak from current peak"""
        # change current's width and remove candidate width from it
        # Note: changes to cur will be reflected in peak_stats list as cur is a list object
        if cand[0] > cur[0]:
            # candidate is right of cur, change cur_end to cand_start
            cur[-1] = cand[-2]
        else:
            # candidate is left of cur, change cur_start to cand_end
            cur[-2] = cand[-1]
        cur[2] = cur[-1] - cur[-2]

    for i in reversed(range(len(peak_stats)-1)):
        cur = peak_stats[i]
        merged = []
        # iterating through every peak with width less than that of current peak
        # to find merge candidates
        for j in range(i+1, count):
            cand = peak_stats[j]
            # if start and end of candidate peak lies between start and end of current peak
            # try to merge with current after applying threshold


            if cand[-2] >= cur[-2] and cand[-1] <= cur[-1]:
                # if prominence height of candidate is less than threshold fraction
                # of that of current peak -> merge
                if cand[1] < threshold * cur[1]:
                    # if base height of candidate is atleast threshold fraction of cur peak height
                    # implies candidate peak is close to current peak in height
                    if cand[3] >= x[cur[0]] * (1 - threshold):
                        merged.append(j)
                    else:
                        # can't merge candidate peak, removing it's edges from current peak
                        remove_cand_from_cur(cur, cand)
                else:
                    # peaks can't be merged, calculate true width of current
                    t_cur_w = cur[2] - cand[2]
                    # find if current peak is left or right of candidate
                    if cur[0] > cand[0]:
                        # current is right of cand, true width is cur_end - cand_end
                        t_cur_w = cur[-1] - cand[-1]
                    else:
                        # current is left of cand, true width is cand_start - cur_start
                        t_cur_w = cand[-2] - cur[-2]
                    # if true width of current is less than candidate, merge current into candidate
                    # while trimming any non-overlapping edges -> just delete current instead
                    if t_cur_w < cand[2]:
                        pass
                        # TODO: merge current? remove false peak: narrow but peakier? 
                        # merged.append(i)
                    remove_cand_from_cur(cur, cand)
        
        # performing merge: just removing merged candidates from peak_dict
        # sorting merge list to delete largest index first
        merged.sort(reverse=True)
        for j in merged:
            del peak_stats[j]
            count -= 1
            
    if max_width_only:
        peak_stats = [max(peak_stats, key=lambda x: x[2]), ]

    return list(zip(*peak_stats))
        
def process_projection_peaks(x, ax=None, ax_plot_diff=None, *args, **kwargs):
    """Finds peaks in projection, clusters them according to prominence and returns
        can also plot signal (pre/post clustering) as per axes provided
    Args:
        `x`: 1D projection of signal
        `ax`: plt axis to visualize peaks post clustering
        `ax_plot_diff`: plt axis to visualize peaks pre clustering
    Returns:
        `res`: [peaks, peak_widths_l, peak_base_heights, starts, ends]
    """
    def plot_peaks(ax, peaks, width_res):
        ax.plot(x, color="y")
        ax.plot(peaks, x[peaks], "x")
        ax.hlines(*width_res[1:], color="red")

    peaks, _ = find_peaks(x)
    peak_prom = peak_prominences(x, peaks)[0]
    width_res = peak_widths(x, peaks, rel_height=1)
    # format: p_i, p_prom, w, w_h, st, en
    
    if ax_plot_diff:
        plot_peaks(ax_plot_diff, peaks, width_res)
        ax_plot_diff.set_ylabel("Pre Clustering")

    peaks, peak_prom, peak_widths_l, peak_base_heights, starts, ends = cluster_peaks(x, peaks, peak_prom, width_res, *args, **kwargs)

    res = [peaks, peak_widths_l, peak_base_heights, starts, ends]
    if ax:
        peaks = np.array(peaks)
        plot_peaks(ax, peaks, list(res[1:]))
        ax.set_ylabel("Post Clustering")
        
    return res

def plot_projections(images_gray, labels=None, n=10, axis=1, plot_diff=False, *args, **kwargs):
    """plots vertical/horizontal projections (sum of pixels) for grayscale images
        after performing peak clustering 
        # TODO: documentation
    """    
    if plot_diff:
        n = 2*n

    for i, axes in gen_subplot_group(n, size=2.5, yield_group=(1+plot_diff)):
        # TODO: streamline experimental image pre processing pipeline
        ret, thresh = cv2.threshold(images_gray[i], 200, 255,cv2.THRESH_BINARY_INV)
        blur_img = cv2.GaussianBlur(thresh, (5, 3), 0)
        
        # TODO: deprecated. 
        # scaled_image = (images_gray[i]) / 255
        
        # scaling image for better visualizations
        scaled_image = blur_img - 255
        project = np.sum(scaled_image, axis=axis)
        
        process_projection_peaks(project, *(list(reversed(axes)) + args), **kwargs)
            
        print(labels[i])
        
        for ax in axes:
            ax.set_xlabel(labels[i] + " : {}".format(len(labels[i])))
            

""" Notes
# TODO: check box filter post merging, prev boxes don't need to be nullified?
        optimize for identical boxes: same tuple by object ID
# Logic
# area filter merge target: ignore or merge (proximity/overlap threshold before merge/ignore)
#     merged noise: separate by distance
# keep top `n` by area?
"""


def merge_routine(bbs, merge_func, passes=2):
    """generalized merge routine for boxes"""
    # one way adjacency? as i starts from `0`, already considered `prev` at i > 0?
    # fragmented box best overlap? components instead of super? for true overlap?
    # recursive merge in adjacency window? agglomerative heirachial clustering?
    # collapsing array?
    # retaining heirarchial structure via embedding Box within Box or subclassing Box
    for _ in range(passes):
        bbs.sort()
        # sorting by x coordinate
        for i, b in enumerate(bbs):
            # TODO: merge loops
            merge_func(bbs, i, b)
        bbs = list(filter(lambda x: x, bbs))
    bbs.sort()
    return bbs


def get_merge_func(adjacency_window_size=1, predecessor_only=False, thresh_func=lambda x: True):
    """returns box merger function which can be passed to `merge_routine`"""
    def merge_candidate(bbs, i, b, adjacency_window_size=1):
        lims = (i-adjacency_window_size, i+(adjacency_window_size if not predecessor_only else 0)+1)
        adjacents = [(j, bbs[j]) for j in range(*lims) \
                     if j != i and j>=0 and j<len(bbs) and bbs[j] is not None]
        
        if adjacents:
            # best overlap in adjacents
            j, a = max(adjacents, key=lambda x: b.overlap(x[1]))
            
            # TODO: comment architecture philosophy
            if thresh_func(b, a):
                # pushing merged box in maximum index btw i, j so it may be referenced in subsequent adjacency iterations
                bbs[max(i, j)] = b + a
                bbs[min(i, j)] = None
    return merge_candidate


def box_merge_routines(bbs):
    # Box merge routines: overlap threshold
    thresh_func = lambda x, y: x.can_merge(y)
    merge_function = get_merge_func(thresh_func=thresh_func)
    bbs = merge_routine(bbs, merge_func=merge_function, passes=1)
    
    # Box merge routines: area Gauss filter, overlap threshold and best adjacent merge
    areas = list(map(lambda x: x.varea(), bbs))
    mean_a = np.mean(areas)
    std_a = np.std(areas)
    thresh_a = mean_a - std_a

    thresh_func = lambda x, y: (x.area() <= thresh_a or y.area() <= thresh_a) and (x.overlap(y) > -2)
    merge_function = get_merge_func(thresh_func=thresh_func)
    bbs = merge_routine(bbs, merge_func=merge_function, passes=1)

    thresh_func = lambda x, _: x.area() <= thresh_a - std_a*0.75
    merge_function = get_merge_func(thresh_func=thresh_func)
    bbs = merge_routine(bbs, merge_func=merge_function, passes=1)

    return bbs
    

Y_AXIS_PROJECTION_DEF_THRESHOLD_PREV = 0.5 # has bugs: see end of file section
Y_AXIS_PROJECTION_DEF_THRESHOLD_TEST = 0.57
def filter_horizontal_noise_yaxis_projection(project_img, mask_img):
    project = np.sum(project_img, axis=1)
    res = process_projection_peaks(project, threshold=Y_AXIS_PROJECTION_DEF_THRESHOLD_TEST)
    peaks, peak_widths_l, peak_base_heights, starts, ends = res
    
    mx_w_ind = np.argmax(peak_widths_l)
    st, en = starts[mx_w_ind], ends[mx_w_ind]

    mask_img[:int(st), :] = 0
    mask_img[int(en):, :] = 0
    

def filter_vertical_noise_xaxis_projection(project_img, mask_img):
    project = np.sum(project_img, axis=0)
    peaks, peak_widths_l, peak_base_heights, starts, ends = process_projection_peaks(project, threshold=0.65)
    mean_width = np.mean(peak_widths_l)
    mean_height = np.mean(list(map(lambda x: project[x], peaks)))
    peak_stats = list(zip(peaks, peak_widths_l, starts, ends))
    peak_stats.sort(key=lambda x: x[2] * 10000 + x[3])
    # high peak candidates are more important to be removed
    # apply peak threshold
    rem_cand = peak_stats[:5] + peak_stats[-5:] # only take edge candidates for removal
    rem_cand.sort(key=lambda x: x[1]) # sort candidates by width candidates

    for cand in rem_cand:
        # if candidate width is less than half of mean_width
        # remove candidate from image
        # TODO: hyperparameters
        if 0 <cand[1] < 0.6 * mean_width and mean_height * 1.5 < project[cand[0]]:
            # if candidate is on left of image
            if cand[-1] < (project_img.shape[1] / 2.):
                mask_img[:, int(cand[-2]):int(cand[-1])] = 0
            # else candidate is on right of image
            else: 
                # TODO: merge filters
                mask_img[:, int(cand[-2]):int(cand[-1])] = 0


def get_contour_bounding_boxes(img, draw_img=None):
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for (k, c) in enumerate(contours):
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # only accept the contour region as a grouping of characters if
        # the ROI is sufficiently large
        if w >= 2 and h >= 2:
            boxes.append((x, y, w, h))
            if draw_img is not None:
                cv2.rectangle(draw_img, (x,y), (x+w, y+h), (255,0,0), 1)
    return boxes


def filter_by_dips(filter_targets):
    """filter boxes by maximum dips in sorted box split scores
    
    Deprecated.
    """
    filter_targets.sort(key=lambda x: x[0][0], reverse=True)
    
    dips = [0, ]
    for i in range(1, len(filter_targets)):
        dips.append(filter_targets[i-1][0][0] - filter_targets[i][0][0])   
    cands = filter_targets[:np.argmax(dips)] # split by max dip in stats
    
    # cands = filter_targets[:5] # truncate to get candidates
    cands.sort(key=lambda x: x[0][1], reverse=True)
    dips = [0, ]
    for i in range(1, len(cands)):
        dips.append(cands[i-1][0][1] - cands[i][0][1])
    
    cands = cands[:np.argmax(dips)]
    return cands


def get_box_reports(clone_masked, bbs):
    """returns box reports which are used as heuristics to determine viability of box split"""
    blurred_clone = cv2.GaussianBlur(clone_masked, (3,3), 0)
    canny_res = cv2.Canny(blurred_clone, 300, 900)
    project = np.sum(canny_res, axis=0) # clone_masked
    peaks, peak_widths_l, peak_base_heights, starts, ends = process_projection_peaks(project, threshold=0.75)
    # TODO: machine learning n_peaks -> dual digits
    peaks = sorted(peaks)
    peaks_in_boxes = []
    wasted_peaks = []
    p_i = 0
    
    bbs.sort()
    boxes = list(map(lambda x: x.box.box, bbs))
    
    # TODO: optimization: b_widths from find contour bounding boxes
    b_widths = list(map(lambda x: x[2], boxes))
    w_mean, w_std = np.mean(b_widths), np.std(b_widths)
    breps = []
    for box, bb in zip(boxes, bbs):
        
        # Split candidate stats
        brep = bb.get_split_scores(peaks=peaks,
                                         project=project,
                                         peak_widths_l=peak_widths_l,
                                         peak_base_heights=peak_base_heights)
        breps.append(brep)
        peaks_in_boxes.append([])
        (x, y, w, h) = box
        # NOTE: convert to float to maintain python2 compatibility
        w_z_score = (w-w_mean) / float(w_std)
        if w_z_score < 0:
            w_z_score = 0
        while p_i < len(peaks) and peaks[p_i] <= x+w:
            # check apply higher threshold? use higher gaussian kernel?
            peak_imp_score = w_z_score*10#peak_widths_l[p_i] * (project[peaks[p_i]] - peak_base_heights[p_i])
            if x <= peaks[p_i] <= x+w:
                peaks_in_boxes[-1].append((peaks[p_i], peak_imp_score))
            else:
                wasted_peaks.append(peaks[p_i])
            p_i += 1
    
    while p_i < len(peaks):
        wasted_peaks.append(peaks[p_i])
        p_i += 1
    return breps, peaks_in_boxes


def get_split_scores(breps, peaks_in_boxes):
    """generates box split scores from box reports"""
    # needs: peaks_in_boxes, breps; returns scores
    peak_counts_l = [len(x) for x in peaks_in_boxes]
    peak_imp_l = []
    for x in peaks_in_boxes:
        cur = 0
        if x:
            cur = sum(list(zip(*x))[1])/1.
            if not math.isnan(cur):
                  cur = int(cur)
        peak_imp_l.append(cur)

    peak_counts_s = ""
    scores = []
    for j, b in enumerate(breps):
        hrep, wrep, z_score_widths, z_score_heights, distances, rel_heights, peak_counts, peak_imp, heights = b
        dscore = max(distances) if distances else 0
        relscore = min(rel_heights) if rel_heights else 0
        bcount = len(distances)
        hscore = np.sum(heights)
        
        score_1 = peak_counts_l[j]*10 + peak_imp_l[j]
        score_2 = dscore/3. + bcount*13 + hscore*0.7 + relscore*10

        scores.append((score_1, score_2, j))
    return scores


def filter_boxes(filters):
    """filter boxes by applying FIXED threshold to split scores

    # TODO: make thresholding more dynamic, huge accuracy gains possible
    """
    # needs: filters, returns cands, peak_counts_s (test label)
    # from sklearn.preprocessing import MinMaxScaler
    # r = MinMaxScaler()
    y = list(map(lambda x: x[2], filters))
    X = np.array(list(map(lambda x: (x[0], x[1]), filters)))
    
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    zscores = (X - x_mean) / x_std
#     filters = list(r.fit_transform(X))
    filters = list(zip(filters, zscores))

    cands = filters # filter_by_dips(filters)
    cands = list(filter(lambda x: x[0][0] > 50 and x[0][1] > 45, cands)) # 50, 45 # outlier: (70, 23.8) 45 close
    peak_counts_s = "{}".format([("%.2f"%cands[j][0][0], "%.2f"%(cands[j][0][1]), cands[j][0][2]+1) for j, c in enumerate(cands)])

    return cands, peak_counts_s


def box_split_routine(clone_masked, bbs):
    """generic box splitting routine for boxes which may contain multiple digits"""
    # test: detect multiple peaks in conjoined images:
    breps, peaks_in_boxes = get_box_reports(clone_masked, bbs)
    # generating split scores for every Box from reports
    filters = get_split_scores(breps, peaks_in_boxes)
    
    # filtering boxes based on received scores
    cands, peak_counts_s = filter_boxes(filters)
    for cand in cands:
        b_ind = cand[0][2]
        splits = []
        b1, b2 = bbs[b_ind].split_node(discard_noise=True)
        if b1:
            splits.append(b1)
        if b2:
            splits.append(b2)
        
        bbs[b_ind] = splits
    
    bbsnew = []
    for j, bb in enumerate(bbs):
        if type(bb) is list:
            bbsnew.extend(bb)
        else:
            bbsnew.append(bb)
    return bbsnew


def box_filter_routines(bbs):
    # TODO: DEBUG: experimental filter logic
    # remove horizontal noise:
    def is_not_horizontal_noise(bb):
        return bb.box[2] < 2 * bb.box[3] or bb.box[3] > 5

    return [bb for bb in bbs if is_not_horizontal_noise(bb)]


def get_digit_bounding_boxes(
    thresh, blur_img, debug=False, pyplot_axis_res=None,
    pyplot_axis_prev=None, label=None, return_clone=False):
    """returns every digit bouding box in number image"""
    # TODO: avoid copying thresh
    masked = np.copy(thresh)
    # y axis projection
    filter_horizontal_noise_yaxis_projection(blur_img, masked)
    # x axis projection
    filter_vertical_noise_xaxis_projection(blur_img, masked)
    
    # TODO: optimization: limit image copies
    # TODO: optimization: reuse Gaussian blurred/thresholded image for digit classification
    clone_masked = masked.copy() # used in box_split_routine
    
    if debug:
        box_filtered = masked.copy() # for drawing bounding boxes
        clone = masked.copy()
    # DEBUG: draw_img: used to compare boxes found, used in get_contour_bounding_boxes
    # TODO: try drawing boxes on gaussian Blurred image instead of thresholded image
        boxes = get_contour_bounding_boxes(masked, draw_img=clone)
    else:
        boxes = get_contour_bounding_boxes(masked)
    
    bbs = list(map(MergedBox, boxes))
    # TODO: DEBUG: EXPERIMENTAL
    bbs = box_filter_routines(bbs)
    bbs = box_merge_routines(bbs)

    # Box SPLIT routines: canny peaks counts/importance, box width Gauss filter
    bbs = box_split_routine(clone_masked, bbs)

    if debug:
        # DEBUG: image labels
        bbs.sort(key=lambda x: x.area(), reverse=False)
        
        peak_counts_s = "{} vs found: {}, weights_z: {}".format(
            len(label),
            str(len(bbs)),
            list(map(lambda x: math.ceil(x*100)/100., stats.zscore(list(map(lambda x: x.varea(), bbs)))))
        )
        
        for box in bbs:
            (x, y, w, h) = box.box
            cv2.rectangle(box_filtered, (x,y), (x+w, y+h), (255, 0, 0), 1)
        
        pyplot_axis_res.imshow(box_filtered, cmap="gray")
        pyplot_axis_res.set_xlabel(peak_counts_s)

        pyplot_axis_prev.imshow(clone, cmap="gray") # clone, thresh
        pyplot_axis_prev.set_xlabel(label + " : {}".format(len(label)))

    if return_clone:
        return clone_masked, bbs

    return bbs


def image_augmentation_pipeline(image_gray):
    """applying Gaussian Blur and inverted binary thresholding"""
    ret, thresh = cv2.threshold(image_gray, 200, 255,cv2.THRESH_BINARY_INV)
    # Gaussian filter will remove noise, kernel size has been optimized by trial and error
    # the x projection has much higher noise fluctuations and is more affected
    # by kernel choices
    blur_img = cv2.GaussianBlur(thresh, (3, 3), 0)

    return thresh, blur_img


def plot_digit_boxes(images_gray, labels, n=30): # debug function
    # TODO: pipeline image transformations and remove repetitive code
    for i, axes in gen_subplot_group(n*2, size=1.0, yield_group=2):
        ax1, ax2 = axes
        img_g = images_gray[i]
        thresh, blur_img = image_augmentation_pipeline(img_g)
        boxes = get_digit_bounding_boxes(thresh, blur_img, debug=True, pyplot_axis_res=ax1, pyplot_axis_prev=ax2, label=labels[i])


def get_digit_bounding_boxes_baseline(img_thresh, blur_img, thresh=True):
    """basline contour bounding box to establish accuracy with noise cleaning 
        but without merge/split routines"""
    masked = np.copy(img_thresh)
    # y axis projection
    filter_horizontal_noise_yaxis_projection(blur_img, masked)
    # x axis projection
    filter_vertical_noise_xaxis_projection(blur_img, masked)

    boxes = get_contour_bounding_boxes(masked)
    boxes = list(map(MergedBox, boxes))

    # TODO: apply area filter
    if thresh:
        areas = list(map(lambda x: x.area(), boxes))
        widths = list(map(lambda x: x.box[2], boxes))

        z_score_areas = stats.zscore(areas)
        z_score_widths = stats.zscore(widths)

        threshold_area = 1.8
        threshold_width = 2
        filter_func = lambda x: abs(z_score_widths[x[0]]) < threshold_width and abs(z_score_areas[x[0]]) < threshold_area
        boxes = list(map(lambda x: x[1], filter(filter_func, enumerate(boxes))))
    
    return boxes


def accuracy_boxes(images_gray, labels, baseline=False):
    """
    Computes accuracy for bounding box detection algorithm

    Found 20% bounding box detection error rate
    """
    processed = 0
    incorrect = 0
    # get max height and width
    mxw = [0, ]
    mxh = 0

    t1 = time.time()
    for i, (img, lb) in enumerate(zip(images_gray, labels)):
        img_g = images_gray[i]
        thresh, blur_img = image_augmentation_pipeline(img_g)
        if baseline:
            boxes = get_digit_bounding_boxes_baseline(thresh, blur_img)
        else:
            boxes = get_digit_bounding_boxes(thresh, blur_img, debug=False)
        for box in boxes:
            if box.box[2] > mxw[-1]:
                mxw.append(box.box[2])
                mxw.sort()
            if box.box[3] > mxh:
                mxh = box.box[3]
        if len(boxes) != len(labels[i]):
            incorrect += 1
        processed += 1
        if processed % 1000 == 0:
            t2 = time.time()
            print("1000 processed in time: {0:.2f}s".format(t2-t1))
            print("% incorrect: {0:.2f}".format(incorrect/float(processed)*100))
    t2 = time.time()
    print("Net processing time: {0:.2f}s".format(t2-t1))
    print("% NET ERROR RATE: {0:.2f}".format(incorrect/float(processed)*100))
    print("Max width, height of digits: ", mxw, mxh)


def get_separated_digits(image_gray, ret_boxes=False, mx_width=None, mx_height=None):
    """given a grayscale number image, returns separated and normalized digit image clips"""
    thresh, blur_img = image_augmentation_pipeline(image_gray)
    clone, boxes = get_digit_bounding_boxes(thresh, blur_img, debug=False, return_clone=True)
    digit_images = []
    for box in boxes:
        digit_images.append(box.cut_from_image(clone, mx_width, mx_height))
    if ret_boxes:
        return digit_images, [box.box.box for box in boxes]  # box is instance of MergedBox
    return digit_images


def generate_digit_training_data(images_gray, labels, limit=10, skip_incorrect_data=True, cache=True):
    """generates separated digits and labels as training data
    
    if `skip_incorrect_data` is set, ignores all numbers for which no. of bounding boxes doesn't match label size

    returns digit images, digit labels and digit unique identifier for saving/referencing
    """
    training_data_pickle_file = "train.pkl"

    if cache and os.path.isfile(training_data_pickle_file):
        with open(training_data_pickle_file, "rb") as fd:
            digit_images, digit_labels, digit_ids, skip_count = pickle.load(fd)
        print("skipped: {}, {}".format(skip_count, skip_count/float(len(labels))*100))
        return digit_images, digit_labels, digit_ids

    digit_images = []
    digit_labels = []
    digit_ids = []
    skip_count = 0

    if limit is None:
        # take all of data if limit is None
        limit = len(labels)

    for i, (img_g, label) in enumerate(zip(images_gray, labels)):
        if i > limit:
            break
        d_imgs = get_separated_digits(img_g)
        d_labels = list(label)
        d_ids = [] 
        if len(d_labels) != len(d_imgs):
            if skip_incorrect_data:
                skip_count += 1
                continue
            min_len = min(len(d_labels), len(d_imgs))
            d_imgs = d_imgs[:min_len]
            d_labels = d_labels[:min_len]
        # digit identifier to save digit file. In format : <digit_value-(0. 9)> : <number_label> : <digit_index_in_number>
        d_ids = list(map(lambda x: "{}:{}:{}".format(x[1], label, x[0]), enumerate(d_labels)))
        digit_ids.extend(d_ids)
        digit_images.extend(d_imgs)
        digit_labels.extend(d_labels)

    if cache:
        with open(training_data_pickle_file, "wb") as fd:
            pickle.dump([digit_images, digit_labels, digit_ids, skip_count], fd)
    print("Numbers skipped due to inaccuracies: {}, {:.2f}%".format(skip_count, (skip_count/float(limit))*100))

    return digit_images, digit_labels, digit_ids


def visualize_digits(digit_images, digit_labels):
    """visualize digit clippings"""
    data = list(zip(digit_images, digit_labels))
    np.random.shuffle(data)
    for i, axes in gen_subplot_group(25, cols=5, size=2.3, colsize=1.0, yield_group=2):
        for j, ax in enumerate(axes):
            # ax.axis('off')
            ax.imshow(data[i*2+j][0], cmap="gray")
            ax.set_xlabel(data[i*2+j][1], fontsize=20)


def train_digit_classifier(images_gray, labels):
    """routine to train digit classifier given grayscale images and labels"""
    digit_images, digit_labels, digit_ids = generate_digit_training_data(images_gray, labels)
    classifier = DigitClassifier()
    classifier.train(digit_images, digit_labels)
