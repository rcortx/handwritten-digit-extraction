# ALPHA = True
# AREA_FILTERS = True
# AREA_FILTERS_2 = True

# # TODO: pipeline image transformations and remove repetetive code
# bb_heights = []
# bb_widths = []
# for i, axes in gen_subplot_group(30*2, size=1.0, yield_group=2):
#     print("i: {}".format(i))
#     ax1, ax2 = axes
#     img_g = images_gray[i]
#     ret, thresh = cv2.threshold(img_g, 200, 255,cv2.THRESH_BINARY_INV)
#     # Gaussian filter will remove noise, kernel size has been optimized by trial and error
#     # the x projection has much higher noise fluctuations and is more affected
#     # by kernel choices
#     blur_img = cv2.GaussianBlur(thresh, (3, 3), 0)
    
#     ###
#     # y axis projection
#     ###
    
#     project = np.sum(blur_img, axis=1)
#     peaks, peak_widths_l, peak_base_heights, starts, ends = process_projection_peaks(project)
    
#     mx_w_ind = np.argmax(peak_widths_l)
#     st, en = starts[mx_w_ind], ends[mx_w_ind]
    
#     masked = np.copy(thresh)
#     masked[:st, :] = 0
#     masked[en:, :] = 0
    
#     ###
#     # x axis projection
#     ###
    
#     project = np.sum(blur_img, axis=0)
#     peaks, peak_widths_l, peak_base_heights, starts, ends = process_projection_peaks(project, threshold=0.65)
#     mean_width = np.mean(peak_widths_l)
#     mean_height = np.mean(list(map(lambda x: project[x], peaks)))
#     print("height: ", mean_height)
#     peak_stats = list(zip(peaks, peak_widths_l, starts, ends))
#     peak_stats.sort(key=lambda x: x[2] * 10000 + x[3])
#     # high peak candidates are more important to be removed
#     # apply peak threshold
#     rem_cand = peak_stats[:5] + peak_stats[-5:] # only take edge candidates for removal
#     rem_cand.sort(key=lambda x: x[1]) # sort candidates by width candidates
# #     rem_cand = rem_cand[:2] # take first two least width candidates
# #     print("shape", img_g.shape[1] / 2)
# #     print("mean width: ", mean_width)
#     for cand in rem_cand:
#         if labels[i] == "1300105195":
#             print("mean widht: ", mean_width, ", mean height: ", mean_height)
#             print("cand: ", cand, ", height: ", project[cand[0]])
#         # if candidate width is less than half of mean_width
#         # remove candidate from image
#         # TODO: hyperparameters
#         if 0 <cand[1] < 0.6 * mean_width and mean_height * 1.5 < project[cand[0]]:
#             # if candidate is on left of image
#             if cand[-1] < (img_g.shape[1] / 2):
#                 masked[:, cand[-2]:cand[-1]] = 0
#             # else candidate is on right of image
#             else: 
#                 # TODO: merge filters
#                 masked[:, cand[-2]:cand[-1]] = 0
    
#     ###
#     # Bounding box detection routines
#     ###
    
#     clone = masked.copy()
#     clone2 = masked.copy()
#     clone_masked = masked.copy()
#     im2, contours, hierarchy = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     b_widths = []
#     b_heights = []
#     boxes = []
#     for (k, c) in enumerate(contours):
#         # compute the bounding box of the contour
#         (x, y, w, h) = cv2.boundingRect(c)
#         b_widths.append(w)
#         b_heights.append(h)
#         # only accept the contour region as a grouping of characters if
#         # the ROI is sufficiently large
#         if w >= 2 and h >= 2:
#             boxes.append((x, y, w, h))
#             cv2.rectangle(clone, (x,y), (x+w, y+h), (255,0,0), 1)
    
#     # ALPHA TEST
#     bbs = list(map(MergedBox, boxes))
#     bbs = list(map(lambda x: x.box.box, box_merge_routines(bbs)))
#     # ALPHA TEST ENDS
    
#     # sort boxes by start x
#     boxes.sort(key=lambda x: x[0])
    
#     ###
#     # Box util func: deprecated 
#     ###
    
#     def merge_boxes(box1, box2):
#         """merge boxes by taking max/min bounds to encompass largest area"""
#         x1, y1, w1, h1 = box1
#         xe1 = x1+w1
#         ye1 = y1+h1
        
#         x2, y2, w2, h2 = box2
#         xe2 = x2+w2
#         ye2 = y2+h2
#         xn, yn, xen, yen = min(x1, x2), min(y1, y2), max(xe1, xe2), max(ye1, ye2)
#         return xn, yn, xen-xn, yen-yn
    
#     def box_area(box):
#         """width * height"""
#         return box[-1] * box[-2]
    
#     box_filtered = clone2
    
    
#     ###
#     # Box merge routines: overlap threshold
#     ###
    
#     bound_thresh = 0 # virtually, boundary extends to this many pixels to consider overlap
#     w_thresh = 0.5 # threshold of width overlap after which boxes are merged
#     # 0.34 worked well!
    
#     for b in range(1, len(boxes)): 
#         xp, yp, wp, hp = boxes[b-1]
#         x, y, w, h = boxes[b]
        
#         if xp + wp + bound_thresh > x:
#             # overlap in x detected
#             overlap_width = xp + wp + bound_thresh - x
#             if overlap_width / wp > w_thresh or overlap_width / w > w_thresh:                
#                 boxes[b] = merge_boxes(boxes[b-1], boxes[b])
#                 boxes[b-1] = None
    
    
    
#     boxes = list(filter(lambda x: x, boxes))
#     boxes.sort(key=lambda x: x[0])
    
#     ###
#     # Box merge routines: area Gauss filter and best adjacent merge
#     ###
    
#     if AREA_FILTERS:
#     # TODO: check area, if less than 30% of mean area, merge
#     # taking top 75% boxes by area for robust mean and std calculation
#     # TODO: expensive operations, check viability
#         top_75_boxes = sorted(list(map(box_area, boxes)))#[int(len(boxes)*0.1):]
#         box_area_mean = np.mean(top_75_boxes)
#         box_area_std = np.std(top_75_boxes)
#         area_thresh = (box_area_mean - box_area_std)
        
#         for b in range(1, len(boxes)):
#             box1ar = box_area(boxes[b])
#             box2ar = box_area(boxes[b-1])
#             if area_thresh >= box1ar or area_thresh >= box2ar:
#                 if labels[i] == "1601123503":
#                     print("threshold: ", area_thresh, box_area_mean, box_area_std, min(box1ar, box2ar))
#             if area_thresh >= box1ar or area_thresh >= box2ar:                
#                 overlap_prev = boxes[b-1][0] + boxes[b-1][2] - boxes[b][0]
#                 overlap_nex = None
#                 if b < len(boxes)-1:
#                     overlap_nex = boxes[b][0] + boxes[b][2] - boxes[b+1][0]
#                 if (overlap_nex and overlap_nex > overlap_prev):
#                     continue
#                 if  boxes[b-1][0] + boxes[b-1][2] + 2 > boxes[b][0]:
#                     boxes[b] = merge_boxes(boxes[b-1], boxes[b])
#                     boxes[b-1] = None

#         boxes = list(filter(lambda x: x, boxes))

#         ###
#         # Box merge routines: area Gauss filter and best adjacent merge (duplicated??)
#         ###
    
#     if AREA_FILTERS_2:
#         for b in range(len(boxes)):
#             boxar = box_area(boxes[b])
#             if area_thresh - box_area_std*0.75 >= boxar:
#     #             if labels[i] == "1601123503":
#     #                 print("threshold: ", area_thresh, box_area_mean, box_area_std, min(box1ar, box2ar))              
#                 overlap_prev = None
#                 if b > 1:
#                     overlap_prev = boxes[b-1][0] + boxes[b-1][2] - boxes[b][0]
#                 overlap_nex = None
#                 if b < len(boxes)-1:
#                     overlap_nex = boxes[b][0] + boxes[b][2] - boxes[b+1][0]
#                 if (overlap_nex and overlap_prev): 
#                     if (overlap_nex > overlap_prev):
#                         overlap_prev = None
#                     else: overlap_nex = None

#                 if  overlap_nex:
#                     # TODO: try merging b+1 with b-1?
#                     boxes[b+1] = merge_boxes(boxes[b+1], boxes[b])
#                     boxes[b] = None
#                 else:
#                     boxes[b] = merge_boxes(boxes[b-1], boxes[b])
#                     boxes[b-1] = None
#         boxes = list(filter(lambda x: x, boxes))
    
#     ###
#     # Drawing Box routine
#     ###
               
    
#     if ALPHA:
#     # ALPHA TEST
#         for box in bbs:
#             (x, y, w, h) = box
#             cv2.rectangle(box_filtered, (x,y), (x+w, y+h), (255, 0, 0), 1)
#     # ALPHA TEST ENDS
#     else:
#         for box in boxes:
#             (x, y, w, h) = box
#             cv2.rectangle(box_filtered, (x,y), (x+w, y+h), (255, 0, 0), 1)
               
    
#     ###
#     # Box SPLIT routines: canny peaks counts/importance, box width Gauss filter
#     ###
    
#     # test: detect multiple peaks in conjoined images:
#     blurred_clone = cv2.GaussianBlur(clone_masked, (3,3), 0)
#     canny_res = cv2.Canny(blurred_clone, 300, 900)
#     project = np.sum(canny_res, axis=0) # clone_masked
#     peaks, peak_widths_l, peak_base_heights, starts, ends = process_projection_peaks(project, threshold=0.75)
#     # TODO: machine learning n_peaks -> dual digits
#     peaks = sorted(peaks)
#     peaks_in_boxes = []
#     wasted_peaks = []
#     p_i = 0
#     boxes.sort(key=lambda x: x[0])
#     b_widths = list(map(lambda x: x[2], boxes))
#     w_mean, w_std = np.mean(b_widths), np.std(b_widths)
#     for box in boxes:
#         peaks_in_boxes.append([])
#         (x, y, w, h) = box
#         w_z_score = (w-w_mean) / w_std
#         if w_z_score < 0:
#             w_z_score = 0
#         while p_i < len(peaks) and peaks[p_i] <= x+w:
#             # check apply higher threshold? use higher gaussian kernel?
#             peak_imp_score = w_z_score*10#peak_widths_l[p_i] * (project[peaks[p_i]] - peak_base_heights[p_i])
#             if x <= peaks[p_i] <= x+w:
#                 peaks_in_boxes[-1].append((peaks[p_i], peak_imp_score))
#             else:
#                 wasted_peaks.append(peaks[p_i])
#             p_i += 1
    
#     while p_i < len(peaks):
#         wasted_peaks.append(peaks[p_i])
#         p_i += 1
    
    
#     ###
#     # Visualizations
#     ###
    
    
#     peak_counts = [len(x) for x in peaks_in_boxes]
#     peak_imp = [(int(sum(list(zip(*x))[1])/1) if x else 0) for x in peaks_in_boxes]
#     peak_counts = " ".join(map(lambda x, y: "|{}:{}".format(x, y), peak_counts, peak_imp))
    
#     bb_widths.append(b_widths)
#     bb_heights.append(b_heights)
    
# #     ax1.imshow(clone, cmap="gray")
# #     ax1.set_xlabel(labels[i] + " : {}".format(len(labels[i])))
# #     ax1.set_ylabel("boxes ori")
    
#     ax1.imshow(box_filtered, cmap="gray")
#     ax1.set_xlabel(labels[i] + " : {}".format(len(labels[i])) + " : " + peak_counts + " w: {}".format(len(wasted_peaks)))
    
#     canny_res = cv2.Canny(clone_masked, 300, 900)
#     ax2.imshow(clone, cmap="gray") # clone, thresh
#     ax2.set_xlabel(labels[i] + " : {}".format(len(labels[i])))
# #     ax2.set_ylabel("box filtered")
#     # any box with 30% or higher overlap in vertical is merged
#     # merge smaller below threshold with closest larger
    
    
    
    
    
    


import numpy as np
import ipdb


class Box(object):

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
    """MergedBox class to facilitate merging and holding multiple bounding box coordinates
    
    heirarchial structure: may hold multiple Box/MergedBox objects
    
    class design has been focused on seamless interoperability beween Box and MergedBox 
        instances in terms of add/merge, area and overlap computations
    """
    
    
    OVERLAP_X_THRESH = 0.5 # fraction of width overlap required to consider merging
    
    def __str__(self):
        return "MergedBox(x:{}, y:{}, w:{}, h:{})".format(*self.box.box)

    def __repr__(self):
        return self.__str__()
    
    def __lt__(self, other):
        """for sorting and box comparisions, comparing by x coordinate"""
        return self.box < other.box

    # def __getitem__(self, key):
    #     return self.boxes[key]

    # def __iter__(self):
    #     return self.boxes

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
        """boxes can be MergedBox/Box instance, list of Box/MergedBox instances, list of tuples of box coordinates, or single tuple of box coordinates"""
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
        if type(b_obj) in MergedBox:
            box = b_obj
        elif type(b_obj) is Box:
            boxes = [b_obj, ]
        else:
            # assuming b_obj is a tuple of box coordinates
            boxes = [Box(b_obj), ]
        return boxes

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
        for i in range(1, len(self.boxes)):
            cur = cur.merge(self.boxes[i])
        return cur

    def merge(self, b_obj):
        # Note: destroys constituent boxes!
        # TODO: which .box? parent vs child
        return self + b_obj
    
    def varea(self):
        """virtual area -> of the superbox encompassing all constituent boxes, is an overestimation and may include noise"""
        return self.box.area()
    
    def area(self):
        """actual area of Box by adding constituent box areas"""
        return sum(map(lambda b: b.area(), self.boxes))
    
    def sort(self):
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
    
    def bifurcate(self, discard=False):
        """divides the box into two boxes by sensible overlap heuristic (minimize overlap?)"""
        pass
    
    def area_thresh(self, area_thresh):
        """applies area threshold on itself and returns Bool
            # TODO: figure out whether to do on superbox or true area?
        """
        pass
    
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
                flat.append(box.box)
        return flat
    
    def peaks(self):
        pass
    
    def box_encloses(self, x):
        pass

    def distance(self, other):
        """opposite of overlap: convenience function"""
        return -self.overlap(other)
    

boxes = [(21, 23, 3, 3), (93, 21, 7, 6), (77, 11, 12, 17), (112, 9, 16, 20), (98, 9, 11, 19), (58, 9, 12, 18), (60, 10, 9, 16), (154, 8, 13, 21), (39, 8, 13, 21), (42, 17, 9, 11), (187, 7, 17, 25), (188, 8, 15, 14), (132, 7, 16, 21), (141, 15, 3, 3), (170, 6, 17, 23), (176, 7, 10, 12), (26, 6, 9, 23)]
print(len(boxes))




"""
# TODO: check box filter post merging, prev boxes don't need to be nullified?
        optimize for identical boxes: same tuple by object ID
"""


# Logic
# overlap merge
# area filter merge target: ignore or merge (proximity/overlap threshold before merge/ignore)
#     merged noise: separate by distance
# keep top `n` by area?




def merge_routine(bbs, merge_func, passes=2):
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
    def merge_candidate(bbs, i, b, adjacency_window_size=1):
        lims = (i-adjacency_window_size, i+(adjacency_window_size if not predecessor_only else 0)+1)
        adjacents = [(j, bbs[j]) for j in range(*lims) \
                     if j != i and j>=0 and j<len(bbs) and bbs[j] is not None]

        # for j, b2 in enumerate(adjacents):
        #     pass
        
        if adjacents:
            # best overlap in adjacents
            j, a = max(adjacents, key=lambda x: b.overlap(x[1]))
            
            # TODO: comment architecture philosophy
            if thresh_func(b, a):
            #if b.can_merge(a):
                # pushing merged box in maximum index btw i, j so it may be referenced in subsequent adjacency iterations
                bbs[max(i, j)] = b + a
                bbs[min(i, j)] = None
    return merge_candidate





b = MergedBox([[1, 1, 2, 2], [4, 4, 1, 1], ])      
print(b.ioverlaps())

bbs = [b, ]

def box_merge_routines(bbs):
    thresh_func = lambda x, y: x.can_merge(y)
    merge_function = get_merge_func(thresh_func=thresh_func)
    bbs = merge_routine(bbs, merge_func=merge_function, passes=1)
    
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
    
box_merge_routines(bbs)





# ALPHA TEST
bbs = list(map(MergedBox, boxes))

def test_bb(bbs):
    bbs.sort()
    print("@check sorted order: ", bbs)
    for i, b in enumerate(bbs):
        # if not i:
        #     continue
        # prev = bbs[i-1]
        # if b.can_merge(prev):
        #     bbs[i] = b + prev
        #     bbs[i-1] = None
        # continue
        # TODO: incorporate into ALPHA testing
        adjacents = [(j, bbs[j]) for j in range(i-1, i+2) \
                         if j != i and j>=0 and j<len(bbs) and bbs[j] is not None]
        print(adjacents)
        if adjacents:
            # best overlap in adjacents
            j, a = max(adjacents, key=lambda x: b.overlap(x[1]))
            # if i < len(bbs):
            #     nx = bbs[i+1]
            #     if b.overlap(nx) >= 0:
            #         import ipdb
            #         ipdb.set_trace()

            # TODO: comment architecture philosophy
            if b.can_merge(a):
            #if b.can_merge(a):
                # pushing merged box in maximum index btw i, j so it may be referenced in subsequent adjacency iterations
                print("merging...")
                bbs[max(i, j)] = b + a
                bbs[min(i, j)] = None
    bbs = list(filter(lambda x: x, bbs))
    print("std test: ", len(bbs))       
    return bbs        
import pprint

pprint.pprint(list(zip(sorted(boxes), sorted(bbs))), indent=2)

bbs = test_bb(bbs)

# print("@PRE: ", bbs)
# bbs = list(map(lambda x: x.box.box, box_merge_routines(bbs)))
# # ALPHA TEST ENDS
# # print(boxes)
# # print(bbs)
# print(len(bbs))

res = [(21, 23, 3, 3), (26, 6, 9, 23), (39, 8, 13, 21), (58, 9, 12, 18), (77, 11, 12, 17), (93, 21, 7, 6), (98, 9, 11, 19), (112, 9, 16, 20), (132, 7, 16, 21), (154, 8, 13, 21), (170, 6, 17, 23), (187, 7, 17, 25)]
pprint.pprint(list(zip(bbs, res)), indent=2)


# class Box(object):
#     """Box class to facilitate merging and holding multiple bounding box coordinates"""
    
#     OVERLAP_X_EXTEND = 0 # pixels to virtually extend width to consider overlap
#     OVERLAP_X_THRESH = 0.5 # fraction of width overlap required to consider merging
    
#     @property
#     def box(self):
#         # optimized via caching results, check correctness, add tests
#         if self.cached:
#             return self._res
        
#         self._res = self.internal_merge()
#         self.cached = True
#         return self._res
    
#     @box.setter
#     def box(self, boxes):
#         # boxes has to be list of tuples of box conordinates
#         self.cached = False
#         self.boxes = boxes
        
#     def __str__(self):
#         return "Box(x:{}, y:{}, w:{}, h:{})".format(*self.box)
    
#     def __lt__(self, other):
#         """for sorting and box comparisions, comparing by x coordinate"""
#         return self.box[0] < other.box[0]
        
#     def __init__(self, boxes=[]):
#         """boxes has to be list of tuples of box coordinates"""
#         # TODO: box label encapsulation, update in init args, add, sub
#         # TODO: add support to hold peaks
#         if boxes and ((type(boxes[0]) is not tuple) and (type(boxes[0]) is not list)):
#             # if boxes is non empty and first element of boxes is non iterable, implies: boxes is single
#             # tuple of coordinates
#             boxes = [boxes, ]
#             # raise TypeError("argument `boxes` has to be list of tuples of box coordinates")
#         self.cached = False
#         self.boxes = boxes 
    
#     def __add__(self, b_obj):
#         """b_obj can be instance of Box or list of box coordinates"""
#         if type(b_obj) == Box:
#             boxes = b_obj.boxes
#         else:
#             # assuming b_obj is a tuple of box coordinates
#             boxes = [b_obj, ]
#         return Box(self.boxes + boxes)
    
#     def __sub__(self, b_obj):
#         """b_obj can be instance of Box or list of box coordinates"""
#         if type(b_obj) == Box:
#             boxes = b_obj.boxes
#         else:
#             # assuming b_obj is a tuple of box coordinates
#             boxes = [b_obj, ]
#         # assuming common box elements are same array objects
#         return Box(list(set(self.boxes).difference(boxes)))

    
#     def internal_merge(self):
#         # TODO: optimize by batch max/min
#         if not self.boxes:
#             return [0, 0, 0, 0]
#         cur = self.boxes[0]
#         for i in range(1, len(self.boxes)):
#             cur = Box.merge_boxes(cur, self.boxes[i])
#         return cur
    
#     def merge_boxes(box1, box2):
#         """merge boxes by taking max/min bounds to encompass largest area"""
#         x1, y1, w1, h1 = box1
#         xe1 = x1+w1
#         ye1 = y1+h1
#         x2, y2, w2, h2 = box2
#         xe2 = x2+w2
#         ye2 = y2+h2
#         xn, yn, xen, yen = min(x1, x2), min(y1, y2), max(xe1, xe2), max(ye1, ye2)
#         return xn, yn, xen-xn, yen-yn
    
#     def box_area(box):
#         return box[-1] * box[-2]
    
#     def varea(self):
#         """virtual area -> of the superbox encompassing all constituent boxes, is an overestimation and may include noise"""
#         return Box.box_area(self.box)
    
#     def area(self):
#         """actual area of Box by adding constituent box areas"""
#         return sum(map(Box.box_area, self.boxes))
    
#     def sort(self):
#         # sort all internal boxes as per `x` start
#         self.boxes.sort(key=lambda x: x[0])
    
#     def overlap_calc(b1, b2, thresh=True):
#         """horizontal overlap, negative implies there is distance between boxes"""
#         x1, y1, w1, h1 = b1
#         x2, y2, w2, h2 = b2
#         res = x1 + w1 + (Box.OVERLAP_X_EXTEND if thresh else 0) - x2
#         if x2 < x1:
#             # implies b2 is behind b1
#             return -res
#         return res
    
#     def overlap(self, b_obj, thresh=True):
#         """overlap between box objects"""
#         b1 = self.box
#         b2 = b_obj.box
#         return Box.overlap_calc(b1, b2, thresh)
    
#     def best_overlap(self, b1_obj, b2_obj, thresh=True):
#         """useful func for merging box with adjacent boxes in sorted list"""
#         if self.overlap(b1_obj, thresh) > self.overlap(b2_obj, thresh):
#             return b1_obj
#         return b2_obj
    
#     def ioverlaps(self):
#         """calculates array of overlaps between sorted internal boxes"""
#         self.sort()
#         overlaps = []
#         if not self.boxes:
#             return overlaps
#         bp = self.boxes[0]
#         for b in range(1, len(self.boxes)):
#             overlaps.append(Box.overlap_calc(bp, self.boxes[b]))
#             bp = self.boxes[b]
#         return overlaps
    
#     def bifurcate(self, discard=False):
#         """divides the box into two boxes by sensible overlap heuristic (minimize overlap?)"""
#         pass
    
#     def area_thresh(self, area_thresh):
#         """applies area threshold on itself and returns Bool
#             # TODO: figure out whether to do on superbox or true area?
            
#         """
#         pass
    
#     def can_merge(self, b_obj):
#         """returns whether merge possible with b_obj after applying overlap threshold"""
#         overlap = self.overlap(b_obj)
#         # if overlap > width threshold of current box or candidate box: perform merge
#         if overlap > self.box[2] * Box.OVERLAP_X_THRESH or overlap > b_obj.box[2] * Box.OVERLAP_X_THRESH:
#             return True
#         return False
    
#     def peaks(self):
#         pass
    
#     def box_encloses(self, x):
#         pass