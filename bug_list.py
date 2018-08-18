"""
BUG LIST:

1. NOTE: 2-3% accuracy loss by increasing threshold from 0.5 -> 0.57
projection threshold: removes all digits instead of noise
# BUg:
# 1602300000_ae61d5.png
# i = 2509
# # # ERROR in third projection used for splitting
# horizontal noise filter removing all digits
# solution: 
# works awesome with process_projection_peaks(project, threshold=0.75)
# Y THRESH 0.56 disappear
# Y THRESH 0.57 works!

2. 
# BUG: nan float to int conversion: fix    
    peak_imp_l = []
    for x in peaks_in_boxes:
        cur = 0
        if x:
            cur = sum(list(zip(*x))[1])/1
            if not math.isnan(cur):
                  cur = int(cur)
        peak_imp_l.append(cur)
# prev bug source:
#     peak_imp_l = [(int(sum(list(zip(*x))[1])/1) if x else 0) for x in peaks_in_boxes]

3.
max image width:
318 33
2nd highest width: 70 33
(multiple images merged)

recursive split required


without box merge/split routines:
    99.75% error rate
    processing time: 2-3s / 6000 images

with box merge/split routines:
    21.5% error rate
    processing time: 18.8s / 6000 images


"""


""" Classification Reports:

training done. 167.11262702941895s
[7 7 0 2 1]
Classification report for classifier LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False):
             precision    recall  f1-score   support

          0       0.85      0.88      0.87       150
          1       0.77      0.84      0.80       147
          2       0.76      0.73      0.75        83
          3       0.72      0.77      0.74        56
          4       0.48      0.50      0.49        44
          5       0.59      0.55      0.57        42
          6       0.78      0.77      0.77        99
          7       0.75      0.56      0.64        43
          8       0.48      0.50      0.49        46
          9       0.85      0.70      0.77        47

avg / total       0.74      0.74      0.74       757


Confusion matrix:
[[132   4   1   2   3   2   3   1   2   0]
 [  7 123   2   3   3   1   1   2   3   2]
 [  1   5  61   5   1   3   3   0   3   1]
 [  1   3   2  43   2   2   0   0   3   0]
 [  4   3   0   0  22   0   6   3   4   2]
 [  3   4   1   1   2  23   4   1   3   0]
 [  5   1   3   1   6   5  76   0   2   0]
 [  1   1   4   4   3   1   1  24   3   1]
 [  1  12   6   1   0   1   2   0  23   0]
 [  0   4   0   0   4   1   2   1   2  33]]


PCA transformation done. 10.36s
PCA(copy=True, iterated_power='auto', n_components=200, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
training done. 400.68s
[3 0 1 1 6]
Classification report for classifier LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False):
             precision    recall  f1-score   support

          0       0.88      0.91      0.90       928
          1       0.83      0.90      0.87       863
          2       0.82      0.83      0.82       515
          3       0.85      0.84      0.85       350
          4       0.81      0.75      0.78       278
          5       0.84      0.79      0.82       267
          6       0.87      0.89      0.88       601
          7       0.80      0.77      0.78       262
          8       0.78      0.58      0.66       267
          9       0.83      0.81      0.82       286

avg / total       0.84      0.84      0.84      4617


Confusion matrix:
[[848  25  12   1   7   8  18   2   5   2]
 [ 13 778  15   6   8   6   5  11   6  15]
 [ 19  23 425   8   1   2  15   9  13   0]
 [ 10  13  14 295   2   5   0   7   3   1]
 [ 20   6   1   2 208   6  13   4   4  14]
 [ 10  12   5  11   2 211  13   2   1   0]
 [ 18   9  17   1  11   3 537   3   1   1]
 [  3  12  10   7   6   5   4 202   2  11]
 [ 14  39  21  10   4   2  12   8 154   3]
 [  8  17   1   5   7   2   1   5   9 231]]


PCA transformation done. 6.23s
PCA(copy=True, iterated_power='auto', n_components=100, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
training done. 169.52s
[6 6 7 4 4]
Classification report for classifier LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False):
             precision    recall  f1-score   support

          0       0.87      0.92      0.90       928
          1       0.79      0.89      0.84       863
          2       0.82      0.81      0.81       515
          3       0.85      0.80      0.82       350
          4       0.78      0.75      0.76       278
          5       0.84      0.75      0.79       267
          6       0.88      0.90      0.89       601
          7       0.84      0.72      0.77       262
          8       0.74      0.62      0.67       267
          9       0.86      0.81      0.83       286

avg / total       0.83      0.83      0.83      4617


Confusion matrix:
[[853  27   7   1  11   5  16   1   7   0]
 [ 12 767  18   8   8   6   2   7  18  17]
 [ 16  35 417   7   2   6  13   8  10   1]
 [ 20  19  11 280   7   6   2   3   1   1]
 [ 17  10   1   0 208   3  17   5   5  12]
 [ 14  11  12   9   5 199  10   1   6   0]
 [ 21  10  13   1   6   7 538   0   4   1]
 [  4  18  14  12  10   1   7 189   3   4]
 [ 11  50  17   9   3   2   4   3 165   3]
 [  8  21   1   3   6   1   0   9   5 232]]





"""