from fileinput import filename
import os 
import cv2
import numpy as np


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten() // 255
    y_pred_f = y_pred.flatten() // 255
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_average_sets(y_true, y_pred):
    dice = []
    for i in range(len(y_true)):
        dice.append(dice_coef(y_true[i], y_pred[i]))
    print(np.mean(dice))
    return dice

true = []
pred = []
filename = []
for file in os.listdir("./test/merge23label/"):
    t = cv2.imread("./test/merge23label/" + file)
    p = cv2.imread("./test/merge23result/" + file)
    filename.append(file)
    true.append(t)
    pred.append(p)
# print(filename)
print(dice_average_sets(true, pred))
