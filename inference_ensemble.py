
import cv2
import os
import shutil
import numpy as np


### custom inference results path
def ensemble(path1=None, path2=None, path3=None, outputpath=None):

    SR_list_1 = os.listdir(path1)

    if os.path.exists(outputpath) and os.path.isdir(outputpath):
        shutil.rmtree(outputpath)
    try:
        os.mkdir(outputpath)
    except OSError:
        print("Creation of the output directory '%s' failed " % outputpath)

    for idx in range(0, len(SR_list_1), 2):
        SR_left_1 = cv2.imread(path1 + SR_list_1[idx]).astype(np.float64)
        SR_left_2 = cv2.imread(path2 + SR_list_1[idx]).astype(np.float64)
        SR_left_3 = cv2.imread(path3 + SR_list_1[idx]).astype(np.float64)


        SR_right_1 = cv2.imread(path1 + SR_list_1[idx + 1]).astype(np.float64)
        SR_right_2 = cv2.imread(path2 + SR_list_1[idx + 1]).astype(np.float64)
        SR_right_3 = cv2.imread(path3 + SR_list_1[idx + 1]).astype(np.float64)

        SR_left = np.array((SR_left_1 + SR_left_2 + SR_left_3) / 3., dtype='float64')
        SR_right = np.array((SR_right_1 + SR_right_2 + SR_right_3) / 3., dtype='float64')

        cv2.imwrite(outputpath + SR_list_1[idx], SR_left)
        cv2.imwrite(outputpath + SR_list_1[idx + 1], SR_right)

if __name__ == '__main__':

    print("Start ensemble...")
    ensemble(path1=r'./results/perceptual_loss/',
                path2=r'./results/scam_ffc_TTA_Mean_03_14_23.7060/Flickr1024/',
                path3=r'./results/slowftcam/',
                outputpath=r'./results_final/')
    print("Finish ensemble...")