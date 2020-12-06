import os
import sys

import cv2
import numpy as np
import argparse


def profile_method(img):
    '''
    Segmentation of line image to single characters. Based on horizontal profile.
    '''
    inverted = cv2.bitwise_not(img)
    inverted = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    print(inverted.shape)

    profile = inverted.sum(axis=0)
    print("Profile shape:", profile.shape)

    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    m = np.max(profile)
    print("Profile shape:", profile.shape)
    #     vis = np.zeros((100, profile.shape[0]))
    #     print (vis.shape)
    #
    #     print ("Max:", m)
    #
    #     for i in range(profile.shape[0]):
    #         #print (profile[i])
    #         h = int(float(profile[i]) / m * 100)
    #         #print (h)
    #         for j in range(100-h,100):
    #             vis[j,i] = 255

    #     cv2.imshow('image', vis)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # cv2.imwrite("profile.png", vis)

    # imgComposed = np.zeros((inverted.shape[0] + 100 + 2, inverted.shape[1]))
    # imgComposed[0:100,:] = vis
    # imgComposed[100 + 2 :,:] = img

    # cv2.imwrite("profile_visu.png", imgComposed)

    candidates = []
    threshold = 2
    state = 0  # 0 = gap
    lastGap = -1
    for i in range(profile.shape[0]):
        h = float(profile[i]) / m * 100
        if h <= threshold:  # gap
            # cv2.line(result, (i,0), (i, result.shape[0]), (0,255,0), 1)
            if state == 1:
                # print (lastGap, i)
                candidates.append((lastGap, i))
            lastGap = i
            state = 0

        else:
            state = 1

    for c in candidates:
        cv2.line(result, (c[0], 0), (c[0], result.shape[0]), (255, 0, 0), 1)
        cv2.line(result, (c[1], 0), (c[1], result.shape[0]), (255, 0, 0), 1)

    cv2.imshow('result', result)
    cv2.waitKey(0)

    return candidates


def profile_method_2(img):
    '''
    Segmentation of line image to single characters. Based on horizontal profile.
    '''
    inverted = cv2.bitwise_not(img)
    inverted = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    profile = inverted.sum(axis=1)
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    m = np.max(profile)

    #     vis = np.zeros((100, profile.shape[0]))
    #     print (vis.shape)
    #
    #     print ("Max:", m)
    #
    #     for i in range(profile.shape[0]):
    #         #print (profile[i])
    #         h = int(float(profile[i]) / m * 100)
    #         #print (h)
    #         for j in range(100-h,100):
    #             vis[j,i] = 255

    #     cv2.imshow('image', vis)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # cv2.imwrite("profile.png", vis)

    # imgComposed = np.zeros((inverted.shape[0] + 100 + 2, inverted.shape[1]))
    # imgComposed[0:100,:] = vis
    # imgComposed[100 + 2 :,:] = img

    # cv2.imwrite("profile_visu.png", imgComposed)

    candidates = []
    threshold = 5
    state = 0  # 0 = gap
    lastGap = -1
    for i in range(profile.shape[0]):
        h = float(profile[i]) / m * 100
        if h <= threshold:  # gap
            # cv2.line(result, (i,0), (i, result.shape[0]), (0,255,0), 1)
            if state == 1:
                # print (lastGap, i)
                candidates.append((lastGap, i))
            lastGap = i
            state = 0

        else:
            state = 1

    for c in candidates:
        cv2.line(result, (0, c[0]), (result.shape[1], c[0]), (255, 0, 0), 1)
        cv2.line(result, (0, c[1]), (result.shape[1], c[1]), (0, 0, 255), 1)

    cv2.imwrite('result/cv2_segment.jpg', result)

    return candidates


if __name__ == '__main__':
    img = cv2.imread('test_line_image/true_line/20201024234435.png', 0)
    profile_method_2(img)
