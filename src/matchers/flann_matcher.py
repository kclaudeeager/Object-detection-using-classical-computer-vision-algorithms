import cv2
import numpy as np

def match(des1, des2, k=2):
    '''
    Match features using FLANN matcher
    :param des1: descriptors of the first image
    :param des2: descriptors of the second image
    :param k: number of nearest neighbors
    :return: list of matches
    '''
    # Debugging: Print sizes of descriptor sets
    print(f"des1 size: {des1.shape if des1 is not None else 'None'}, des2 size: {des2.shape if des2 is not None else 'None'}")

    if des1 is None or des2 is None or len(des1) < k or len(des2) < k:
        print("Not enough descriptors to perform knnMatch")
        return []

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=k)
    return matches

def filter_matches(matches, ratio=0.7):
    '''
    Filter matches using Lowe's ratio test
    :param matches: list of matches
    :param ratio: Lowe's ratio
    :return: list of good matches
    '''
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches