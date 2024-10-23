import cv2
import numpy as np

def match(des1, des2, k=2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=k)
    return matches

def filter_matches(matches, ratio=0.7):
    good_matches = []
    for match in matches:
        if len(match) < 2:
            continue  # Skip matches with fewer than 2 elements
        m, n = match
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches