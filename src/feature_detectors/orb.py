import cv2
import numpy as np

def detect_and_compute(image, mask=None):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, mask)
    return keypoints, descriptors

def create_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask