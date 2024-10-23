import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_compute(image, template, method=cv2.TM_CCOEFF_NORMED):
    assert image is not None, "Image is None"
    assert template is not None, "Template is None"
    # cv2.imshow('template', template)
    print(image.shape, template.shape)
    # Convert to grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(template.shape) > 2:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    print(image.shape, template.shape)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(image, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc

    return top_left, max_val

def create_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask