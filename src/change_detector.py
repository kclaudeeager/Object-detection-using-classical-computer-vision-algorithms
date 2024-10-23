from datetime import datetime
import cv2
import numpy as np
from scipy.spatial.distance import cosine

from scipy.spatial.distance import hamming
from src.feature_detectors import template_matching


def detect_hamming_changes(original_obj, current_frame, bbox, detector,mask):
    # Crop the current frame to the object region using the bounding box
    x, y, w, h = cv2.boundingRect(bbox)
    cropped_frame = current_frame[y:y+h, x:x+w]
    
    if cropped_frame.size == 0:
        print("Failed to crop object from frame")
        return float('inf')  # Unable to crop object

    # Detect features in both original and current objects
    kp1, des1 = detector.detect_and_compute(original_obj, mask)
    kp2, des2 = detector.detect_and_compute(cropped_frame, mask)
    
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        print("Failed to compute features")
        return float('inf')  # Unable to compute features

    # Match features using BFMatcher with Hamming distance (suitable for binary descriptors like ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate change based on average distance of top matches
    num_good_matches = min(len(matches), 50)  # Consider top 50 matches or all if less than 50
    if num_good_matches == 0:
        return float('inf')  # No good matches found

    avg_distance = sum(match.distance for match in matches[:num_good_matches]) / num_good_matches

    # Normalize the distance to a 0-1 range (assuming max Hamming distance is 256 for 8-bit descriptors)
    normalized_change = avg_distance / 256

    # Calculate the ratio of good matches to total keypoints
    match_ratio = num_good_matches / min(len(kp1), len(kp2))

    # Combine normalized distance and match ratio for final change score
    change_score = (1 - match_ratio) * normalized_change

    return change_score


def detect_changes(original_obj, current_frame, bbox, detector):
    try:
        # Warp the mask to the current frame
        h, w = current_frame.shape[:2]
    
        # Crop the current frame to the object region using the bounding box
        current_obj = current_frame.copy()
        x, y, w, h = cv2.boundingRect(bbox)
        cropped_frame = current_obj[y:y+h, x:x+w]
        
        if cropped_frame.size == 0:
            print("Failed to crop object from frame")
            return float('inf')  # Unable to crop object

        # Save the cropped object
        right_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # cv2.imwrite(f'assets/cropped_object_{right_now}.jpg', cropped_frame)

        if detector == template_matching:
            # For template matching, directly compare the template with the cropped region
            # res = cv2.matchTemplate(cropped_frame, original_obj, cv2.TM_CCOEFF_NORMED)
            # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # similarity = max_val
            # change = 1 - similarity
            # print(f"Template Matching Similarity: {similarity:.2f}")
            # return change
            diff = cv2.absdiff(original_obj, cropped_frame)
            change = np.mean(diff) / 255
            print(f"template_matching Change: {change:.2f}")
            return change
        else:
            # Detect features in both original and current objects
            kp1, des1 = detector.detect_and_compute(original_obj, None)
            kp2, des2 = detector.detect_and_compute(cropped_frame, None)
            
            if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
                print("Failed to compute features")
                return float('inf')  # Unable to compute features
            
            # Compute the mean descriptor for each object
            mean_des1 = np.mean(des1, axis=0)
            mean_des2 = np.mean(des2, axis=0)
            
            # Compute the cosine similarity between the mean descriptors
            similarity = 1 - cosine(mean_des1, mean_des2)
            # Compute the euclidean distance between the mean descriptors
            euclidean_distance = np.linalg.norm(mean_des1 - mean_des2)
            # Compute the hamming distance between the mean descriptors
            hamming_distance = hamming(mean_des1, mean_des2)
            
            print(f"Cosine Similarity: {similarity:.2f}")
            print(f"Euclidean Distance: {euclidean_distance:.2f}")
            print(f"Hamming Distance: {hamming_distance:.2f}")
            
            # Convert similarity to change (0 similarity = 1 change, 1 similarity = 0 change)
            change = 1 - similarity
            
            return change
    except Exception as e:
        print(f"Error in detect_changes: {e}")
        return float('inf')


def detect_changes_from_features(obj_des, frame_des, good_matches, mask):
    if len(good_matches) == 0:
        return 1.0  # Maximum change if no matches

    # Ensure mask is binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Extract the descriptors for the good matches
    obj_matched_des = np.array([obj_des[m.queryIdx] for m in good_matches])
    frame_matched_des = np.array([frame_des[m.trainIdx] for m in good_matches])

    # Calculate the mean descriptor for each set
    obj_mean_des = np.mean(obj_matched_des, axis=0)
    frame_mean_des = np.mean(frame_matched_des, axis=0)

    # Calculate cosine similarity
    similarity = 1 - cosine(obj_mean_des, frame_mean_des)

    # Convert similarity to change (0 similarity = 1 change, 1 similarity = 0 change)
    change = 1 - similarity

    # Apply mask to change value
    masked_change = change * (np.sum(mask) / (mask.size + 1e-6))  # Normalize by mask size

    # Ensure masked change is between 0 and 1
    masked_change = np.clip(masked_change, 0, 1)

    return masked_change