import cv2
import numpy as np
import yaml
import argparse
from src.input_handler import InputHandler
from src.feature_detectors import sift, surf, orb, template_matching
from src.matchers import flann_matcher, brute_force_matcher
from src.change_detector import *
from src.visualizer import draw_matches, draw_bounding_box

def main(config):
    # Load object images
    object_images = [cv2.imread(img_path) for img_path in config['object_images']]
    object_names = config['object_names']

    # Select feature detector
    if config['feature_detector'] == 'sift':
        detector = sift
    elif config['feature_detector'] == 'surf':
        detector = surf
    elif config['feature_detector'] == 'orb':
        detector = orb
    else:
        detector = template_matching

    # Select matcher
    matcher = flann_matcher if config['matcher'] == 'flann' else brute_force_matcher
    object_features = [detector.detect_and_compute(img) for img in object_images] if detector != template_matching else [img for img in object_images]

    with InputHandler(config['input_source']) as input_handler:
        prev_frame = input_handler.get_frame()
        if prev_frame is None:
            print("Failed to read from input source")
            return

        frame_count = 0
        while True:
            curr_frame = input_handler.get_frame()
            if curr_frame is None:
                break

            frame_count += 1
            if frame_count % config['sampling_rate'] != 0:
                continue

            for i, obj_feature in enumerate(object_features):
                if detector == template_matching:
                    # Template matching
                    template = obj_feature
                    top_left, max_val= detector.detect_and_compute(curr_frame, template)
                    h, w = template.shape[:2]
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    bbox = np.array([[top_left,bottom_right ]], dtype=np.int32)
                    change = detect_changes(template, curr_frame, bbox, detector)
                    print(f"{object_names[i]}: {change:.2f}")

                    # Visualize
                    draw_bounding_box(curr_frame, (bbox[0][0], bbox[0][1]), f"{object_names[i]}: {change:.2f}")
                    if change > config['change_threshold']:
                        print(f"Detected change for {object_names[i]} about {change:.2f}")

                    if config['show_matches']:
                        match_img = draw_matches(template, None, curr_frame, None, [], f"change_{change:.2f}")
                        cv2.imshow('Matches', match_img)
                else:
                    # Feature matching
                    obj_kp, obj_des = obj_feature
                    frame_kp, frame_des = detector.detect_and_compute(curr_frame)
                    
                    if config['feature_detector'] != 'sift':
                        # Convert descriptors to CV_32F
                        if obj_des is not None and obj_des.dtype != np.float32:
                            obj_des = obj_des.astype(np.float32)
                        if frame_des is not None and frame_des.dtype != np.float32:
                            frame_des = frame_des.astype(np.float32)

                    # Match features
                    matches = matcher.match(obj_des, frame_des)
                    good_matches = matcher.filter_matches(matches, ratio=config['match_ratio'])
                    print(f"Object {i}: {len(good_matches)} matches")
                    if len(good_matches) > config['min_matches']:
                        # Compute homography
                        src_pts = np.float32([obj_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                        if M is not None:
                            # Get bounding box
                            h, w = object_images[i].shape[:2]
                            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M)
                            bbox = np.int32(dst)

                            # Detect changes
                            obj_mask = detector.create_mask(object_images[i])
                            change = detect_changes(object_images[i], curr_frame, bbox, detector)
                            print(f"{object_names[i]}: {change:.2f}")

                            # Visualize
                            draw_bounding_box(curr_frame, (bbox[0][0], bbox[2][0]), f"{object_names[i]}: {change:.2f}")
                            if change > config['change_threshold']:
                                print(f"Detected change for {object_names[i]} about {change:.2f}")

                            if config['show_matches']:
                                match_img = draw_matches(object_images[i], obj_kp, curr_frame, frame_kp, good_matches, f"change_{change:.2f}")
                                cv2.imshow('Matches', match_img)

            # cv2.imshow('Frame', curr_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            prev_frame = curr_frame.copy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scene Change Detection")
    parser.add_argument("--config", default='config.yaml', help="Path to configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)