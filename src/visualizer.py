from datetime import datetime
import cv2

def draw_matches(img1, kp1, img2, kp2, matches,label, color=None):
    matched_image=cv2.drawMatches(img1, kp1, img2, kp2, matches, None, matchColor=color, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matched_image_rgb = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)
    # create and save that image the unique image
    right_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_name = f"matche_{label}@{right_now}.jpg"
    #cv2.imwrite(f'assets/{image_name}', matched_image_rgb)
    return matched_image_rgb
    
    
# def draw_bounding_box(frame, location, size, label, color=(0, 255, 0)):
#     x, y = location
#     w, h = size
#     cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#     cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
#     cv2.imshow('Frame', frame)
#     # create and save that image
#     cv2.imwrite('assets/output.jpg', frame)
    

def draw_bounding_box(image, bbox, label):
    cv2.rectangle(image, bbox[0], bbox[1], (0, 255, 0), 2)
    cv2.putText(image, label, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('image', image)
    