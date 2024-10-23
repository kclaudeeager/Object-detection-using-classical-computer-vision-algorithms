import cv2
import os

class InputHandler:
    def __init__(self, source):
        self.source = source
        self.cap = None

    def __enter__(self):
        if self.source.isdigit():
            self.cap = cv2.VideoCapture(int(self.source))
        elif os.path.isfile(self.source):
            # Check if the file extension is a video format, case-insensitive
            _, ext = os.path.splitext(self.source)
            if ext.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    raise IOError(f"Failed to open video file: {self.source}")
            else:
                # Assume it's an image if not a recognized video format
                self.cap = cv2.imread(self.source)
                if self.cap is None:
                    raise IOError(f"Failed to load image: {self.source}")
        else:
            raise IOError(f"Invalid input source: {self.source}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.cap, cv2.VideoCapture):
            self.cap.release()

    def get_frame(self):
        if isinstance(self.cap, cv2.VideoCapture):
            ret, frame = self.cap.read()
            return frame if ret else None
        elif isinstance(self.cap, np.ndarray):  # For single images
            frame = self.cap.copy()
            self.cap = None  # Ensure we only return the image once
            return frame
        return None