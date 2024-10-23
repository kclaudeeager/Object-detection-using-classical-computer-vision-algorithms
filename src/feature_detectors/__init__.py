from .sift import detect_and_compute as sift_detect_and_compute, create_mask as sift_create_mask
from .surf import detect_and_compute as surf_detect_and_compute, create_mask as surf_create_mask
from .orb import detect_and_compute as orb_detect_and_compute, create_mask as orb_create_mask
from .template_matching import detect_and_compute as template_detect_and_compute, create_mask as template_create_mask

feature_detectors = {
    'sift': (sift_detect_and_compute, sift_create_mask),
    'surf': (surf_detect_and_compute, surf_create_mask),
    'orb': (orb_detect_and_compute, orb_create_mask),
    'template': (template_detect_and_compute, template_create_mask)
}