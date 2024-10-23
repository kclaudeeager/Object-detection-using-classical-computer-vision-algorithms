from .flann_matcher import match as flann_match, filter_matches as flann_filter_matches
from .brute_force_matcher import match as bf_match, filter_matches as bf_filter_matches

matchers = {
    'flann': (flann_match, flann_filter_matches),
    'brute_force': (bf_match, bf_filter_matches)
}