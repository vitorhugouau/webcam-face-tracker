"""
SORT-like data association module with Hungarian algorithm (linear assignment)
Computes IoU between detected boxes and tracker predictions, then matches them.
"""

import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment


@jit(nopython=True)
def iou(bb_test, bb_gt):
    """
    Computes IoU between two bboxes in the form [x1, y1, x2, y2]
    """
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.25):
    """
    Assigns detections to tracked objects (both as bounding boxes).

    Returns:
        matches: array of shape (N, 2) with detection index and tracker index
        unmatched_detections: array of detection indices not matched
        unmatched_trackers: array of tracker indices not matched
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

    # Hungarian algorithm (maximize IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_trackers = list(range(len(trackers)))

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append([r, c])
            unmatched_detections.remove(r)
            unmatched_trackers.remove(c)

    if len(matches) > 0:
        matches = np.array(matches, dtype=int)
    else:
        matches = np.empty((0, 2), dtype=int)

    return matches, np.array(unmatched_detections, dtype=int), np.array(unmatched_trackers, dtype=int)
