import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment


@jit(nopython=True)
def iou(bb_test, bb_gt):
   
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
   
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)

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
