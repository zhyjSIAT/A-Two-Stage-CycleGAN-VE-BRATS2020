from sklearn.metrics import roc_auc_score,precision_recall_curve, auc
import sklearn.metrics as metrics
from scipy.spatial.distance import directed_hausdorff
import numpy as np

def auc_roc(y_pred, y_true):
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    precision, recall, _ = precision_recall_curve(y_true_flat, y_pred_flat)
    auproc = roc_auc_score(y_true_flat, y_pred_flat)
    # Compute the AUPRC
    auprc = auc(recall, precision)
    return auprc

def iou(y_pred, y_true):
    intersection = (y_pred * y_true).sum()
    union = y_pred.sum() + y_true.sum() - intersection
    iou = intersection / union
    return iou.item()

def recall(y_pred, y_true):
    true_positives = (y_pred * y_true).sum()
    false_negatives = ((1 - y_pred) * y_true).sum()
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    return recall.item()

def precision(y_pred, y_true):
    true_positives = (y_pred * y_true).sum()
    false_positives = (y_pred * (1 - y_true)).sum()
    precision = true_positives / (true_positives + false_positives + 1e-7)
    return precision.item()

def f1_score(y_pred, y_true):
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)
    f1 = 2 * (p * r) / (p + r + 1e-7)
    return f1

def hausdorff_95(mask1, mask2):
    """
    Calculate the Hausdorff 95 metric between two binary segmentation masks.
    Returns the 95th percentile of the directed Hausdorff distances between mask1 and mask2.
    """
    # Find the edge coordinates of each mask
    mask1 = mask1.numpy()
    mask2 = mask2.numpy()
    edge1 = np.transpose(np.nonzero(mask1))
    edge2 = np.transpose(np.nonzero(mask2))
    # Calculate directed Hausdorff distances between the edge coordinates
    hausdorff_distances = []
    for edge in [edge1, edge2]:
        d = directed_hausdorff(edge, edge1)[0]
        hausdorff_distances.append(d)
    # Calculate the 95th percentile of the directed Hausdorff distances
    hausdorff_95 = np.percentile(hausdorff_distances, 95)
    return hausdorff_95


def dice_score(pred, targs):
    pred = (pred>0).float()
    dice = 2. * (pred*targs).sum() / (pred+targs).sum()
    return dice.numpy()