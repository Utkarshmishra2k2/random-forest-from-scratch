"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
import numpy as np

def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    if len(labels) == 0:
        return 0.0

    counts = np.bincount(labels)
    probabilities = counts / len(labels)
    return 1 - sum(probabilities ** 2)

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    left = features[:,feature_index] <= threshold
    right = features[:,feature_index] > threshold

    # : → take all rows
    # feature_index → take that column

    return features[left],labels[left],features[right],labels[right]

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    total = len(parent_labels)
    left_weight = len(left_labels) / total
    right_weight = len(right_labels) / total
    children_impurity = (
        left_weight * impurity(left_labels)
        + 
        right_weight * impurity(right_labels)
    )

    return impurity(parent_labels) - children_impurity

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices, *args):
    best = {
        "feature_index": None,
        "threshold": None,
        "score": 0.0
    }

    for feature_index in feature_indices:
        thresholds = np.unique(features[:, feature_index])

        for threshold in thresholds:
            left_features, left_labels, right_features, right_labels = split_dataset(
                features, labels, feature_index, threshold
            )

            if len(left_labels) == 0 or len(right_labels) == 0:
                continue

            score = split_score(labels, left_labels, right_labels)

            if score > best["score"]:
                best["feature_index"] = feature_index
                best["threshold"] = threshold
                best["score"] = score

    return best

# Step 5 - should_stop (not yet solved)
# TODO: implement

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

