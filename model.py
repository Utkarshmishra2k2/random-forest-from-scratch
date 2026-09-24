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

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...
    if len(set(labels)) == 1:
        return True
    
    if depth >= max_depth:
        return True

    if len(labels) < min_samples_split:
        return True
    
    return False

# Step 6 - leaf_prediction
def leaf_prediction(labels):
    # TODO: choose a single class label to output for a leaf given the labels that reached it
    counts = np.bincount(labels)
    return int(np.argmax(counts))

    # np.bincount(labels) counts each class, and 
    # np.argmax(counts) returns the class with the 
    # highest count. int() ensures the output is a 
    # normal Python integer.

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    # TODO: recursively grow a decision tree, returning a nested dict of leaf/internal nodes.
    if should_stop(labels,depth,max_depth,min_samples_split):
        return {
            "leaf":True,
            "prediction":leaf_prediction(labels)
        }

    if feature_subset is None:
        feature_subset = range(features.shape[1])
        
    split = best_split(features,labels,feature_subset)

    if split["feature_index"] is None or split["score"] <= 0:
        return {
            "leaf":True,
            "prediction":leaf_prediction(labels)
        }
    feature_index = split["feature_index"]
    threshold = split["threshold"]

    left_features, left_labels, right_features, right_labels = split_dataset(
        features,
        labels,
        feature_index,
        threshold
        )

    left_tree = build_tree(
        left_features,
        left_labels,
        max_depth,
        min_samples_split,
        feature_subset,
        depth + 1
        )


    right_tree = build_tree(
        right_features,
        right_labels,
        max_depth,
        min_samples_split,
        feature_subset,
        depth + 1
        )

    return {
        "leaf": False,
        "feature_index": int(feature_index),
        "threshold": float(threshold),
        "left": left_tree,
        "right": right_tree
        }

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    # TODO: walk the example down the fitted tree until you reach a leaf, then return its prediction.
   # If it is a leaf, return its prediction
    if tree["leaf"]:
        return int(tree["prediction"])

    feature_index = tree["feature_index"]
    threshold = tree["threshold"]

    # Follow the left branch
    if example[feature_index] <= threshold:
        return predict_example_tree(tree["left"], example)

    # Follow the right branch
    return predict_example_tree(tree["right"], example)

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    # TODO: return predicted class for each row of features using the fitted tree.
    predictions = []
    for example in features:
        prediction = predict_example_tree(tree, example)
        predictions.append(prediction)

    return np.array(predictions, dtype=int)

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

