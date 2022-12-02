import json
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def cluster2label(y, y_pred):
    clusters = defaultdict(list)
    num_clusters = len(set(y_pred))
    classes = list(set(y))
    num_classes = len(classes)

    for idx, c in enumerate(y_pred):
        clusters[c].append((idx, y_pred[idx]))

    cluster_label_counts = dict()
    replcae_class = {}

    for c in range(num_clusters):
        cluster_label_counts[c] = [0] * num_classes
        instances = clusters[c]
        for i, _ in instances:
            cluster_label_counts[c][y[i]] += 1

        a = cluster_label_counts[c]
        cluster_label_idx = max(range(len(a)), key = lambda x: a[x])
        cluster_label = classes[cluster_label_idx]
        replcae_class[c] = cluster_label
    for idx, c in enumerate(y_pred):
        y_pred[idx] = replcae_class[c]
    return y_pred

y_true = pd.read_csv('iris.data', header=None).pop(4)
y_pred = pd.read_csv('predict.txt', header=None)[0]
y_true = LabelEncoder().fit_transform(y_true)

y_pred = cluster2label(y_true, y_pred)

accuracy = accuracy_score(y_true, y_pred)

with open('metrics.json', 'w') as fd:
    json.dump(
        {'accuracy': accuracy},
        fd
    )