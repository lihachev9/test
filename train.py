import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv('iris.data', header=None)
data.pop(4)

km = KMeans(n_clusters=3)
km.fit(data)
y_pred = km.predict(data)

with open('predict.txt', 'w') as f:
    print(*y_pred, sep='\n', end='', file=f)
