import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.cluster import KMeans
# from sklearn import preprocessing
from random import randint


def rotulo_e_batimentos(filtro, sinal):
    dfs = {}
    for bd in ['BIDMC', 'CapnoBase', 'TROIKA']:
        x = []
        y = []
        df = pd.read_csv(r'data-sets/' + bd + '/'+sinal.upper()+'/filtered_' + filtro + '.csv')
        for i in df.index:
            print('Indice: ' + str(i))
            peaks, _ = find_peaks(np.array(df.iloc[i]), distance=50)

            peaks = peaks[int(len(peaks) * 0.1):int(len(peaks) * 0.9)]

            peaks = [x for x in peaks if
                     np.array(df.iloc[i])[x] < np.mean(np.array(df.iloc[i])) + np.std(np.array(df.iloc[i])) * 3]

            for peak in peaks:
                x.append(np.array(df.iloc[i])[peak - 60:peak + 60])
                y.append(i)
        dfs[bd] = pd.DataFrame({'x': x, 'y': y})
    return dfs


'''
x = [(np.array(df.iloc[i])[p], p) for p in peaks]
scaler = preprocessing.MinMaxScaler()
x = scaler.fit_transform(x)

x = [(c[0]*4, c[1]) for c in x]

print(max([c[0] for c in x]), min([c[0] for c in x]))
print(max([c[1] for c in x]), min([c[1] for c in x]))

kmeans = KMeans(n_clusters=5, n_init='auto').fit(x)
color = ['b' if x == 1 else 'r' if x == 0 else 'g' for x in kmeans.labels_]
'''
