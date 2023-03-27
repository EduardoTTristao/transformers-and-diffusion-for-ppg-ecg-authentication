import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.cluster import KMeans
# from sklearn import preprocessing
from random import randint


def rotulo_e_batimentos(filtro, sinal, modo='treino'):
    dfs = {}
    for bd in ['BIDMC', 'CapnoBase', 'TROIKA']:
        x = []
        y = []

        print(bd)

        df = pd.read_csv(r'data-sets/' + bd + '/'+sinal.upper()+'/filtered_' + filtro + '.csv')

        if modo == 'treino':
            indices = df.index[0:int(len(df.index)*0.7)]
        else:
            if modo == 'teste':
                indices = df.index[int(len(df.index)*0.7):int(len(df.index)*0.85)]
            else:
                indices = df.index[int(len(df.index)*0.85):0]

        for i in indices:
            print('Lendo indivíduo número: ' + str(i))

            np_array = np.array(df.iloc[i])

            peaks, _ = find_peaks(np_array, distance=50)

            peaks = peaks[int(len(peaks) * 0.1):int(len(peaks) * 0.9)]

            # mean = np.mean(np_array)
            # std = np.std(np_array)

            for peak in peaks:
                # if np_array[peak] < mean + std * 4:
                x.append(np_array[peak - 60:peak + 60])
                y.append(i + (42 if bd == 'BIDMC' else 0 if bd == 'CapnoBase' else 95))

        dfs[bd] = pd.DataFrame({'x': x, 'y': y})
    return dfs


def autenticacoes_permissoes(df, num_adversarios):
    df_final = {'x': [], 'y': []}
    for x in range(0, df['y'].nunique()):
        users = df[df['y'] == x].sample(n=num_adversarios)
        invaders = df[df['y'] != x].sample(n=num_adversarios)
        for user_template in users['x']:
            for invader in invaders['x']:
                df_final['x'].append([user_template, invader])
                df_final['y'].append(0)
            for user in users['x']:
                df_final['x'].append([user_template, user])
                df_final['y'].append(1)
    return df_final


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
