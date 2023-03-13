import sys

import pandas as pd
from scipy import signal
import pywt
import matplotlib.pyplot as plt

def main(tipo):
    dfs_name = ['BIDMC', "CapnoBase", "TROIKA"]
    for sinal_avaliado in ['ECG', 'PPG']:
        for name in dfs_name:
            df = pd.read_csv('data-sets/'+name+'/'+sinal_avaliado+'/raw.csv').drop("Unnamed: 0", axis=1)
            if tipo == 'butterworth':
                param = pd.read_csv('PARAM_BUTTER')
                p = param.loc[(param['Dataset'] == name) & (param['sinal'] == sinal_avaliado)].iloc[0]
                sos = signal.butter(p['ordem'], [p['undercut'], p['uppercut']],btype='bandpass',fs=p['samplerate'],
                                    output='sos')
            if tipo == 'ChebyshevII':
                param = pd.read_csv('PARAM_CHEBY')
                p = param.loc[(param['Dataset'] == name) & (param['sinal'] == sinal_avaliado)].iloc[0]
                sos = signal.cheby1(p['ordem'], p['ripple'], [p['undercut'], p['uppercut']], btype='bandpass',
                                    output='sos')
            if tipo == 'ChebyshevI':
                param = pd.read_csv('PARAM_CHEBY')
                p = param.loc[(param['Dataset'] == name) & (param['sinal'] == sinal_avaliado)].iloc[0]
                sos = signal.cheby2(p['ordem'], p['ripple'], [p['undercut'], p['uppercut']], btype='bandpass',
                                    output='sos')
            df = df.apply(lambda row: pd.Series(signal.sosfilt(sos, row)), axis=1)
            df.to_csv('data-sets/'+name+'/'+sinal_avaliado+'/filtered_'+tipo+'.csv')


if __name__ == '__main__':
    main(sys.argv[1])
