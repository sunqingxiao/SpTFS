import os
import sys
import random
import numpy as np

def main():
    prec_data = np.load('prec-data.npz')
    precs = prec_data['precs']

    num_modes = precs.shape[0]
    mean_precs = np.zeros((num_modes), dtype='float32')
    median_precs = np.zeros((num_modes), dtype='float32')
    per20_precs = np.zeros((num_modes), dtype='float32')
    per40_precs = np.zeros((num_modes), dtype='float32')
    per60_precs = np.zeros((num_modes), dtype='float32')
    per80_precs = np.zeros((num_modes), dtype='float32')
    stdev_precs = np.zeros((num_modes), dtype='float32')
    
    for mode in range(0, num_modes):
        mean_precs[mode] = np.mean(precs[mode])
        median_precs[mode] = np.median(precs[mode])
        per20_precs[mode] = np.percentile(precs[mode], 20)
        per40_precs[mode] = np.percentile(precs[mode], 40)
        per60_precs[mode] = np.percentile(precs[mode], 60)
        per80_precs[mode] = np.percentile(precs[mode], 80)
        stdev_precs[mode] = np.std(precs[mode], ddof=1)
    
    print('mean precisions: {}'.format(mean_precs))
    print('median precisions: {}'.format(median_precs))
    print('20% quantile of: {}'.format(per20_precs))
    print('40% quantile of: {}'.format(per40_precs))
    print('60% quantile of: {}'.format(per60_precs))
    print('80% quantile of: {}'.format(per80_precs))

if __name__=='__main__':
    main()
