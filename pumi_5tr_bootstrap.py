"""
Functions for loading and manipulating
2-back task RNN data (5-trial), specifically for tracking WM info analysis
"""

import numpy as np
from scipy import io
from scipy import linalg as la
import itertools


################################### load_data_rnn ###################################
def load_data_rnn(subject_id=2, num_unit = 7, tw=None, normalize=False, dataset = 'rnn', bootstrap = 0):
    """
    Return trial-averaged RNN responses for a single subject 

    Returns S x T x N matrix rnn_pmi_avg, where
        S is the number of stimuli
        T is the number of timesteps
        N is the number of RNN hidden layer units

    rnn_pmi_avg contains trial-averaged RNN activity

    """
    
    # Load raw data
    data = io.loadmat(f'2Bonly_{num_unit}units_5tr_{subject_id}.mat')
    rnn = data['EEG_data']  # channels x time x epochs (misnomer - this should be RNN_data, called it EEG_data to be consistent with EEG)
    stim = data['orimat']  
    
    # Extract data from a time window if tw is not None
    if tw is not None:
        rnn = rnn[:, tw, :] 
    
    pmi = stim[:, 1]  # stimulus that is a PMI during 3rd epoch

    # Subtract mean from each trial
    if normalize:
        for i in range(rnn.shape[2]):
            rnn[:, :, i] = rnn[:, :, i] - rnn[:, :, i].mean(1)[:, np.newaxis]


    # Sort by PMI and UMI identity and average over trials
    pmi_stim = np.unique(pmi)
    rnn_pmi_avg = np.zeros((len(pmi_stim),rnn.shape[1], rnn.shape[0]))

    for i, ipmi in enumerate(pmi_stim):
        if bootstrap == 0:
            iepochs = np.argwhere(pmi == ipmi)
            # print(iepochs.shape)
        else:
            iepochs = np.argwhere(pmi == ipmi)
            iepochs = np.random.choice(np.squeeze(iepochs),len(iepochs))
        rnn_pmi_avg[i] = np.squeeze(rnn[:, :, iepochs].mean(2)).T
    
    return rnn_pmi_avg

