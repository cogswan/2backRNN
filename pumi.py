"""
Functions for loading and manipulating
2-back task data for dPCA
"""

import numpy as np
from scipy import io
from scipy import linalg as la
import itertools


################################### load EEG data ###################################
def load_data(subject_id=1, tw=None, normalize=False, dataset = 'eeg', decision = False): 
    """
    Return trial-averaged EEG responses for a single subject 

    Returns (D x) S x T x N matrix eeg_avg, where
        (D is the number of decisions)
        S is the number of stimuli
        T is the number of timesteps
        N is the number of EEG channels

    eeg_avg contains trial-averaged EEG response

    """
    # Load raw data
    data = io.loadmat(f'{subject_id}_2b_cont_eegmat.mat')
        
    eeg = data['EEG_data']  # channels x time x epochs
    stim = data['orimat']  # stimulus identity each trial
    resp = data['respmat']
    
    ori = stim[:, 0]  # stimulus identity n
    dec = resp[:, 1]  # decision variable at n+1
    ori_dec = stim[:, 1] #stimulus n+1 used for decision analysis
    
    # Subtract mean from each trial
    if normalize:
        for i in range(eeg.shape[2]):
            eeg[:, :, i] = eeg[:, :, i] - eeg[:, :, i].mean(1)[:, np.newaxis]

    # Keep samples from a certain time window
    if tw is not None:
        eeg = eeg[:, tw, :]
        
    # Sort by stimulus identity and average over trials
    ori_stim = np.unique(ori)
    dec_stim = np.array([1,2])
    
    if decision is False:
        eeg_avg = np.zeros((len(ori_stim), eeg.shape[1], eeg.shape[0]))
        for i, iori in enumerate(ori_stim):
            iepochs = np.argwhere(ori == iori) 
            eeg_avg[i] = eeg[:, :, iepochs].mean(2).T
    else:
         eeg_avg = np.zeros((len(dec_stim), len(ori_stim), eeg.shape[1], eeg.shape[0]))
         for i, iori in enumerate(ori_stim):
            for j, jdec in enumerate(dec_stim):
                iepochs = np.intersect1d(np.argwhere(ori_dec == iori), np.argwhere(dec == jdec))
                eeg_avg[j,i] = eeg[:, :, iepochs].mean(2).T 
                
    return eeg_avg


################################### load RNN data ###################################
def load_data_rnn(subject_id=1, num_unit = 7, tw=None, normalize=False, dataset = 'rnn', decision = False):
    """
    Return trial-averaged RNN responses for a single network

    Returns (D x) S x T x N matrix rnn_avg, where
        (D is the number of decisions)
        S is the number of stimuli
        T is the number of timesteps
        N is the number of RNN units

    rnn_avg contains trial-averaged RNN response

    """
    
    # Load raw data
    data = io.loadmat(f'2Bonly_{num_unit}units_2tr_{subject_id}.mat')
    rnn = data['EEG_data']  # channels x time x epochs (misnomer - this should be RNN_data, called it EEG_data to be consistent with EEG)
    stim = data['orimat'] 
    resp = data['respmat']
    
    # Extract data for individual stimulus events from each stimulus event
    ori = stim[:, 0]  #stimulus n
    dec = resp[:, 1]  # decision
    ori_dec = stim[:, 1] #stimulus n+1 used for decision analysis
    
    # Subtract mean from each trial
    if normalize:
        for i in range(eeg.shape[2]):
            rnn[:, :, i] = rnn[:, :, i] - rnn[:, :, i].mean(1)[:, np.newaxis]
            
    # Keep samples from a certain time window
    if tw is not None:
        rnn = rnn[:, tw, :]
        
    # Sort by PMI and UMI identity and average over trials
    ori_stim = np.unique(ori)
    dec_stim = np.array([1,2])
    
    if decision is False: 
        rnn_avg = np.zeros((len(ori_stim), 6, rnn.shape[0]))
        for i, iori in enumerate(ori_stim):
            iepochs = np.argwhere(ori == iori)
            rnn_avg[i] = np.squeeze(rnn[:,:,iepochs].mean(2)).T
        
    else:
        rnn_avg = np.zeros((len(dec_stim),len(ori_stim), 6, rnn.shape[0]))
        
        for i, iori in enumerate(ori_stim):
            for j, jdec in enumerate(dec_stim):
                iepochs = np.intersect1d(np.argwhere(ori_dec == iori), np.argwhere(dec == jdec))
                rnn_avg[j,i] = np.squeeze(rnn[:,:,iepochs].mean(2)).T

    return rnn_avg