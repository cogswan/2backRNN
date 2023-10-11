# 2backRNN
Code for project "Priority-based transformations of stimulus representation in visual working memory"
(https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009062)

'2back_rnn_train.ipynb' trains and tests 7-unit and 60-unit RNNs (with circular input) and plots the PCA projections shown in Figure 4.
'dpca_timecourse.ipynb' plots timecourses of dPCA projection of stimulus means, and timecourses of scalar transform, for both RNN and EEG (Figures 5 and 6).
'subspace_angle.ipynb' finds angles between each pair of subspaces identified by dPCA for both RNN and EEG (Figure 7)
'pev_calc.ipynb' finds cumulative percent explained variance of each dPC identified.
'track_wm_info.ipynb' implements the method to track the disappearance of WM information using dPCA.
The '.py' files contain support functions.
