"""
dPCA class with methods for
* fitting stimulus or decision dPCA model
* computing PEV
"""

import numpy as np
import itertools

from dPCA.dPCA import dPCA as _dPCA


class dPCA:
    """
    Reduce dimensionality via dPCA

    Arguments
    ---------
    data: numpy array
        S x T x N matrix of EEG responses, or
        D x S x T x N for computing decision dPCs
    n_dim: int
        dimensionality to reduce to
    decision: bool
        whether to compute decision dPCs
        or stimulus dPCs
    old_version: bool
        whether to use our old dPCA analysis
        that minimizes all temporal variability
    regularizer: float
        regularization coefficient

    Attributes
    -------
    Z: numpy array
        S x T x n_dim matrix of EEG responses
        reduced to n_dim dimensions using
        stimulus dPC's
    encoder: numpy array
        N x n_dim matrix with encoding vectors
        corresponding to top n_dim stimulus dPCs
    decoder: numpy array
        N x n_dim matrix with decoding vectors
        corresponding to top n_dim stimulus dPCs
    pev: numpy array

    """

    def __init__(self, data, n_dim, decision=False, old_version=False, regularizer=1e-8):

        # Set regularizer
        self._set_regularizer(regularizer)

        # Set variable labels
        self._set_variables(data.shape)

        # Set variable for which to compute dPC's
        if decision:
            self.dpca_variable = 'd'
            assert len(data.shape) == 4  # must be D x S x T x N
        else:
            self.dpca_variable = 's'
        assert self.dpca_variable in self.labels

        # Compute dPC's
        self.fit_dpca(data, self.dpca_variable, n_dim, old_version)

        # Compute variance explained
        variance_types = ['global', 'stimulus']
        if self.n_variables == 3:
            variance_types += ['decision']
        self.pev = {
            v: self.variance_explained(data, v[0])
            for v in variance_types
        }

    def fit_dpca(self, data, variable_label, n_dim, old_version):

        self.n_dim = n_dim

        # Transpose to N x S x T for built-in dPCA function
        data_transpose = self._transpose(data)

        # Group terms to define dPC's of interest
        if old_version:  # don't group any terms
            join = None
        else:  # group together all terms related to variable of interest
            join = {
                variable_label: self._group_variables(variable_label)
            }

        # Do dPCA
        self.dpca_model = _dPCA(
            labels=self.labels,
            join=join,
            n_components=self.n_dim,
            regularizer=self.regularizer)
        Z_transpose = self.dpca_model.fit_transform(
            data_transpose)[variable_label]  # n_dim x (D x) S x T

        # Transpose to S x T x n_dim
        self.Z = self._untranspose(Z_transpose)

        # Extract stimulus dPC encoder and decoder
        self.encoder = self.dpca_model.P[variable_label]
        self.decoder = self.dpca_model.D[variable_label]

        # Check decoding vectors are not huge (if they
        # are that's a sign of overfitting)
        decoder_norms = np.linalg.norm(self.decoder, axis=0).max()
        if decoder_norms > 100:
            print(f"WARNING: some decoding vectors are very large",
                  f"(max norm: {decoder_norms: .2e})",
                  ", indicating overfitting")

    def variance_explained(self, data, variance_type, cumulative=True):
        """
        Compute percent variance explained

        Arguments
        ---------
        data: numpy array
            S x T x N matrix of EEG responses, or
            D x S x T x N for computing decision dPCs
        variance_type: str
            'g' for global variance
            's' for stimulus variance
            'd' for decision variance
        cumulative: bool
            whether or not to return cumulative PEV
            or individual components' PEV's
        """

        pev = []

        # Get target data to reconstruct
        if variance_type == 'g':
            X = data.reshape(-1, data.shape[-1])  # (T * S * D) x N
            X -= X.mean(0)  # center the data
            X = X.T  # N x (T * S * D)
        else:
            X_margs = self.dpca_model._marginalize(self._transpose(data))
            X = X_margs[variance_type]  # N x (T * S * D)

        # Loop over dPC's
        for i in range(self.n_dim):

            # Extract ith dPC encoder/decoder pair
            sl = slice(0, i + 1) if cumulative else slice(i, i + 1)
            F, D = self.encoder[:, sl], self.decoder[:, sl]

            # Construct autoencoder matrix from this
            # encoder/decoder pair
            W = F.dot(D.T)

            # Calculate PEV
            pev.append(percent_variance_explained(X, W))

        return pev

    def _set_regularizer(self, regularizer):
        if regularizer < 0:
            raise ValueError('invalid regularizer coefficient: must be greater than 0')
        if regularizer == 0:
            raise ValueError('must use a non-zero regularizer for these data!!')
        self.regularizer = regularizer

    def _set_variables(self, data_shape):

        # Count number of variables
        self.n_variables = len(data_shape) - 1

        # Set labels
        if self.n_variables == 2:  # S x T x N
            self.labels = 'st'
        elif self.n_variables == 3:  # D x S x T x N
            self.labels = 'dst'
        else:
            raise ValueError(f'data must have shape (S x T x N) or (D x S x T x N), but has shape {data_shape}')

        # Check that # of labels matches # of variables
        assert len(self.labels) == self.n_variables

    def _group_variables(self, variable_label):
        """
        Group together all terms that depend
        on given variable label
        """
        grouped_terms = []
        for i in range(len(self.labels)):
            for var in itertools.combinations(self.labels, i + 1):
                if variable_label in var:
                    grouped_terms.append(''.join(var))
        return grouped_terms

    def _transpose_axes(self):
        return tuple(np.r_[
            self.n_variables, np.arange(self.n_variables)])

    def _transpose(self, data):
        return np.transpose(data, self._transpose_axes())

    def _untranspose(self, data_transposed):
        transpose_axes = self._transpose_axes()
        untranspose_axes = tuple([
            transpose_axes.index(i)
            for i in range(len(transpose_axes))
        ])
        return np.transpose(data_transposed, untranspose_axes)


def percent_variance_explained(X, W):
    X_recon = W.dot(X)
    mse = ((X - X_recon) ** 2).sum()
    var = (X ** 2).sum()
    return 1 - mse / var
