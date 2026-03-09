# Additional external libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper
# from sionna.mimo import StreamManagement, lmmse_equalizer, zf_equalizer, mf_equalizer,\
#                             lmmse_matrix
# from sionna.utils import flatten_dims, split_dim, flatten_last_dims,\
#                              expand_to_rank, inv_cholesky
from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, UMa, UMi, RMa, PanelArray
from .block import config, dtypes, Block
from sionna.channel import GenerateOFDMChannel
from sionna.utils import insert_dims

# def get_stream_management(direction,
#                           num_rx,
#                           num_tx,
#                           num_streams_per_ut,
#                           num_ut_per_sector):
#     """
#     Instantiate a StreamManagement object.
#     It determines which data streams are intended for each receiver
#     """
#     if direction == 'downlink':
#         num_streams_per_tx = num_streams_per_ut * num_ut_per_sector
#         # RX-TX association matrix
#         rx_tx_association = np.zeros([num_rx, num_tx])
#         idx = np.array([[i1, i2] for i2 in range(num_tx) for i1 in
#                         np.arange(i2*num_ut_per_sector,
#                                   (i2+1)*num_ut_per_sector)])
#         rx_tx_association[idx[:, 0], idx[:, 1]] = 1

#     else:
#         num_streams_per_tx = num_streams_per_ut
#         # RX-TX association matrix
#         rx_tx_association = np.zeros([num_rx, num_tx])
#         idx = np.array([[i1, i2] for i1 in range(num_rx) for i2 in
#                         np.arange(i1*num_ut_per_sector,
#                                   (i1+1)*num_ut_per_sector)])
#         rx_tx_association[idx[:, 0], idx[:, 1]] = 1

#     stream_management = StreamManagement(
#         rx_tx_association, num_streams_per_tx)
#     return stream_management

# def estimate_achievable_rate(sinr_eff_db_last,
#                              num_ofdm_sym,
#                              num_subcarriers):
#     """ Estimate achievable rate """
#     # [batch_size, num_bs, num_ut_per_sector]
#     rate_achievable_est = log2(tf.cast(1, sinr_eff_db_last.dtype) +
#                                db_to_lin(sinr_eff_db_last))

#     # Broadcast to time/frequency grid
#     # [batch_size, num_bs, num_ofdm_sym, num_subcarriers, num_ut_per_sector]
#     rate_achievable_est = insert_dims(
#         rate_achievable_est, 2, axis=-2)
#     rate_achievable_est = tf.tile(rate_achievable_est,
#                                   [1, 1, num_ofdm_sym, num_subcarriers, 1])
#     return rate_achievable_est

# def init_result_history(batch_size,
#                         num_slots,
#                         num_bs,
#                         num_ut_per_sector):
#     """ Initialize dictionary containing history of results """
#     hist = {}
#     for key in ['pathloss_serving_cell',
#                 'tx_power', 'olla_offset',
#                 'sinr_eff', 'pf_metric',
#                 'num_decoded_bits', 'mcs_index',
#                 'harq', 'num_allocated_re']:
#         hist[key] = tf.TensorArray(
#             size=num_slots,
#             element_shape=[batch_size,
#                            num_bs,
#                            num_ut_per_sector],
#             dtype=tf.float32)
#     return hist

# def record_results(hist,
#                    slot,
#                    sim_failed=False,
#                    pathloss_serving_cell=None,
#                    num_allocated_re=None,
#                    tx_power_per_ut=None,
#                    num_decoded_bits=None,
#                    mcs_index=None,
#                    harq_feedback=None,
#                    olla_offset=None,
#                    sinr_eff=None,
#                    pf_metric=None,
#                    shape=None):
#     """ Record results of last slot """
#     if not sim_failed:
#         for key, value in zip(['pathloss_serving_cell', 'olla_offset', 'sinr_eff',
#                                'num_allocated_re', 'tx_power', 'num_decoded_bits',
#                                'mcs_index', 'harq'],
#                               [pathloss_serving_cell, olla_offset, sinr_eff,
#                                num_allocated_re, tx_power_per_ut, num_decoded_bits,
#                                mcs_index, harq_feedback]):
#             hist[key] = hist[key].write(slot, tf.cast(value, tf.float32))
#         # Average PF metric across resources
#         hist['pf_metric'] = hist['pf_metric'].write(
#             slot, tf.reduce_mean(pf_metric, axis=[-2, -3]))
#     else:
#         nan_tensor = tf.cast(tf.fill(shape,
#                                      float('nan')), dtype=tf.float32)
#         for key in hist:
#             hist[key] = hist[key].write(slot, nan_tensor)
#     return hist

# def clean_hist(hist, batch=0):
#     """ Extract batch, convert to Numpy, and mask metrics when user is not
#     scheduled """
#     # Extract batch and convert to Numpy
#     for key in hist:
#         try:
#             # [num_slots, num_bs, num_ut_per_sector]
#             hist[key] = hist[key].numpy()[:, batch, :, :]
#         except:
#             pass

#     # Mask metrics when user is not scheduled
#     hist['mcs_index'] = np.where(
#         hist['harq'] == -1, np.nan, hist['mcs_index'])
#     hist['sinr_eff'] = np.where(
#         hist['harq'] == -1, np.nan, hist['sinr_eff'])
#     hist['tx_power'] = np.where(
#         hist['harq'] == -1, np.nan, hist['tx_power'])
#     hist['num_allocated_re'] = np.where(
#         hist['harq'] == -1, 0, hist['num_allocated_re'])
#     hist['harq'] = np.where(
#         hist['harq'] == -1, np.nan, hist['harq'])
#     return hist

# class PostEqualizationSINR(Block):
#     # pylint: disable=line-too-long
#     r"""
#     Abstract block that computes the SINR after equalization

#     This function computes the post-equalization SINR for every transmitted
#     stream from the :class:`~sionna.phy.ofdm.PrecodedChannel`.
#     A stream goes from a specific transmitter to a specific
#     receiver and is characterized by a precoding vector and an
#     equalization vector.

#     Every transmitter is equipped with `num_tx_ant` antennas and every receiver
#     is equipped with `num_rx_ant` antennas. All transmitters send the same number
#     of streams :math:`S`. A transmitter can allocate different power to different streams.

#     Let
#     :math:`\mathbf{H}_{i,j}\in\mathbb{C}^{\text{num_rx_ant}\times\text{num_tx_ant}}`
#     be the complex channel matrix between receiver :math:`i` and transmitter
#     :math:`j`. We denote by
#     :math:`\mathbf{g}_{j_,s}\in\mathbb{C}^{\text{num_tx_ant}}` the precoding
#     vector
#     for stream :math:`s` sent by transmitter :math:`j`.
#     Then, the received signal at receiver :math:`i` can be expressed as:

#     .. math::
#         \mathbf{y}_i = \sum_{j,s} \mathbf{H}_{i,j} \mathbf{g}_{j,s} \sqrt{p_{j,s}} x_{j,s} + \mathbf{n}_{i} 

#     where :math:`x_{j,s}` and :math:`p_{j,s}` are the unit-power transmit symbol
#     and associated transmission power for stream :math:`s`, respectively, and
#     :math:`\mathbf{n}_{i}` is the additive noise, distributed as
#     :math:`\mathcal{C}\mathcal{N}(0,\sigma^2 \mathbf{I})`.

#     By stacking the precoding vectors into a matrix :math:`\mathbf{G}_j=\left[\mathbf{g}_{j,1}, \ldots, \mathbf{g}_{j,S}\right]`,
#     and using the definition of the precoded channel :math:`\widetilde{\mathbf{H}}_{i,j}` in
#     :eq:`effective_precoded_channel`, the received signal can be rewritten as:

#     .. math::
#         \mathbf{y}_i = \sum_j \widetilde{\mathbf{H}}_{i,j} \mathop{\text{diag}}(x_{j,1},...,x_{j,S}) + \mathbf{n}_{i}

#     Next, let :math:`\mathbf{f}_{i,j,s} \in\mathbb{C}^{\text{num_rx_ant}}`
#     be the equalization vector for stream :math:`s` of transmitter :math:`j`,
#     applied by the intended receiver :math:`i`. Then, the useful signal power for stream :math:`s` of transmitter :math:`j` is:

#     .. math::
#         u_{i,j,s} = p_{j,s} \left| \mathbf{f}_{i,j,s}^\mathsf{H} \mathbf{H}_{i,j} \mathbf{g}_{j, s} \right|^2.

#     We assume that the transmitted symbols :math:`x_{j,s}` are uncorrelated among each
#     other. Then, the interference power for this stream can be written
#     as: 

#     .. math::
#         v_{i,j,s} = \sum_{(j',s') \ne (j,s)} p_{j',s'} \left| \mathbf{f}_{i,j,s}^\mathsf{H} \mathbf{H}_{i,j'} \mathbf{g}_{j', s'} \right|^2.

#     The post-equalization noise power can be expressed as:

#     .. math::
#         n_{i,j,s} = \sigma^2 \| \mathbf{f}_{i,j,s} \|^2.

#     With these definitions, the SINR for this stream which is finally computed as:

#     .. math::
#         \mathrm{SINR}_{i,j,s} = \frac{u_{i,j,s}}{v_{i,j,s} + n_{i,j,s}}.

#     Note, that the intended receiver :math:`i` for a particular stream
#     :math:`(j,s)` is defined by the :class:`~sionna.phy.mimo.StreamManagement`
#     object.


#     Parameters
#     ----------
#     resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
#         ResourceGrid to be used

#     stream_management : :class:`~sionna.phy.mimo.StreamManagement`
#         StreamManagement to be used

#     precision : `None` (default) | "single" | "double"
#         Precision used for internal calculations and outputs.
#         If set to `None`,
#         :attr:`~sionna.phy.config.Config.precision` is used.

#     Input
#     -----
#     h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
#         Effective channel after precoding as defined in :eq:`effective_precoded_channel`

#     no : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_effective_subcarriers] (or only the first n dims), `tf.float`
#         Noise variance

#     h_eff_hat : `None` (default) | [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
#         Estimated effective channel after precoding. If set to `None`,
#         the actual channel realizations are used.

#     Output
#     ------
#     sinr : [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx, num_streams_per_rx], `tf.float`
#         SINR after equalization
#     """
#     def __init__(self,
#                  resource_grid,
#                  stream_management,
#                  precision=None,
#                  **kwargs):
#         super().__init__(precision=precision, **kwargs)
#         self._resource_grid = resource_grid
#         self._stream_management = stream_management

#     def get_per_rx_channels(self, h_eff):
#         # pylint: disable=line-too-long
#         r""" Extract desired and undesired channels for each receiver

#         Input
#         -----
#         h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`, `tf.complex`
#             Effective precoded channel. Can be estimated or true.

#         Output
#         ------
#         h_eff_desired : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `tf.complex`
#             Desired effective channels

#         h_eff_undesired : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_interfering_streams_per_rx], `tf.complex`
#             Undesired effective channels

#         """
#         # Reshape h_eff for the construction of desired/interfering channels:
#         # [num_rx, num_tx, num_streams_per_tx, batch_size, num_rx_ant, ,...
#         #  ..., num_ofdm_symbols, num_effective_subcarriers]
#         perm = [1, 3, 4, 0, 2, 5, 6]
#         h_eff = tf.transpose(h_eff, perm)

#         # Flatten first three dimensions:
#         # [num_rx*num_tx*num_streams_per_tx, batch_size, num_rx_ant, ...
#         #  ..., num_ofdm_symbols, num_effective_subcarriers]
#         h_eff = flatten_dims(h_eff, 3, 0)

#         # Gather desired and undesired channels
#         ind_desired = self._stream_management.detection_desired_ind
#         ind_undesired = self._stream_management.detection_undesired_ind
#         h_eff_desired = tf.gather(h_eff, ind_desired, axis=0)
#         h_eff_undesired = tf.gather(h_eff, ind_undesired, axis=0)

#         # Split first dimension to separate RX and TX:
#         # [num_rx, num_streams_per_rx, batch_size, num_rx_ant, ...
#         #  ..., num_ofdm_symbols, num_effective_subcarriers]
#         h_eff_desired = split_dim(h_eff_desired,
#                                  [self._stream_management.num_rx,
#                                   self._stream_management.num_streams_per_rx],
#                                  0)
#         h_eff_undesired = split_dim(h_eff_undesired,
#                                    [self._stream_management.num_rx, -1], 0)

#         # Permutate dims to
#         # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,..
#         #  ..., num_rx_ant, num_streams_per_rx(num_interfering_streams_per_rx)]
#         perm = [2, 0, 4, 5, 3, 1]
#         h_eff_desired = tf.transpose(h_eff_desired, perm)
#         h_eff_undesired = tf.transpose(h_eff_undesired, perm)

#         return h_eff_desired, h_eff_undesired

#     def compute_interference_covariance_matrix(self, no=None, h_eff_undesired=None):
#         # pylint: disable=line-too-long
#         r"""Compute the interference covariance matrix

#         Input
#         -----
#         no : `None` (default) | [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant], `tf.float`
#             Noise variance

#         h_eff_undesired : `None` (default) | [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_interfering_streams_per_rx], `tf.complex`
#             Undesired effective channels. If set to `None`, the actual channel realizations are used.

#         Output
#         ------
#         s : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_rx_ant], `tf.complex`
#             Interference covariance matrix
#         """
#         s_no = 0.
#         if no is not None:
#             # Diagonal matrix
#             no = tf.cast(no, self.cdtype)
#             s_no = tf.linalg.diag(no)

#         s_inf = 0.
#         if h_eff_undesired is not None:
#             s_inf = tf.matmul(h_eff_undesired, h_eff_undesired, adjoint_b=True)

#         s = s_no + s_inf

#         return s

#     def compute_desired_signal_power(self, h_eff_desired, f):
#         # pylint: disable=line-too-long
#         r""" Compute the desired signal power

#         Input
#         -----
#         h_eff_desired : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `tf.complex`
#             Desired effective channels

#         f : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_rx_ant], `tf.complex`
#             Receive combining vectors

#         Output
#         ------
#         signal_power : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx], `tf.float`
#             Desired signal power
#         """
#         signal_power = tf.einsum('...mn,...nm->...m', f, h_eff_desired)
#         signal_power = tf.abs(signal_power)**2
#         return signal_power

#     def compute_total_power(self, h_eff_desired, h_eff_undesired, f):
#         """
#         Compute the total power from all transmitters

#         Input
#         -----
#         h_eff_desired : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `tf.complex`
#             Desired effective channels

#         h_eff_undesired : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_interfering_streams_per_rx], `tf.complex`
#             Undesired effective channels

#         f : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,
#         num_streams_per_rx, num_rx_ant], `tf.complex`
#             Receive combining vectors

#         Output
#         ------
#         total_power : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, 1], `tf.float`
#             Total power
#         """
#         h_eff = tf.concat([h_eff_desired, h_eff_undesired], axis=-1)
#         total_power = tf.abs(tf.matmul(f, h_eff))**2
#         total_power = tf.reduce_sum(total_power, axis=-1)
#         return total_power

#     def compute_noise_power(self, no, f):
#         # pylint: disable=line-too-long
#         r""" Compute the noise power

#         Input
#         -----
#         no : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant], `tf.float`
#             Noise variance

#         f : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_streams_per_rx, num_rx_ant], `tf.complex`
#             Receive combining vectors
#         """
#         no = tf.expand_dims(tf.math.real(no), axis=-2)
#         noise_power = tf.reduce_sum(tf.abs(f)**2 * no, axis=-1)
#         return noise_power

#     def compute_sinr(self, h_eff_desired, h_eff_undesired, no, f):
#         # pylint: disable=line-too-long
#         r""" Compute the SINR

#         Input
#         -----
#         h_eff_desired : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `tf.complex`
#             Desired effective channels

#         h_eff_undesired : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_interfering_streams_per_rx], `tf.complex`
#             Undesired effective channels

#         no : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant], `tf.float`
#             Noise variance

#         f : [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers, num_rx_ant, num_streams_per_rx], `tf.complex`
#             Equalization matrix

#         Output
#         ------
#         sinr : [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx, num_streams_per_rx], `tf.float`
#             Post-equalization SINR
#         """
#         signal_power = self.compute_desired_signal_power(h_eff_desired, f)
#         total_power = self.compute_total_power(h_eff_desired,h_eff_undesired,f)
#         # For numerical stability, avoid negative values
#         interference_power = tf.maximum(total_power - signal_power, tf.cast(0, self.rdtype))
#         noise_power = self.compute_noise_power(no, f)
#         sinr = tf.math.divide_no_nan(signal_power,
#                                      interference_power + noise_power)

#         # Reshape to desired dimensions
#         sinr = tf.transpose(sinr, [0, 2, 3, 1, 4])
#         return sinr

#     @abstractmethod
#     def call(self, h_eff, no, h_eff_hat=None):
#         pass

# class LMMSEPostEqualizationSINR(PostEqualizationSINR):
#     # pylint: disable=line-too-long
#     r"""
#     Block that computes the SINR after LMMSE equalization

#     The equalization matrix is the one computed by
#     :meth:`~sionna.phy.mimo.lmmse_matrix`.

#     Parameters
#     ----------
#     resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
#         ResourceGrid to be used

#     stream_management : :class:`~sionna.phy.mimo.StreamManagement`
#         StreamManagement to be used

#     precision : `None` (default) | "single" | "double"
#         Precision used for internal calculations and outputs.
#         If set to `None`,
#         :attr:`~sionna.phy.config.Config.precision` is used.

#     Input
#     -----
#     h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
#         Effective channel after precoding as defined in :eq:`effective_precoded_channel`

#     no : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_effective_subcarriers] (or only the first n dims), `tf.float`
#         Noise variance

#     h_eff_hat : `None` (default) | [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
#         Estimated effective channel after precoding. If set to `None`,
#         the actual channel realizations are used.

#     interference_whitening : `bool` (default=True)
#         If set to `True`, also the interference from undesired streams (e.g.,
#         from other cells) is whitened

#     Output
#     ------
#     sinr : [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx, num_streams_per_rx], `tf.float`
#         SINR after equalization
#     """
#     def call(self, h_eff, no, h_eff_hat=None, interference_whitening=True):
#         if h_eff_hat is None:
#             h_eff_hat = h_eff

#         #  Ensure that noise variance has the right dimensions
#         no = expand_to_rank(no, 5, -1)
#         no = tf.broadcast_to(no, [tf.shape(h_eff)[0],
#                                   h_eff.shape[1],
#                                   h_eff.shape[2],
#                                   h_eff.shape[5],
#                                   h_eff.shape[6]])
#         # [batch_size, num_rx, num_ofdm_symbols, num_effective_subcarriers,...
#         #  ... num_rx_ant]
#         no = tf.transpose(no, [0, 1, 3, 4, 2])

#         # Get estimated desired and undesired channels
#         h_eff_desired, h_eff_undesired = self.get_per_rx_channels(h_eff_hat)

#         # Compute estimated interference covariance matrix
#         if interference_whitening:
#             s = self.compute_interference_covariance_matrix(
#                                 no=no,
#                                 h_eff_undesired=h_eff_undesired)
#         else:
#             s = self.compute_interference_covariance_matrix(
#                     no=no)

#         # Whiten channels
#         l_inv = inv_cholesky(s) # Compute whitening matrix
#         h_eff_desired = tf.matmul(l_inv, h_eff_desired)
#         h_eff_undesired = tf.matmul(l_inv, h_eff_undesired)

#         # Compute equalization matrix
#         f = lmmse_matrix(h_eff_desired, precision=self.precision)

#         # Compute SINR
#         sinr = self.compute_sinr(h_eff_desired, h_eff_undesired,
#                                  tf.ones_like(no), f)

#         return sinr

# class PrecodedChannel(Block):
#     # pylint: disable=line-too-long
#     r"""
#     Abstract base class to compute the effective channel after precoding

#     Its output can be used to compute the :class:`~sionna.phy.ofdm.PostEqualizationSINR`.

#     Let
#     :math:`\mathbf{H}_{i,j}\in\mathbb{C}^{\text{num_rx_ant}\times\text{num_tx_ant}}`
#     be the channel matrix between transmitter :math:`j`
#     and receiver :math:`i` and let
#     :math:`\mathbf{G}_{j}\in\mathbb{C}^{\text{num_tx_ant}\times\text{num_streams_per_tx}}`
#     be the precoding matrix of transmitter :math:`j`. 

#     The effective channel :math:`\widetilde{\mathbf{H}}_{i,j}\in\mathbb{C}^{\text{num_rx_ant}\times\text{num_streams_per_tx}}`
#     after precoding is given by

#     .. math::
#         :label: effective_precoded_channel

#         \widetilde{\mathbf{H}}_{i,j} = \mathbf{H}_{i,j}\mathbf{G}_{j}\mathop{\text{diag}}(\sqrt{p_{j,1}},...,\sqrt{p_{j,\text{num_streams_per_tx}}})

#     where :math:`p_{j,s}` is the transmit power of stream :math:`s` of transmitter :math:`j`.

#     Parameters
#     ----------
#     resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
#         ResourceGrid to be used

#     stream_management : :class:`~sionna.phy.mimo.StreamManagement`
#         StreamManagement to be used

#     precision : `None` (default) | "single" | "double"
#         Precision used for internal calculations and outputs.
#         If set to `None`,
#         :attr:`~sionna.phy.config.Config.precision` is used.

#     Input
#     -----
#     h : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
#         Actual channel realizations

#     tx_power : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] (or first n dims), `tf.float32`
#         Power of each stream for each transmitter

#     h_hat : `None` (default) | [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
#         Channel knowledge based on which the precoding is computed. If set to `None`,
#         the actual channel realizations are used.

#     Output
#     ------
#     h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols num_effective_subcarriers], `tf.complex`
#         The effective channel after precoding. Nulled subcarriers are
#         automatically removed.
#     """
#     def __init__(self,
#                  resource_grid,
#                  stream_management,
#                  precision=None,
#                  **kwargs):
#         super().__init__(precision=precision, **kwargs)
#         assert isinstance(resource_grid, sionna.phy.ofdm.ResourceGrid)
#         assert isinstance(stream_management, sionna.phy.mimo.StreamManagement)
#         self._resource_grid = resource_grid
#         self._stream_management = stream_management
#         self._remove_nulled_scs = RemoveNulledSubcarriers(self._resource_grid)

#     def get_desired_channels(self, h_hat):
#         # pylint: disable=line-too-long
#         r"""
#         Get the desired channels for precoding

#         Input
#         -----
#         h_hat : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
#             Channel knowledge based on which the precoding is computed

#         Output
#         ------
#         h_pc_desired : [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx, num_tx_ant], `tf.complex`
#             Desired channels for precoding
#         """
#         # h_hat has shape
#         # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols...
#         # ..., fft_size]

#         # Transpose:
#         # [num_tx, num_rx, num_rx_ant, num_tx_ant, num_ofdm_symbols,...
#         #  ..., fft_size, batch_size]
#         h_pc_desired = tf.transpose(h_hat, [3, 1, 2, 4, 5, 6, 0])

#         # Gather desired channel for precoding:
#         # [num_tx, num_rx_per_tx, num_rx_ant, num_tx_ant, num_ofdm_symbols,...
#         #  ..., fft_size, batch_size]
#         h_pc_desired = tf.gather(h_pc_desired,
#                                  self._stream_management.precoding_ind,
#                                  axis=1, batch_dims=1)
#         # Flatten dims 1,2:
#         # [num_tx, num_rx_per_tx * num_rx_ant, num_tx_ant, num_ofdm_symbols,...
#         #  ..., fft_size, batch_size]
#         h_pc_desired = flatten_dims(h_pc_desired, 2, axis=1)

#         # Transpose:
#         # [batch_size, num_tx, num_ofdm_symbols, fft_size,...
#         #  ..., num_streams_per_tx, num_tx_ant]
#         h_pc_desired = tf.transpose(h_pc_desired, [5, 0, 3, 4, 1, 2])

#         num_streams_per_tx = self._stream_management.num_streams_per_tx

#         # Check if number of streams per tx matches the channel dimensions
#         if h_pc_desired.shape[-2] != num_streams_per_tx:
#             msg = "The required number of streams per transmitter" \
#                   + " does not match the channel dimensions"
#             raise ValueError(msg)


#         return h_pc_desired

#     def compute_effective_channel(self, h, g):
#         # pylint: disable=line-too-long
#         r"""Compute effective channel after precoding

#         Input
#         -----
#         h : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
#             Actual channel realizations

#         g : [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx], `tf.complex`
#             Precoding matrix
#         Output
#         ------
#         h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols num_effective_subcarriers], `tf.complex`
#             The effective channel after precoding. Nulled subcarriers are
#             automatically removed.
#         """
#         # Input dimensions:
#         # h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,...
#         #     ..., num_ofdm_symbols, fft_size]
#         # g: [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant,
#         #     ..., num_streams_per_tx]

#         # Transpose h to shape:
#         # [batch_size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant,...
#         #  ..., num_tx_ant]
#         h = tf.transpose(h, [0, 1, 3, 5, 6, 2, 4])
#         h = tf.cast(h, g.dtype)

#         # Add one dummy dimension to g to be broadcastable to h:
#         # [batch_size, 1, num_tx, num_ofdm_symbols, fft_size, num_tx_ant,...
#         #  ..., num_streams_per_tx]
#         g = tf.expand_dims(g, 1)

#         # Compute post precoding channel:
#         # [batch_size, num_rx, num_tx, num_ofdm_symbols, fft_size, num_rx_ant,...
#         #  ..., num_streams_per_tx]
#         h_eff = tf.matmul(h, g)

#         # Permute dimensions to common format of channel tensors:
#         # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,...
#         #  ..., num_ofdm_symbols, fft_size]
#         h_eff = tf.transpose(h_eff, [0, 1, 5, 2, 6, 3, 4])

#         # Remove nulled subcarriers:
#         # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,...
#         #  ..., num_ofdm_symbols num_effective_subcarriers]
#         h_eff = self._remove_nulled_scs(h_eff)

#         return h_eff

#     def apply_tx_power(self, g, tx_power):
#         r"""Apply transmit power to precoding vectors

#         Input
#         -----
#         g : [batch_size, num_tx, num_ofdm_symbols, fft_size, num_tx_ant, num_streams_per_tx], `tf.complex`
#             Precoding vectors

#         tx_power : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] (or first n dims), `tf.float32`
#             Power of each stream for each transmitter
#         """
#         # [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size,
#         # ...num_streams_per_tx]
#         tx_power = expand_to_rank(tx_power, 6, axis=-1)
#         # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_streams_per_tx]
#         tx_power = tf.transpose(tx_power, [0, 1, 3, 4, 5, 2])
#         tx_power = tf.broadcast_to(tx_power, tf.shape(g))

#         # Apply tx power to precoding matrix
#         g = tf.cast(tf.sqrt(tx_power), self.cdtype) * g

#         return g

#     @abstractmethod
#     def call(self, h, tx_power, h_hat=None, **kwargs):
#         pass

# class RZFPrecodedChannel(PrecodedChannel):
#     # pylint: disable=line-too-long
#     r"""
#     Compute the effective channel after RZF precoding

#     The precoding matrices are obtained from :func:`~sionna.phy.mimo.rzf_precoding_matrix`.

#     Parameters
#     ----------
#     resource_grid : :class:`~sionna.phy.ofdm.ResourceGrid`
#         ResourceGrid to be used

#     stream_management : :class:`~sionna.phy.mimo.StreamManagement`
#         StreamManagement to be used 

#     precision : `None` (default) | "single" | "double"
#         Precision used for internal calculations and outputs.
#         If set to `None`,
#         :attr:`~sionna.phy.config.Config.precision` is used.

#     Input
#     -----
#     h : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
#         Actual channel realizations

#     tx_power : [batch_size, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size] (or first n dims), `tf.float32`
#         Power of each stream for each transmitter

#     h_hat : `None` (default) | [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`
#         Channel knowledge based on which the precoding is computed. If set to `None`,
#         the actual channel realizations are used.

#     alpha : `0.` (default) | [batch_size, num_tx, num_ofdm_symbols, fft_size] (or first n dims), `float`
#         Regularization parameter for RZF precoding. If set to `0`, RZF is equivalent
#         to ZF precoding.

#     Output
#     ------
#     h_eff : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `tf.complex`
#         The effective channel after precoding. Nulled subcarriers are
#         automatically removed.
#     """
#     def call(self, h, tx_power, h_hat=None, alpha=0.):
#         """
#         Compute the effective channel after precoding
#         """
#         if h_hat is None:
#             h_hat = h

#         # Get desired channels for precoding
#         # [batch_size, num_tx, num_ofdm_symbols, fft_size,
#         #  ..., num_streams_per_tx, num_tx_ant]
#         h_pc_desired = self.get_desired_channels(h_hat)

#         # Compute precoding matrix
#         #[batch_size, num_tx, num_ofdm_symbols, fft_size]
#         alpha = tf.cast(alpha, self.rdtype)
#         alpha = expand_to_rank(alpha, 4, axis=-1)
#         alpha = tf.broadcast_to(alpha, tf.shape(h_pc_desired)[:4])

#         # [batch_size, num_tx, num_ofdm_symbols, fft_size,
#         #  ..., num_tx_ant,num_streams_per_tx]
#         g = rzf_precoding_matrix(h_pc_desired,
#                                  alpha,
#                                  precision=self.precision)
#         # Apply transmit power to precoding matrix
#         g = self.apply_tx_power(g, tx_power)

#         # Compute effective channel
#         h_eff = self.compute_effective_channel(h, g)

#         return h_eff

class ChannelMatrix(Block):
    def __init__(self,
                 resource_grid,
                 batch_size,
                 num_rx,
                 num_tx,
                 coherence_time = 100,
                 precision=None):
        super().__init__(precision=precision)
        self.resource_grid = resource_grid
        self.coherence_time = coherence_time
        self.batch_size = batch_size
        # Fading autoregressive coefficient initialization
        self.rho_fading = config.tf_rng.uniform([batch_size, num_rx, num_tx],
                                                minval=.95,
                                                maxval=.99,
                                                dtype=self.rdtype)
        # Fading initialization
        self.fading = tf.ones([batch_size, num_rx, num_tx],
                              dtype=self.rdtype)

    def call(self, channel_model):
        """ Generate OFDM channel matrix"""

        # Instantiate the OFDM channel generator
        ofdm_channel = GenerateOFDMChannel(channel_model,
                                           self.resource_grid)

        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
        h_freq = ofdm_channel(self.batch_size)
        return h_freq

    def update(self,
               channel_model,
               h_freq,
               slot):
        """ Update channel matrix every coherence_time slots """
        # Generate new channel realization
        h_freq_new = self.call(channel_model)

        # Change to new channel every coherence_time slots
        change = tf.cast(tf.math.mod(
            slot, self.coherence_time) == 0, self.cdtype)
        h_freq = change * h_freq_new + \
            (tf.cast(1, self.cdtype) - change) * h_freq
        return h_freq

    def apply_fading(self,
                     h_freq):
        """ Apply fading, modeled as an autoregressive process, to channel matrix """
        # Multiplicative fading factor evolving via an AR process
        # [batch_size, num_rx, num_tx]
        self.fading = tf.cast(1, self.rdtype) - self.rho_fading + self.rho_fading * self.fading + \
            config.tf_rng.uniform(
                self.fading.shape, minval=-.1, maxval=.1, dtype=self.rdtype)
        self.fading = tf.maximum(self.fading, tf.cast(0, self.rdtype))
        # [batch_size, num_rx, 1, num_tx, 1, 1, 1]
        fading_expand = insert_dims(self.fading, 1, axis=2)
        fading_expand = insert_dims(fading_expand, 3, axis=4)

        # Channel matrix in the current slot
        h_freq_fading = tf.cast(tf.math.sqrt(
            fading_expand), self.cdtype) * h_freq
        return h_freq_fading
    
# class SystemLevelSimulator(Block):
#     def __init__(self,
#                  batch_size,
#                  num_rings,
#                  num_ut_per_sector,
#                  carrier_frequency,
#                  resource_grid,
#                  scenario,
#                  direction,
#                  ut_array,
#                  bs_array,
#                  bs_max_power_dbm,
#                  ut_max_power_dbm,
#                  coherence_time,
#                  pf_beta=0.98,
#                  max_bs_ut_dist=None,
#                  min_bs_ut_dist=None,
#                  temperature=294,
#                  o2i_model='low',
#                  average_street_width=20.0,
#                  average_building_height=5.0,
#                  precision=None):
#         super().__init__(precision=precision)

#         assert scenario in ['umi', 'uma', 'rma']
#         assert direction in ['uplink', 'downlink']
#         self.scenario = scenario
#         self.batch_size = int(batch_size)
#         self.resource_grid = resource_grid
#         self.direction = direction
#         self.coherence_time = tf.cast(coherence_time, tf.int32)  # [slots]
#         self.num_bs = 1
#         self.num_ut = self.num_bs
#         self.num_ut_ant = ut_array.num_ant
#         self.num_bs_ant = bs_array.num_ant
#         if bs_array.polarization == 'dual':
#             self.num_bs_ant *= 2
#         if self.direction == 'uplink':
#             self.num_tx, self.num_rx = self.num_ut, self.num_bs
#             self.num_tx_ant, self.num_rx_ant = self.num_ut_ant, self.num_bs_ant
#             self.num_tx_per_sector = self.num_ut_per_sector
#         else:
#             self.num_tx, self.num_rx = self.num_bs, self.num_ut
#             self.num_tx_ant, self.num_rx_ant = self.num_bs_ant, self.num_ut_ant
#             self.num_tx_per_sector = 1

#         # Assume 1 stream for UT antenna
#         self.num_streams_per_ut = resource_grid.num_streams_per_tx

#         # Set TX-RX pairs via StreamManagement
#         self.stream_management = get_stream_management(direction,
#                                                        self.num_rx,
#                                                        self.num_tx,
#                                                        self.num_streams_per_ut,
#                                                        num_ut_per_sector)
#         # Noise power per subcarrier
#         self.no = tf.cast(BOLTZMANN_CONSTANT * temperature *
#                           resource_grid.subcarrier_spacing, self.rdtype)

#         # Slot duration [sec]
#         self.slot_duration = resource_grid.ofdm_symbol_duration * \
#             resource_grid.num_ofdm_symbols

#         # Initialize channel model based on scenario
#         self._setup_channel_model(
#             scenario, carrier_frequency, o2i_model, ut_array, bs_array,
#             average_street_width, average_building_height)


#     def _setup_channel_model(self, scenario, carrier_frequency, o2i_model,
#                              ut_array, bs_array, average_street_width,
#                              average_building_height):
#         """ Initialize appropriate channel model based on scenario """
#         common_params = {
#             'carrier_frequency': carrier_frequency,
#             'ut_array': ut_array,
#             'bs_array': bs_array,
#             'direction': self.direction,
#             'enable_pathloss': True,
#             'enable_shadow_fading': True,
#             'precision': self.precision
#         }

#         if scenario == 'umi':  # Urban micro-cell
#             self.channel_model = UMi(o2i_model=o2i_model, **common_params)
#         elif scenario == 'uma':  # Urban macro-cell
#             self.channel_model = UMa(o2i_model=o2i_model, **common_params)
#         elif scenario == 'rma':  # Rural macro-cell
#             self.channel_model = RMa(
#                 average_street_width=average_street_width,
#                 average_building_height=average_building_height,
#                 **common_params)

#     def _group_by_sector(self,
#                          tensor):
#         """ Group tensor by sector
#         - Input: [batch_size, num_ut, num_ofdm_symbols]
#         - Output: [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
#         """
#         tensor = tf.reshape(tensor, [self.batch_size,
#                                      self.num_bs,
#                                      self.num_ut_per_sector,
#                                      self.resource_grid.num_ofdm_symbols])
#         # [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
#         return tf.transpose(tensor, [0, 1, 3, 2])

#     @tf.function(jit_compile=True)
#     def call(self,
#              num_slots,
#              alpha_ul,
#              p0_dbm_ul,
#              bler_target,
#              olla_delta_up,
#              mcs_table_index=1,
#              fairness_dl=0,
#              guaranteed_power_ratio_dl=0.5):

#         # -------------- #
#         # Initialization #
#         # -------------- #
#         # Initialize result history
#         hist = init_result_history(self.batch_size,
#                                    num_slots,
#                                    self.num_bs,
#                                    self.num_ut_per_sector)

#         # Reset OLLA and HARQ/SINR feedback
#         last_harq_feedback, sinr_eff_feedback, num_decoded_bits = \
#             self._reset(bler_target, olla_delta_up)

#         # Initialize channel matrix
#         self.channel_matrix = ChannelMatrix(self.resource_grid,
#                                             self.batch_size,
#                                             self.num_rx,
#                                             self.num_tx,
#                                             self.coherence_time,
#                                             precision=self.precision)
#         # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_sym,
#         #  num_subcarriers]
#         h_freq = self.channel_matrix(self.channel_model)

#         # --------------- #
#         # Simulate a slot #
#         # --------------- #
#         def simulate_slot(slot,
#                           hist,
#                           harq_feedback,
#                           sinr_eff_feedback,
#                           num_decoded_bits,
#                           h_freq):
#             try:
#                 # ------- #
#                 # Channel #
#                 # ------- #
#                 # Update channel matrix
#                 h_freq = self.channel_matrix.update(self.channel_model,
#                                                     h_freq,
#                                                     slot)

#                 # Apply fading
#                 h_freq_fading = self.channel_matrix.apply_fading(h_freq)

#                 # --------- #
#                 # Scheduler #
#                 # --------- #
#                 # Estimate achievable rate
#                 # [batch_size, num_bs, num_ofdm_sym, num_subcarriers, num_ut_per_sector]
#                 rate_achievable_est = estimate_achievable_rate(
#                     self.olla.sinr_eff_db_last,
#                     self.resource_grid.num_ofdm_symbols,
#                     self.resource_grid.fft_size)

#                 # SU-MIMO Proportional Fairness scheduler
#                 # [batch_size, num_bs, num_ofdm_sym, num_subcarriers,
#                 #  num_ut_per_sector, num_streams_per_ut]
#                 is_scheduled = self.scheduler(
#                     num_decoded_bits,
#                     rate_achievable_est)

#                 # N. allocated subcarriers
#                 num_allocated_sc = tf.minimum(tf.reduce_sum(
#                     tf.cast(is_scheduled, tf.int32), axis=-1), 1)
#                 # [batch_size, num_bs, num_ofdm_sym, num_ut_per_sector]
#                 num_allocated_sc = tf.reduce_sum(
#                     num_allocated_sc, axis=-2)

#                 # N. allocated resources per slot
#                 # [batch_size, num_bs, num_ut_per_sector]
#                 num_allocated_re = \
#                     tf.reduce_sum(tf.cast(is_scheduled, tf.int32),
#                                   axis=[-1, -3, -4])

#                 # ------------- #
#                 # Power control #
#                 # ------------- #
#                 # Compute pathloss
#                 # [batch_size, num_rx, num_tx, num_ofdm_symbols], [batch_size, num_ut, num_ofdm_symbols]
#                 pathloss_all_pairs, pathloss_serving_cell = get_pathloss(
#                     h_freq_fading,
#                     rx_tx_association=tf.convert_to_tensor(
#                         self.stream_management.rx_tx_association))
#                 # Group by sector
#                 # [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
#                 pathloss_serving_cell = self._group_by_sector(
#                     pathloss_serving_cell)

#                 if self.direction == 'uplink':
#                     # Open-loop uplink power control
#                     # [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
#                     tx_power_per_ut = open_loop_uplink_power_control(
#                         pathloss_serving_cell,
#                         num_allocated_sc,
#                         alpha=alpha_ul,
#                         p0_dbm=p0_dbm_ul,
#                         ut_max_power_dbm=self.ut_max_power_dbm)
#                 else:
#                     # Channel quality estimation:
#                     # Estimate interference from neighboring base stations
#                     # [batch_size, num_ut, num_ofdm_symbols]

#                     one = tf.cast(1, pathloss_serving_cell.dtype)

#                     # Total received power
#                     # [batch_size, num_ut, num_ofdm_symbols]
#                     rx_power_tot = tf.reduce_sum(
#                         one / pathloss_all_pairs, axis=-2)
#                     # [batch_size, num_bs, num_ut_per_sector, num_ofdm_symbols]
#                     rx_power_tot = self._group_by_sector(rx_power_tot)

#                     # Interference from neighboring base stations
#                     interference_dl = rx_power_tot - one / pathloss_serving_cell
#                     interference_dl *= dbm_to_watt(self.bs_max_power_dbm)

#                     # Fair downlink power allocation
#                     # [batch_size, num_bs, num_ofdm_symbols, num_ut_per_sector]
#                     tx_power_per_ut, _ = downlink_fair_power_control(
#                         pathloss_serving_cell,
#                         interference_dl + self.no,
#                         num_allocated_sc,
#                         bs_max_power_dbm=self.bs_max_power_dbm,
#                         guaranteed_power_ratio=guaranteed_power_ratio_dl,
#                         fairness=fairness_dl,
#                         precision=self.precision)

#                 # For each user, distribute the power uniformly across
#                 # subcarriers and streams
#                 # [batch_size, num_bs, num_tx_per_sector,
#                 #  num_streams_per_tx, num_ofdm_sym, num_subcarriers]
#                 tx_power = spread_across_subcarriers(
#                     tx_power_per_ut,
#                     is_scheduled,
#                     num_tx=self.num_tx_per_sector,
#                     precision=self.precision)

#                 # --------------- #
#                 # Per-stream SINR #
#                 # --------------- #
#                 # [batch_size, num_bs, num_ofdm_sym, num_subcarriers,
#                 #  num_ut_per_sector, num_streams_per_ut]
#                 sinr = get_sinr(tx_power,
#                                 self.stream_management,
#                                 self.no,
#                                 self.direction,
#                                 h_freq_fading,
#                                 self.num_bs,
#                                 self.num_ut_per_sector,
#                                 self.num_streams_per_ut,
#                                 self.resource_grid)

#                 # --------------- #
#                 # Link adaptation #
#                 # --------------- #
#                 # [batch_size, num_bs, num_ut_per_sector]
#                 mcs_index = self.olla(num_allocated_re,
#                                       harq_feedback=harq_feedback,
#                                       sinr_eff=sinr_eff_feedback)

#                 # --------------- #
#                 # PHY abstraction #
#                 # --------------- #
#                 # [batch_size, num_bs, num_ut_per_sector]
#                 num_decoded_bits, harq_feedback, sinr_eff, _, _ = self.phy_abs(
#                     mcs_index,
#                     sinr=sinr,
#                     mcs_table_index=mcs_table_index,
#                     mcs_category=int(self.direction == 'downlink'))

#                 # ------------- #
#                 # SINR feedback #
#                 # ------------- #
#                 # [batch_size, num_bs, num_ut_per_sector]
#                 sinr_eff_feedback = tf.where(num_allocated_re > 0,
#                                              sinr_eff,
#                                              tf.cast(0., self.rdtype))

#                 # Record results
#                 hist = record_results(hist,
#                                       slot,
#                                       sim_failed=False,
#                                       pathloss_serving_cell=tf.reduce_sum(
#                                           pathloss_serving_cell, axis=-2),
#                                       num_allocated_re=num_allocated_re,
#                                       tx_power_per_ut=tf.reduce_sum(
#                                           tx_power_per_ut, axis=-2),
#                                       num_decoded_bits=num_decoded_bits,
#                                       mcs_index=mcs_index,
#                                       harq_feedback=harq_feedback,
#                                       olla_offset=self.olla.offset,
#                                       sinr_eff=sinr_eff,
#                                       pf_metric=self.scheduler.pf_metric)

#             except tf.errors.InvalidArgumentError as e:
#                 print(f"SINR computation did not succeed at slot {slot}.\n"
#                       f"Error message: {e}. Skipping slot...")
#                 hist = record_results(hist, slot,
#                                       shape=[self.batch_size,
#                                              self.num_bs,
#                                              self.num_ut_per_sector], sim_failed=True)

#             # ------------- #
#             # User mobility #
#             # ------------- #
#             self.ut_loc = self.ut_loc + self.ut_velocities * self.slot_duration

#             # Set topology in channel model
#             self.channel_model.set_topology(
#                 self.ut_loc, self.bs_loc, self.ut_orientations,
#                 self.bs_orientations, self.ut_velocities,
#                 self.in_state, self.los, self.bs_virtual_loc)

#             return [slot + 1, hist, harq_feedback, sinr_eff_feedback,
#                     num_decoded_bits, h_freq]

#         # --------------- #
#         # Simulation loop #
#         # --------------- #
#         _, hist, *_ = tf.while_loop(
#             lambda i, *_: i < num_slots,
#             simulate_slot,
#             [0, hist, last_harq_feedback, sinr_eff_feedback,
#              num_decoded_bits, h_freq])

#         for key in hist:
#             hist[key] = hist[key].stack()
#         return hist


# # Communication direction
# direction = 'downlink'  # 'uplink' or 'downlink'

# # 3GPP scenario parameters
# scenario = 'umi'  # 'umi', 'uma' or 'rma'

# # Number of rings of the hexagonal grid
# # With num_rings=1, 7*3=21 base stations are placed
# num_rings = 1

# # N. users per sector
# num_ut_per_sector = 10

# # Max/min distance between base station and served users
# max_bs_ut_dist = 80  # [m]
# min_bs_ut_dist = 0  # [m]

# # Carrier frequency
# carrier_frequency = 3.5e9  # [Hz]

# # Transmit power for base station and user terminals
# bs_max_power_dbm = 56  # [dBm]
# ut_max_power_dbm = 26  # [dBm]

# # Channel is regenerated every coherence_time slots
# coherence_time = 100  # [slots]

# # MCS table index
# # Ranges within [1;4] for downlink and [1;2] for uplink, as in TS 38.214
# mcs_table_index = 1

# # Number of examples
# batch_size = 1

# # Create the antenna arrays at the base stations
# bs_array = PanelArray(num_rows_per_panel=2,
#                       num_cols_per_panel=3,
#                       polarization='dual',
#                       polarization_type='VH',
#                       antenna_pattern='38.901',
#                       carrier_frequency=carrier_frequency)

# # Create the antenna array at the user terminals
# ut_array = PanelArray(num_rows_per_panel=1,
#                       num_cols_per_panel=1,
#                       polarization='single',
#                       polarization_type='V',
#                       antenna_pattern='omni',
#                       carrier_frequency=carrier_frequency)

# # n. OFDM symbols, i.e., time samples, in a slot
# num_ofdm_sym = 1
# # N. available subcarriers
# num_subcarriers = 128
# # Subcarrier spacing, i.e., bandwitdh width of each subcarrier
# subcarrier_spacing = 15e3  # [Hz]

# # Create the OFDM resource grid
# resource_grid = ResourceGrid(num_ofdm_symbols=num_ofdm_sym,
#                              fft_size=num_subcarriers,
#                              subcarrier_spacing=subcarrier_spacing,
#                              num_tx=num_ut_per_sector,
#                              num_streams_per_tx=ut_array.num_ant)

# # Compute SINR
# # Note that downlink is assumed
# precoded_channel = RZFPrecodedChannel(resource_grid=resource_grid, stream_management=stream_management)
# h_eff = precoded_channel(h, tx_power=tx_power, alpha=no)
# lmmse_posteq_sinr = LMMSEPostEqualizationSINR(resource_grid=resource_grid, stream_management=stream_management)
# # [batch_size, num_ofdm_symbols, num_effective_subcarriers, num_rx, num_streams_per_rx]
# sinr = lmmse_posteq_sinr(h_eff, no=no)[0, ...]

import tensorflow as tf

# class MixedMCSBatch:
#     def __init__(self, c_mcs, llrs, bit_mask, mcs_id):
#         # 假设数据已经通过 padding 到相同长度，max_bits 为最大比特数
#         self.max_bits = 8352  # 根据最大 MCS 长度决定
#         self.c_mcs = self.pad(c_mcs)
#         self.llrs = self.pad(llrs, is_llrs=True)
#         self.bit_mask = self.pad(bit_mask)
#         self.mcs_id = mcs_id
        
#         # 获取 batch_size
#         self.batch_size = tf.shape(self.c_mcs)[0]

#         # 生成打乱和逆打乱索引
#         self.perm = tf.random.shuffle(tf.range(self.batch_size))
#         self.inv_perm = tf.argsort(self.perm)

#     def pad(self, data, is_llrs=False):
#         """ 对输入数据进行 padding 处理，确保每个 batch 的数据长度一致 """
#         if is_llrs:
#             # 处理 llrs 数据，填充到最大比特数
#             padded_data = tf.image.resize_with_crop_or_pad(data, target_height=self.max_bits // 6, target_width=self.max_bits)
#         else:
#             # 处理其他数据，填充到最大比特数
#             padded_data = tf.image.resize_with_crop_or_pad(data, target_height=self.max_bits)
#         return padded_data

#     def shuffle(self):
#         """ 使用统一的 permutation 索引打乱数据 """
#         self.c_mcs_shuf = tf.gather(self.c_mcs, self.perm, axis=0)
#         self.llrs_shuf = tf.gather(self.llrs, self.perm, axis=0)
#         self.bit_mask_shuf = tf.gather(self.bit_mask, self.perm, axis=0)
#         self.mcs_id_shuf = tf.gather(self.mcs_id, self.perm, axis=0)

#     def restore(self):
#         """ 使用逆索引恢复数据顺序 """
#         self.c_mcs_restored = tf.gather(self.c_mcs_shuf, self.inv_perm, axis=0)
#         self.llrs_restored = tf.gather(self.llrs_shuf, self.inv_perm, axis=0)
#         self.bit_mask_restored = tf.gather(self.bit_mask_shuf, self.inv_perm, axis=0)
#         self.mcs_id_restored = tf.gather(self.mcs_id_shuf, self.inv_perm, axis=0)

#     def compute_loss(self, llr_hat):
#         """ 计算交叉熵损失（仅计算有效比特） """
#         # 将 llr_hat 的形状调整为与 c_mcs 一致
#         llr_hat_bits = tf.reshape(llr_hat, [self.batch_size, self.max_bits])

#         # 计算交叉熵损失，结合 mask 只计算有效比特
#         loss_per_bit = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.c_mcs_shuf, logits=llr_hat_bits)
#         loss = tf.reduce_sum(loss_per_bit * self.bit_mask_shuf) / tf.reduce_sum(self.bit_mask_shuf)

#         return loss

#     def compute_ber(self, llr_hat):
#         """ 计算 BER（按 MCS 类型分） """
#         llr_hat_bits = tf.reshape(llr_hat, [self.batch_size, self.max_bits])

#         ber_per_bit = tf.cast(tf.not_equal(llr_hat_bits > 0, self.c_mcs_shuf > 0), tf.float32) * self.bit_mask_shuf

#         # 按 MCS 类型计算 BER
#         ber_by_mcs = {}
#         for mcs in [0, 1, 2]:
#             idx = tf.where(self.mcs_id_shuf == mcs)[:, 0]
#             ber_by_mcs[mcs] = tf.reduce_sum(tf.gather(ber_per_bit, idx, axis=0)) / tf.reduce_sum(
#                 tf.gather(self.bit_mask_shuf, idx, axis=0)
#             )
#         return ber_by_mcs

class MixedMCSBatch:
    def __init__(self, tensor, shuffle_indices):
        self.tensor = tensor
        
        # # 获取 batch_size
        # self.batch_size = tf.shape(self.tensor)[0]

        # 生成打乱和逆打乱索引
        # self.perm = tf.random.shuffle(tf.range(self.batch_size))
        self.perm = shuffle_indices
        self.inv_perm = tf.argsort(self.perm)

    def shuffle(self):
        """ 使用统一的 permutation 索引打乱数据 """
        self.tensor_shuf = tf.gather(self.tensor, self.perm, axis=0)

    def restore(self):
        """ 使用逆索引恢复数据顺序 """
        self.tensor_restored = tf.gather(self.tensor_shuf, self.inv_perm, axis=0)


# # 使用示例：
# # 假设 c_mcs, llrs, bit_mask, mcs_id 是你传入的张量

# c_mcs = [tf.random.normal((100, 1, 1, 2784)), tf.random.normal((100, 1, 1, 5568)), tf.random.normal((100, 1, 1, 8352))]
# llrs = [tf.random.normal((100, 1, 1, 1392, 2)), tf.random.normal((100, 1, 1, 1392, 4)), tf.random.normal((100, 1, 1, 1392, 6))]
# bit_mask = tf.random.normal((300, 1, 1, 8352))
# mcs_id = tf.random.uniform((300,), minval=0, maxval=3, dtype=tf.int32)

# # 创建 MixedMCSBatch 实例
# batch = MixedMCSBatch(c_mcs, llrs, bit_mask, mcs_id)

# # 打乱数据
# batch.shuffle()

# # 假设网络输出为 llr_hat
# llr_hat = tf.random.normal((300, 1, 1, 1392, 6))

# # 计算损失
# loss = batch.compute_loss(llr_hat)
# print(f"Loss: {loss.numpy()}")

# # 计算 BER
# ber = batch.compute_ber(llr_hat)
# for mcs, mcs_ber in ber.items():
#     print(f"BER for MCS {mcs}: {mcs_ber.numpy()}")
