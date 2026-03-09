import os
gpu_num = 1 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

print("Current directory:", os.getcwd())
import sys
sys.path.append("./")
import sionna
# Load the required sionna components
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from sionna.channel.tr38901 import Antenna, AntennaArray, CDL, TDL
from sionna.channel import OFDMChannel, ApplyOFDMChannel
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, \
    LMMSEEqualizer, RemoveNulledSubcarriers, ResourceGridDemapper, ZFPrecoder
from sionna.utils import BinarySource, ebnodb2no, insert_dims, flatten_last_dims, expand_to_rank, log10
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils.metrics import compute_ber
from sionna.utils import sim_ber

from keras import Model, Sequential
from keras.layers import Layer, LayerNormalization, Conv2D, Conv2DTranspose, Dense, LSTM, Bidirectional, Embedding
from tensorflow import nn
from math import ceil

from comcloak.training.time_consuming_channel import ChannelMatrix
############################################
## Channel configuration
num_ut = 1
num_bs = 1
num_ut_ant = 8
num_bs_ant = 16
num_streams_per_tx = 2
rx_tx_association = np.array([[1]])

carrier_frequency = 2.6e9 # Hz
delay_spread = 300e-9 # s
cdl_model = "B" # CDL model to use
tdl_model = "B" # TDL model to use
speed = 1.0 # Speed for evaluation and training [m/s]
# SNR range for evaluation and training [dB]
ebno_db_min = 5.0
ebno_db_max = 25.0

############################################
## OFDM waveform configuration
subcarrier_spacing = 30e3 # Hz
fft_size = 128 # Number of subcarriers forming the resource grid, including the null-subcarrier and the guard bands
num_ofdm_symbols = 14 # Number of OFDM symbols forming the resource grid
dc_null = True # Null the DC subcarrier
num_guard_carriers = [5, 6] # Number of guard carriers on each side
pilot_pattern = "kronecker" # Pilot pattern
pilot_ofdm_symbol_indices = [2, 11] # Index of OFDM symbols carrying pilots
cyclic_prefix_length = 0 # Simulation in frequency domain. This is useless

############################################
## Modulation and coding configuration
max_num_bits_per_symbol = 6 # Maximum number of bits per symbol among all MCSs
mcs_list = ["QPSK", "16QAM", "64QAM"] # List of MCS
coderate = 0.5 # Coderate for LDPC code
mcs_dict_order = {
    "QPSK": 2, 
    "16QAM": 4,
    "64QAM": 6
}
mcs_dict = {
    "QPSK": 0, 
    "16QAM": 1,
    "64QAM": 2
}
############################################
## Neural receiver configuration
num_conv_channels = 128 # Number of convolutional channels for the convolutional layers forming the neural receiver
compressed_bits = 1024
num_lstm_layers = 2
lstm_hidden_dim_factor = 8
num_experts = 6 # Number of convolutional kernel experts for the CondConv2D layers
############################################
## Training configuration
num_training_iterations = 2000 # Number of training iterations
training_batch_size = 300 # Training batch size
batch_size_mcs = training_batch_size // len(mcs_list)
# mcs_id = []
# for m in mcs_list:
#     mcs_id.extend([mcs_dict[m]] * batch_size_mcs)
# mcs_id = tf.constant(mcs_id, dtype=tf.int32)

model_weights_path = "./comcloak/training/weights/RX_2str_v0" # Location to save the neural receiver weights once training is done
train_log_path = "./comcloak/training/train_log/nrx_log/RX_2str_v0.txt"
############################################
## Evaluation configuration
results_filename = "Receiver_sys_results" # Location to save the results
############################################


stream_manager = StreamManagement(np.array([[1]]), # Receiver-transmitter association matrix
                                  1)               # One stream per transmitter
resource_grid = ResourceGrid(num_ofdm_symbols = num_ofdm_symbols,
                             fft_size = fft_size,
                             subcarrier_spacing = subcarrier_spacing,
                             num_tx = 1,
                             num_streams_per_tx = num_streams_per_tx,
                             cyclic_prefix_length = cyclic_prefix_length,
                             dc_null = dc_null,
                             pilot_pattern = pilot_pattern,
                             pilot_ofdm_symbol_indices = pilot_ofdm_symbol_indices,
                             num_guard_carriers = num_guard_carriers)

ut_array = AntennaArray(num_rows=1,
                        num_cols=int(num_ut_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)
bs_array = AntennaArray(num_rows=1,
                        num_cols=int(num_bs_ant/2),
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)

class FiLM(Layer):
    def __init__(self, feature_dim, cond_dim):
        super(FiLM, self).__init__()
        # Conditional network：input condition -> gamma, beta
        self.fc_gamma = Dense(feature_dim)
        self.fc_beta = Dense(feature_dim)
    
    def call(self, x, cond):
        # cond: [batch, cond_dim]
        gamma = self.fc_gamma(cond)[:, None, None, :]  # [batch, feature_dim]
        beta = self.fc_beta(cond)[:, None, None, :]    # [batch, feature_dim]
        return gamma * x + beta

class CondConv2D(Layer):
    def __init__(self, filters, kernel_size, num_experts, cond_dim, strides=1, padding="same"):
        super(CondConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else (kernel_size, kernel_size)
        self.num_experts = num_experts
        self.cond_dim = cond_dim
        self.strides = strides
        self.padding = padding.upper()  # "SAME" or "VALID"

        # K kernel experts every single one trainable
        # self.expert_kernels = self.add_weight(
        #     shape=(num_experts, self.kernel_size[0], self.kernel_size[1], None, self.filters),
        #     initializer="glorot_uniform",
        #     trainable=True,
        #     name="expert_kernels"
        # )
        # self.expert_bias = self.add_weight(
        #     shape=(num_experts, self.filters),
        #     initializer="zeros",
        #     trainable=True,
        #     name="expert_bias"
        # )

        # Conditional network -> α (softmax to ensure sum to 1)
        self.alpha_layer = Dense(num_experts, activation="softmax")
        # self.cond_network = Sequential([
        #     Dense(128, activation="relu"),
        #     Dense(64, activation="relu"),
        #     Dense(self.num_experts, activation="softmax")
        # ])
    
    def build(self, input_shape):
        in_channels = input_shape[-1]
        # Now we know the number of input channels, we can fix the shape of the expert convolution kernels
        self.expert_kernels = self.add_weight(
            shape=(self.num_experts, self.kernel_size[0], self.kernel_size[1], in_channels, self.filters),
            initializer="glorot_uniform",
            trainable=True,
            name="expert_kernels"
        )
        self.expert_bias = self.add_weight(
            shape=(self.num_experts, self.filters),
            initializer="zeros",
            trainable=True,
            name="expert_bias"
        )
    
    def call(self, inputs, cond):
        """
        inputs: [batch, H, W, C_in]
        cond:   [batch, cond_dim]
        """
        batch_size = tf.shape(inputs)[0]

        # α coefficients (batch, K)
        alphas = self.alpha_layer(cond)

        # Mixture of convolution kernels (batch, kh, kw, Cin, Cout)
        kernels = tf.einsum("bk,k...->b...", alphas, self.expert_kernels)
        bias = tf.einsum("bk,kf->bf", alphas, self.expert_bias)

        # Convolution for each sample in the batch
        # outputs = []
        # for i in range(batch_size):
        #     out = tf.nn.conv2d(
        #         inputs[i:i+1],
        #         kernels[i],
        #         strides=[1, self.strides, self.strides, 1],
        #         padding=self.padding
        #     )
        #     out = tf.nn.bias_add(out, bias[i])
        #     outputs.append(out)
        outputs = tf.concat(outputs, axis=0)

        outputs = tf.nn.conv2d(
            inputs,
            kernels,
            strides=[1, self.strides, self.strides, 1],
            padding=self.padding
        )
        outputs = tf.nn.bias_add(outputs, bias)
        return outputs

class ResidualBlock(Layer):
    r"""
    This Keras layer implements a convolutional residual block made of two convolutional layers with ReLU activation, layer normalization, and a skip connection.
    The number of convolutional channels of the input must match the number of kernel of the convolutional layers ``num_conv_channel`` for the skip connection to work.

    Input
    ------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
        Input of the layer

    Output
    -------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
        Output of the layer
    """

    def build(self, input_shape):

        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_1 = LayerNormalization(axis=(-1, -2, -3))
        # self._conv_1 = CondConv2D(filters=num_conv_channels, kernel_size=3, num_experts=4, cond_dim=16)
        self._conv_1 = Conv2D(filters=num_conv_channels,
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)
        # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
        self._layer_norm_2 = LayerNormalization(axis=(-1, -2, -3))
        self._conv_2 = Conv2D(filters=num_conv_channels,
                              kernel_size=[3,3],
                              padding='same',
                              activation=None)

    def call(self, inputs):
        z = self._layer_norm_1(inputs)
        z = nn.relu(z)
        z = self._conv_1(z)
        z = self._layer_norm_2(z)
        z = nn.relu(z)
        z = self._conv_2(z) # [batch size, num time samples, num subcarriers, num_channels]
        # Skip connection
        z = z + inputs

        return z

class Receiver_Network(Model):
    r"""
    Keras layer implementing a residual convolutional neural receiver.

    This neural receiver is fed with the post-DFT received samples, forming a resource grid of size num_of_symbols x fft_size, and computes LLRs on the transmitted coded bits.
    These LLRs can then be fed to an outer decoder to reconstruct the information bits.

    As the neural receiver is fed with the entire resource grid, including the guard bands and pilots, it also computes LLRs for these resource elements.
    They must be discarded to only keep the LLRs corresponding to the data-carrying resource elements.

    Input
    ------
    y : [batch size, num rx antenna, num ofdm symbols, num subcarriers], tf.complex
        Received post-DFT samples.

    no : [batch size], tf.float32
        Noise variance. At training, a different noise variance value is sampled for each batch example.

    Output
    -------
    : [batch size, num ofdm symbols, num subcarriers, num_bits_per_symbol]
        LLRs on the transmitted bits.
        LLRs computed for resource elements not carrying data (pilots, guard bands...) must be discarded.
    """

    def build(self, input_shape):

        # Input convolution
        self._input_conv = Conv2D(filters=num_conv_channels,
                                   kernel_size=[3,3],
                                   padding='same',
                                   activation=None)
        # Residual blocks
        self._res_block_1 = ResidualBlock()
        self._res_block_2 = ResidualBlock()
        self._res_block_3 = ResidualBlock()
        self._res_block_4 = ResidualBlock()
        # Output conv
        self._output_conv = CondConv2D(filters=max_num_bits_per_symbol,
                                    kernel_size=[3,3],
                                    num_experts=num_experts,
                                    cond_dim=4)

    def call(self, inputs):
        y, no, labels_shuffled = inputs

        # Feeding the noise power in log10 scale helps with the performance
        no = log10(no)

        # Stacking the real and imaginary components of the different antennas along the 'channel' dimension
        y = tf.transpose(y, [0, 2, 3, 1]) # Putting antenna dimension last
        no = insert_dims(no, 3, 1)
        no = tf.tile(no, [1, y.shape[1], y.shape[2], 1])# Feeding the noise power helps with the performance
        # z : [batch size, num ofdm symbols, num subcarriers, 2*rx_ant + 1]
        z = tf.concat([tf.math.real(y),
                       tf.math.imag(y),
                       no], axis=-1)
        # Input conv
        embedding = Embedding(input_dim=3, output_dim=8)
        # [batch size, cond_dim]
        # mcs_id = []
        # for m in labels_shuffled:
        #     mcs_id.append(mcs_dict[m])
        mcs_id = tf.constant(labels_shuffled, dtype=tf.int32)
        cond_vec = embedding(mcs_id)
        z = self._input_conv(z)
        # [batch size, num ofdm symbols, num subcarriers, num conv channels]
        # Residual blocks
        z = self._res_block_1(z)
        z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        # Output conv
        # [batch_size, num_ofdm_symbols, fft_size, num_bits_per_symbol]
        z = self._output_conv(z, cond_vec)

        return z

class E2ESystem(Model):
    def __init__(self, system, training=False, channel_model="CDL"):
        super().__init__()
        self._system = system
        self._training = training
        self._shuffle_indices = tf.random.shuffle(tf.range(training_batch_size)) 
        self._inv_shuffle = tf.argsort(self._shuffle_indices)
        ######################################
        ## Channel
        # A 3GPP CDL channel model is used
        # cdl = CDL(cdl_model, delay_spread, carrier_frequency,
        #           ut_array, bs_array, "downlink", min_speed=speed)
        if channel_model == "CDL":
            self._channel_model = CDL(cdl_model, delay_spread, carrier_frequency,
                    ut_array, bs_array, "downlink", min_speed=speed)
        elif channel_model == "TDL":
            self._channel_model = TDL(tdl_model, delay_spread, carrier_frequency,
                    num_rx_ant=num_ut_ant, num_tx_ant=num_bs_ant, min_speed=speed)
        
        # self._channel = OFDMChannel(cdl, resource_grid, normalize_channel=True, return_channel=True, precoder=True)
        self._channel = ChannelMatrix(resource_grid, training_batch_size, num_ut, num_bs)
        self._channel_freq = ApplyOFDMChannel(add_awgn=True)
        ######################################
        ## Transmitter
        self._binary_source = BinarySource()

        # if training:
        self._num_bits_per_symbol = [mcs_dict_order[mcs] for mcs in mcs_list]
        self._n, self._k, self._encoder, self._mapper = [], [], [], []
        self._num_mcs_supported = len(mcs_list)
        # for mcs_name, num_bits_per_symbol in zip(mcs_list, self._num_bits_per_symbol):
        for num_bits_per_symbol in self._num_bits_per_symbol:
            n = int(resource_grid.num_data_symbols * num_bits_per_symbol)
            k = int(n * coderate)
            self._n.append(n)
            self._k.append(k)
            self._encoder.append(LDPC5GEncoder(k, n))
            self._mapper.append(Mapper("qam", num_bits_per_symbol))

        self._rg_mapper = ResourceGridMapper(resource_grid)
        self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager)
        self._zf_precoder = ZFPrecoder(resource_grid, stream_manager, return_effective_channel=True)
        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of `system`
        self._demapper, self._decoder = [], []
        for idx in range(len(mcs_list)):
            self._demapper.append(Demapper("app", "qam", self._num_bits_per_symbol[idx])) 
            self._decoder.append(LDPC5GDecoder(self._encoder[idx], hard_out=True))
        if "baseline" in system:
            if system == 'baseline-perfect-csi': # Perfect CSI
                self._removed_null_subc = RemoveNulledSubcarriers(resource_grid)
            elif system == 'baseline-ls-estimation': # LS estimation
                self._ls_est = LSChannelEstimator(resource_grid, interpolation_type="nn")
            # Components required by both baselines
            self._lmmse_equ = LMMSEEqualizer(resource_grid, stream_manager)

        elif system == "neural-receiver": # Neural receiver
            self._receiver_sys = Receiver_Network()
            self._rg_demapper = ResourceGridDemapper(resource_grid, stream_manager) # Used to extract data-carrying resource elements
        
    def call(self, ebno_db):

        # If `ebno_db` is a scalar, a tensor with shape [batch size] is created as it is what is expected by some layers
        
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size_mcs], ebno_db)

        ######################################
        ## Transmitter
        x_mcs, no_mcs, b_mcs, c_mcs = [], [], [], []
        for idx in range(len(mcs_list)):
            no_mcs.append(ebnodb2no(ebno_db, self._num_bits_per_symbol[idx], coderate))
            #     c = self._binary_source([batch_size, 1, 1, n])
            b_mcs.append(self._binary_source([batch_size_mcs, 1, num_streams_per_tx, self._k[idx]]))
            c_mcs.append(self._encoder[idx](b_mcs[idx]))# [batch_size_mcs, 1, num_streams_per_tx, self._n[idx]]
            # Modulation
            x_mcs.append(self._mapper[idx](c_mcs[idx])) # [batch_size_mcs, 1, num_streams_per_tx, n/Constellation.num_bits_per_symbol]
        # b = tf.concat(b_mcs, axis=0) # last dim is different for different mcs, so cannot concat
        # c = tf.concat(c_mcs, axis=0) # same here
        no = tf.concat(no_mcs, axis=0) # [batch size]    
        x = tf.concat(x_mcs, axis=0) # [batch size, 1, num_streams_per_tx, n/Constellation.num_bits_per_symbol]
        # labels = ["QPSK"] * 100 + ["16QAM"] * 100 + ["64QAM"] * 100
        # labels_shuffled = [labels[i] for i in self._shuffle_indices.numpy()]
        labels = tf.convert_to_tensor([0] * batch_size_mcs + [1] * batch_size_mcs + [2] * batch_size_mcs)
        labels_shuffled = tf.gather(labels, self._shuffle_indices, axis=0)
        x_shuffled = tf.gather(x, self._shuffle_indices, axis=0)
        no_shuffled = tf.gather(no, self._shuffle_indices, axis=0)
        # Mapping to the resource grid
        x_rg = self._rg_mapper(x_shuffled)
        
        ######################################
        ## Channel
        # A batch of new channel realizations is sampled and applied at every inference
        no_ = expand_to_rank(no_shuffled, tf.rank(x_rg))

        h_freq = self._channel(self._channel_model)
        x_rg, g = self._zf_precoder([x_rg, h_freq])
        y = self._channel_freq([x_rg, h_freq, no_])
        # y shape: [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
        # y, h = self._channel([x_rg, no_])

        ######################################
        ## Receiver
        # Three options for the receiver depending on the value of ``system``
        if "baseline" in self._system:
            if self._system == 'baseline-perfect-csi':
                h_hat = self._removed_null_subc(h_freq) # Extract non-null subcarriers
                err_var = 0.0 # No channel estimation error when perfect CSI knowledge is assumed
            elif self._system == 'baseline-ls-estimation':
                h_hat, err_var = self._ls_est([y, no_shuffled]) # LS channel estimation with nearest-neighbor
            x_hat_shuffled, no_eff_shuffled = self._lmmse_equ([y, h_hat, err_var, no_shuffled]) # LMMSE equalization x_hat:[batch_size, num_tx, num_streams, num_data_symbols]
            no_eff_shuffled = expand_to_rank(no_eff_shuffled, tf.rank(x_hat_shuffled))
            x_hat = tf.gather(x_hat_shuffled, self._inv_shuffle, axis=0)
            no_eff_ = tf.gather(no_eff_shuffled, self._inv_shuffle, axis=0)
            llrs_ = []
            for idx, mcs in enumerate(mcs_dict_order.keys()):
                x_hat_mcs = tf.gather(x_hat, tf.range(idx*batch_size_mcs, (idx+1)*batch_size_mcs), axis=0)
                no_eff_mcs = tf.gather(no_eff_, tf.range(idx*batch_size_mcs, (idx+1)*batch_size_mcs), axis=0)
                llrs_.append(self._demapper[idx]([x_hat_mcs, no_eff_mcs])) # Demapping
        elif self._system == "neural-receiver":
            # The neural receiver computes LLRs from the frequency domain received symbols and N0
            y = tf.squeeze(y, axis=1)
            # y shape: [batch_size, rx_ant, num_ofdm_symbols, fft_size]
            llrs = self._receiver_sys([y, no_shuffled, labels_shuffled])# Feeding the noise power in log10 scale helps with the performance
            # llrs shape: [batch_size, num_ofdm_symbols, fft_size, num_bits_per_symbol]
            # For traditional equalizer we use demmaper because it returns datasymbols only but not here
            # So we need to use resource grid demapper to extract data-carrying REs
            llrs = insert_dims(llrs, 2, 1)# [batch_size, 1, 1, num_ofdm_symbols, fft_size, num_bits_per_symbol]
            # expect input: [batch_size, num_rx, num_streams_per_rx, num_ofdm_symbols, fft_size, data_dim]
            # output: [batch_size, num_rx, num_streams_per_rx, num_data_symbols, data_dim]
            llrs_shuffled = self._rg_demapper(llrs)
            llrs = tf.gather(llrs_shuffled, self._inv_shuffle, axis=0)
            llrs_ = []
            for idx, mcs in enumerate(mcs_dict_order.keys()):
                bits_per_symbol = mcs_dict_order[mcs]
                llrs_mcs = tf.gather(llrs, tf.range(idx*batch_size_mcs, (idx+1)*batch_size_mcs), axis=0)
                llrs_mcs_bits = tf.gather(llrs_mcs, tf.range(bits_per_symbol), axis=-1)
                llrs_mcs_bits = tf.reshape(llrs_mcs_bits, [batch_size_mcs, 1, 1, -1])
                llrs_.append(llrs_mcs_bits)


            # target_shape = c_mcs[2].shape
            # padded_c_mcs = []
            # original_shapes = [c_mcs[0].shape[-1], c_mcs[1].shape[-1], c_mcs[2].shape[-1]]
            # # padding to regular shape for concatenation
            # for c in c_mcs:
            #     # caculate the amount of padding needed
            #     pad_amount = target_shape[-1] - tf.shape(c)[-1]
            #     padded_c = tf.pad(c, [[0, 0], [0, 0], [0, 0], [0, pad_amount]], "CONSTANT")
            #     padded_c_mcs.append(padded_c)
            # c_mcs_pad = tf.concat(padded_c_mcs, axis=0)
            # # Shuffle the coded bits to match the shuffled inputs
            # c_mcs_shuffled = tf.gather(c_mcs_pad, self._shuffle_indices, axis=0)
            # # Separate the llrs and coded bits for each MCS
            # llrs_,c_mcs_ = [],[]
            # for idx, mcs in enumerate(mcs_dict_order.keys()):
            #     bits_per_symbol = mcs_dict_order[mcs]
            #     llrs_mcs = tf.gather(llrs, self._inv_shuffle, axis=0)
            #     llrs_mcs_bits = tf.gather(llrs_mcs, tf.range(bits_per_symbol), axis=-1)
            #     llrs_.append(llrs_mcs_bits)
            #     c_mcs_gather= tf.gather(c_mcs_shuffled, self._inv_shuffle, axis=0)
            #     c_mcs_.append(c_mcs_gather)
            #     # c_mcs_bits = tf.gather(c_mcs_bits, tf.range(bits_per_symbol)

            # # Restore to original shape
            # restored_c_mcs = []
            # for padded_c, valid_length in zip(c_mcs_, original_shapes):
            #     restored_c = tf.strided_slice(padded_c, [0, 0, 0, 0], [100, 1, 1, valid_length])  
            #     restored_c_mcs.append(restored_c)
            # # llr = insert_dims(llr, 2, 1) # Reshape the input to fit what the resource grid demapper is expected
            # # llr = self._rg_demapper(llr) # Extract data-carrying resource elements. The other LLrs are discarded
            # # llr = torch.reshape(llr, [batch_size, 1, 1, n]) # Reshape the LLRs to fit what the outer decoder is expected

        # Outer coding is not needed if the information rate is returned
        if self._training:
            # Compute and return BMD rate (in bit), which is known to be an achievable
            # information rate for BICM systems.
            # Training aims at maximizing the BMD rate
            losses = []
            for idx in range(len(c_mcs)):
                # get the restored data and llrs for the current MCS
                restored_data = c_mcs[idx]  # original restored data
                llrs = llrs_[idx]  # corresponding LLR output
                
                # llrs_flat = tf.reshape(llrs, restored_data.shape)  # ensure shapes match
                
                # compute loss: calculate cross-entropy loss
                # llrs are model outputs (logits), we can directly use tf.nn.sigmoid_cross_entropy_with_logits to compute
                # but need to ensure shapes match
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=restored_data, logits=llrs)
                )
                
                # add the current loss to the losses list
                losses.append(loss)

            # compute the average loss across all MCS types
            total_loss = tf.reduce_mean(losses)
            
            # bce = tf.nn.sigmoid_cross_entropy_with_logits(c_mcs, llr)
            # bce = tf.reduce_mean(bce)
            # loss = -tf.constant(1.0, tf.float32) - bce/tf.math.log(2.)
            # loss = F.mse_loss(H_hat_norm, H_norm)
            # loss = torch.nn.functional.smooth_l1_loss(h_hat, h_ri)
            # loss = torch.mean(torch.abs(h_ri - h_hat)**2)

            return total_loss
        else:
            # Outer decoding
            b_hat = []
            for idx in range(len(llrs_)):
                llrs = llrs_[idx]
                b_hat.append(self._decoder[idx](llrs))
            return b_mcs, b_hat # Ground truth and reconstructed information bits returned for BER/BLER computation
      
# The end-to-end system equipped with the neural receiver is instantiated for training.
# When called, it therefore returns the estimated BMD rate
# model = E2ESystem('neural-receiver', training=True)
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# for i in range(num_training_iterations):
#     # Sampling a batch of SNRs
#     ebno_db = tf.random.uniform(shape=[batch_size_mcs], minval=ebno_db_min, maxval=ebno_db_max)
#     # Forward pass
#     with tf.GradientTape() as tape:
#         loss = model(ebno_db)
#         # Tensorflow optimizers only know how to minimize loss function.
#         # Therefore, a loss function is defined as the additive inverse of the BMD rate
#     # Computing and applying gradients
#     weights = model.trainable_weights
#     grads = tape.gradient(loss, weights)
#     optimizer.apply_gradients(zip(grads, weights))
#     # Periodically printing the progress
#     if i % 5 == 0:
#         print('Iteration {}/{}  Loss: {:.4f} bit'.format(i, num_training_iterations, loss.numpy()), end='\r')
#         with open(train_log_path, "a") as f:
#             # f.write(f"{i},{ebno_db:.2f},{loss.numpy():.6f}\n")
#             f.write(f"{i},{loss.numpy():.6f}\n")
# # Save the weights in a file
# weights = model.get_weights()
# with open(model_weights_path, 'wb') as f:
#     pickle.dump(weights, f)


# model = E2ESystem('neural-receiver')
model = E2ESystem('baseline-ls-estimation')
# BLER = {}
# # Range of SNRs over which the systems are evaluated
# ebno_dbs = tf.range(-15.0, # Min SNR for evaluation
#                      5.0, # Max SNR for evaluation
#                      1.0) # Step
# # Run one inference to build the layers and loading the weights
model(tf.constant(10.0, tf.float32))
# with open(model_weights_path, 'rb') as f:
#     weights = pickle.load(f)
# model.set_weights(weights)

# total_params = model.count_params()
# print(f"Total parameters: {total_params}")

# for ebno_db in ebno_dbs:
#     b_mcs,b_hat = model(ebno_db)
#     BER1 = tf.reduce_mean(tf.cast(tf.not_equal(b_mcs[0], b_hat[0]), tf.float32)).numpy()
#     BER2 = tf.reduce_mean(tf.cast(tf.not_equal(b_mcs[1], b_hat[1]), tf.float32)).numpy()
#     BER3 = tf.reduce_mean(tf.cast(tf.not_equal(b_mcs[2], b_hat[2]), tf.float32)).numpy()
#     print(f"SNR: {ebno_db.numpy()} dB, BER1: {BER1}, BER2: {BER2}, BER3: {BER3}")


# ebno_dbs_list = []
# ber1_list = []
# ber2_list = []
# ber3_list = []


# for ebno_db in ebno_dbs:
#     b_mcs, b_hat = model(ebno_db)
    

#     BER1 = tf.reduce_mean(tf.cast(tf.not_equal(b_mcs[0], b_hat[0]), tf.float32)).numpy()
#     BER2 = tf.reduce_mean(tf.cast(tf.not_equal(b_mcs[1], b_hat[1]), tf.float32)).numpy()
#     BER3 = tf.reduce_mean(tf.cast(tf.not_equal(b_mcs[2], b_hat[2]), tf.float32)).numpy()
    

#     ebno_dbs_list.append(ebno_db.numpy())
#     ber1_list.append(BER1)
#     ber2_list.append(BER2)
#     ber3_list.append(BER3)


# plt.figure(figsize=(10, 6))


# plt.plot(ebno_dbs_list, ber1_list, marker='o', color='b', label='BER1')
# plt.plot(ebno_dbs_list, ber2_list, marker='s', color='g', label='BER2')
# plt.plot(ebno_dbs_list, ber3_list, marker='^', color='r', label='BER3')


# plt.xlabel('SNR (dB)')
# plt.ylabel('BER')
# plt.title('BER vs SNR for BER1, BER2, and BER3')
# plt.legend()


# plt.grid(True)
# plt.savefig('ber_vs_snr.png')  
# plt.close()
