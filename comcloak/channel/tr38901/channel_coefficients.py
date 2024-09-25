
"""
Class for sampling channel impulse responses following 3GPP TR38.901
specifications and giving LSPs and rays.
"""

import torch

from comcloak import PI, SPEED_OF_LIGHT
from torch import sin, cos, acos
import numpy as np

def scatter_nd_torch(indices, updates, shape):
    # 创建一个初始张量
    output = torch.zeros(shape, dtype=updates.dtype)
    
    # 将 indices 展开为一维，计算对应的线性索引
    flat_indices = torch.zeros(indices.shape[0], dtype=torch.long)
    stride = 1
    for i in reversed(range(indices.shape[1])):
        flat_indices += indices[:, i] * stride
        stride *= shape[i]
    
    # 使用 scatter_ 方法更新平坦输出
    for i, idx in enumerate(flat_indices):
        output[idx] = updates[i]

    return output

def gather_pytorch(input_data, indices=None, batch_dims=0, dim=0):
    input_data = torch.tensor(input_data)
    indices = torch.tensor(indices)
    if batch_dims == 0:
        if dim < 0:
            dim = len(input_data.shape) + dim
        data = torch.index_select(input_data, dim, indices.flatten())
        shape_input = list(input_data.shape)
        # shape_ = delete(shape_input, dim)
        # 连接列表
        shape_output = shape_input[:dim] + \
            list(indices.shape) + shape_input[dim + 1:]
        data_output = data.reshape(shape_output)
        return data_output
    else:
        data_output = []
        for data,ind in zip(input_data, indices):
            r = gather_pytorch(data, ind, batch_dims=batch_dims-1)
            data_output.append(r)
        return torch.stack(data_output)

class Topology:
    # pylint: disable=line-too-long
    r"""
    Class for conveniently storing the network topology information required
    for sampling channel impulse responses

    Parameters
    -----------

    velocities : [batch size, number of UTs], torch.float
        UT velocities

    moving_end : str
        Indicated which end of the channel (TX or RX) is moving. Either "tx" or
        "rx".

    los_aoa : [batch size, number of BSs, number of UTs], torch.float
        Azimuth angle of arrival of LoS path [radian]

    los_aod : [batch size, number of BSs, number of UTs], torch.float
        Azimuth angle of departure of LoS path [radian]

    los_zoa : [batch size, number of BSs, number of UTs], torch.float
        Zenith angle of arrival for of path [radian]

    los_zod : [batch size, number of BSs, number of UTs], torch.float
        Zenith angle of departure for of path [radian]

    los : [batch size, number of BSs, number of UTs], torch.bool
        Indicate for each BS-UT link if it is in LoS

    distance_3d : [batch size, number of UTs, number of UTs], torch.float
        Distance between the UTs in X-Y-Z space (not only X-Y plan).

    tx_orientations : [batch size, number of TXs, 3], torch.float
        Orientations of the transmitters, which are either BSs or UTs depending
        on the link direction [radian].

    rx_orientations : [batch size, number of RXs, 3], torch.float
        Orientations of the receivers, which are either BSs or UTs depending on
        the link direction [radian].
    """

    def __init__(self,  velocities,
                        moving_end,
                        los_aoa,
                        los_aod,
                        los_zoa,
                        los_zod,
                        los,
                        distance_3d,
                        tx_orientations,
                        rx_orientations):
        self.velocities = velocities
        self.moving_end = moving_end
        self.los_aoa = los_aoa
        self.los_aod = los_aod
        self.los_zoa = los_zoa
        self.los_zod = los_zod
        self.los = los
        self.tx_orientations = tx_orientations
        self.rx_orientations = rx_orientations
        self.distance_3d = distance_3d


class ChannelCoefficientsGenerator:
    # pylint: disable=line-too-long
    r"""
    Sample channel impulse responses according to LSPs rays.

    This class implements steps 10 and 11 from the TR 38.901 specifications,
    (section 7.5).

    Parameters
    ----------
    carrier_frequency : float
        Carrier frequency [Hz]

    tx_array : PanelArray
        Panel array used by the transmitters.
        All transmitters share the same antenna array configuration.

    rx_array : PanalArray
        Panel array used by the receivers.
        All transmitters share the same antenna array configuration.

    subclustering : bool
        Use subclustering if set to `True` (see step 11 for section 7.5 in
        TR 38.901). CDL does not use subclustering. System level models (UMa,
        UMi, RMa) do.

    dtype : Complex torch.DType
        Defines the datatype for internal calculations and the output
        dtype. Defaults to `torch.complex64`.

    Input
    -----
    num_time_samples : int
        Number of samples

    sampling_frequency : float
        Sampling frequency [Hz]

    k_factor : [batch_size, number of TX, number of RX]
        K-factor

    rays : Rays
        Rays from which to compute thr CIR

    topology : Topology
        Topology of the network

    c_ds : [batch size, number of TX, number of RX]
        Cluster DS [ns]. Only needed when subclustering is used
        (``subclustering`` set to `True`), i.e., with system level models.
        Otherwise can be set to None.
        Defaults to None.

    debug : bool
        If set to `True`, additional information is returned in addition to
        paths coefficients and delays: The random phase shifts (see step 10 of
        section 7.5 in TR38.901 specification), and the time steps at which the
        channel is sampled.

    Output
    ------
    h : [batch size, num TX, num RX, num paths, num RX antenna, num TX antenna, num samples], torch.complex
        Paths coefficients

    delays : [batch size, num TX, num RX, num paths], torch.real
        Paths delays [s]

    phi : [batch size, number of BSs, number of UTs, 4], torch.real
        Initial phases (see step 10 of section 7.5 in TR 38.901 specification).
        Last dimension corresponds to the four polarization combinations.

    sample_times : [number of time steps], torch.float
        Sampling time steps
    """

    def __init__(self,  carrier_frequency,
                        tx_array, rx_array,
                        subclustering,
                        dtype=torch.complex64):
        assert dtype.is_complex, "dtype must be a complex datatype"
        self._dtype = dtype
        
        if dtype == torch.complex32:
            real_dtype = torch.float16
        elif dtype == torch.complex64:
            real_dtype = torch.float32
        elif dtype == torch.complex128:
            real_dtype = torch.float64
        else:
            raise TypeError("real_dtype must be in [torch.float16,torch.float32,torch.float64]")
        
        self._real_dtype = real_dtype
        
        # Wavelength (m)
        self._lambda_0 = torch.tensor(SPEED_OF_LIGHT/carrier_frequency,dtype=real_dtype)
        self._tx_array = tx_array
        self._rx_array = rx_array
        self._subclustering = subclustering

        # Sub-cluster information for intra cluster delay spread clusters
        # This is hardcoded from Table 7.5-5
        self._sub_cl_1_ind = torch.tensor([0,1,2,3,4,5,6,7,18,19], dtype=torch.int32)
        self._sub_cl_2_ind = torch.tensor([8,9,10,11,16,17], dtype=torch.int32)
        self._sub_cl_3_ind = torch.tensor([12,13,14,15], dtype=torch.int32)
        self._sub_cl_delay_offsets = torch.tensor([0, 1.28, 2.56],
                                                    dtype=real_dtype)
        

    def __call__(self, num_time_samples, sampling_frequency, k_factor, rays,
                 topology, c_ds=None, debug=False):
        # Sample times
        sample_times = (torch.arange(num_time_samples,
                dtype=self._real_dtype)/sampling_frequency)

        if not isinstance(rays.aoa,torch.Tensor):
            rays_aoa =torch.tensor(rays.aoa)
        else:
            rays_aoa = rays.aoa
        # Step 10
        phi = self._step_10(rays_aoa.shape)
        # print('phi:------',phi)
        # Step 11
        h, delays = self._step_11(phi, topology, k_factor, rays, sample_times,
                                                                        c_ds)

        # Return additional information if requested
        if debug:
            return h, delays, phi, sample_times

        return h, delays

    ###########################################
    # Utility functions
    ###########################################

    def _unit_sphere_vector(self, theta, phi):
        r"""
        Generate vector on unit sphere (7.1-6)

        Input
        -------
        theta : Arbitrary shape, torch.float
            Zenith [radian]

        phi : Same shape as ``theta``, torch.float
            Azimuth [radian]

        Output
        --------
        rho_hat : ``phi.shape`` + [3, 1]
            Vector on unit sphere

        """
        rho_hat = torch.stack([sin(theta)*cos(phi),
                            sin(theta)*sin(phi),
                            cos(theta)], dim=-1)
        return torch.unsqueeze(rho_hat, dim=-1)

    def _forward_rotation_matrix(self, orientations):
        r"""
        Forward composite rotation matrix (7.1-4)

        Input
        ------
            orientations : [...,3], torch.float
                Orientation to which to rotate [radian]

        Output
        -------
        R : [...,3,3], torch.float
            Rotation matrix
        """
        a, b, c = orientations[...,0], orientations[...,1], orientations[...,2]

        row_1 = torch.stack([cos(a)*cos(b),
            cos(a)*sin(b)*sin(c)-sin(a)*cos(c),
            cos(a)*sin(b)*cos(c)+sin(a)*sin(c)], dim=-1)

        row_2 = torch.stack([sin(a)*cos(b),
            sin(a)*sin(b)*sin(c)+cos(a)*cos(c),
            sin(a)*sin(b)*cos(c)-cos(a)*sin(c)], dim=-1)

        row_3 = torch.stack([-sin(b),
            cos(b)*sin(c),
            cos(b)*cos(c)], dim=-1)

        rot_mat = torch.stack([row_1, row_2, row_3], dim=-2)
        return rot_mat

    def _rot_pos(self, orientations, positions):
        r"""
        Rotate the ``positions`` according to the ``orientations``

        Input
        ------
        orientations : [...,3], torch.float
            Orientation to which to rotate [radian]

        positions : [...,3,1], torch.float
            Positions to rotate

        Output
        -------
        : [...,3,1], torch.float
            Rotated positions
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        return torch.matmul(rot_mat, positions)

    def _reverse_rotation_matrix(self, orientations):
        r"""
        Reverse composite rotation matrix (7.1-4)

        Input
        ------
        orientations : [...,3], torch.float
            Orientations to rotate to  [radian]

        Output
        -------
        R_inv : [...,3,3], torch.float
            Inverse of the rotation matrix corresponding to ``orientations``
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        rot_mat_inv = torch.transpose(rot_mat,-2,-1)
        return rot_mat_inv

    def _gcs_to_lcs(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute the angles ``theta``, ``phi`` in LCS rotated according to
        ``orientations`` (7.1-7/8)

        Input
        ------
        orientations : [...,3] of rank K, torch.float
            Orientations to which to rotate to [radian]

        theta : Broadcastable to the first K-1 dimensions of ``orientations``, torch.float
            Zenith to rotate [radian]

        phi : Same dimension as ``theta``, torch.float
            Azimuth to rotate [radian]

        Output
        -------
        theta_prime : Same dimension as ``theta``, torch.float
            Rotated zenith

        phi_prime : Same dimensions as ``theta`` and ``phi``, torch.float
            Rotated azimuth
        """

        rho_hat = self._unit_sphere_vector(theta, phi)
        rot_inv = self._reverse_rotation_matrix(orientations)
        rot_rho = torch.matmul(rot_inv, rho_hat)
        v1 = torch.tensor([0,0,1], dtype=self._real_dtype)
        v1 = torch.reshape(v1, [1]*(rot_rho.dim()-1)+[3])
        v2 = torch.tensor([1+0j,1j,0], dtype=self._dtype)
        v2 = torch.reshape(v2, [1]*(rot_rho.dim()-1)+[3])
        z = torch.matmul(v1, rot_rho)
        z = torch.clamp(z, torch.tensor(-1., dtype=self._real_dtype),
                             torch.tensor(1., dtype=self._real_dtype))
        theta_prime = acos(z)
        phi_prime = torch.angle((torch.matmul(v2, rot_rho.to(dtype=self._dtype))))
        theta_prime = torch.squeeze(theta_prime, dim=[phi.dim(),
            phi.dim()+1])
        phi_prime = torch.squeeze(phi_prime, dim=[phi.dim(),
            phi.dim()+1])

        return (theta_prime, phi_prime)
    
    def _compute_psi(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute displacement angle :math:`Psi` for the transformation of LCS-GCS
        field components in (7.1-15) of TR38.901 specification

        Input
        ------
        orientations : [...,3], torch.float
            Orientations to which to rotate to [radian]

        theta :  Broadcastable to the first K-1 dimensions of ``orientations``, torch.float
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, torch.float
            Spherical position azimuth [radian]

        Output
        -------
            Psi : Same shape as ``theta`` and ``phi``, torch.float
                Displacement angle :math:`Psi`
        """
        a = orientations[...,0]
        b = orientations[...,1]
        c = orientations[...,2]
        real = sin(c)*cos(theta)*sin(phi-a)
        real += cos(c)*(cos(b)*sin(theta)-sin(b)*cos(theta)*cos(phi-a))
        imag = sin(c)*cos(phi-a) + sin(b)*cos(c)*sin(phi-a)
        psi = torch.angle(torch.complex(real, imag))
        return psi

    def _l2g_response(self, f_prime, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Transform field components from LCS to GCS (7.1-11)

        Input
        ------
        f_prime : K-Dim Tensor of shape [...,2], torch.float
            Field components

        orientations : K-Dim Tensor of shape [...,3], torch.float
            Orientations of LCS-GCS [radian]

        theta : K-1-Dim Tensor with matching dimensions to ``f_prime`` and ``phi``, torch.float
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, torch.float
            Spherical position azimuth [radian]

        Output
        ------
            F : K+1-Dim Tensor with shape [...,2,1], torch.float
                The first K dimensions are identical to those of ``f_prime``
        """
        psi = self._compute_psi(orientations, theta, phi)
        row1 = torch.stack([cos(psi), -sin(psi)], dim=-1)
        row2 = torch.stack([sin(psi), cos(psi)], dim=-1)
        mat = torch.stack([row1, row2], dim=-2)
        f = torch.matmul(mat, torch.unsqueeze(f_prime, -1))
        return f

    def _step_11_get_tx_antenna_positions(self, topology):
        r"""Compute d_bar_tx in (7.5-22), i.e., the positions in GCS of elements
        forming the transmit panel

        Input
        -----
        topology : Topology
            Topology of the network

        Output
        -------
        d_bar_tx : [batch_size, num TXs, num TX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get BS orientations got broadcasting
        tx_orientations = topology.tx_orientations
        tx_orientations = torch.unsqueeze(tx_orientations, dim = 2)

        # Get antenna element positions in LCS and reshape for broadcasting
        tx_ant_pos_lcs = self._tx_array.ant_pos
        tx_ant_pos_lcs = torch.reshape(tx_ant_pos_lcs,
            [1,1]+list(tx_ant_pos_lcs.shape)+[1])

        # Compute antenna element positions in GCS
        tx_ant_pos_gcs = self._rot_pos(tx_orientations, tx_ant_pos_lcs)
        tx_ant_pos_gcs = torch.reshape(tx_ant_pos_gcs,
            tx_ant_pos_gcs.shape[:-1])

        d_bar_tx = tx_ant_pos_gcs

        return d_bar_tx

    def _step_11_get_rx_antenna_positions(self, topology):
        r"""Compute d_bar_rx in (7.5-22), i.e., the positions in GCS of elements
        forming the receive antenna panel

        Input
        -----
        topology : Topology
            Topology of the network

        Output
        -------
        d_bar_rx : [batch_size, num RXs, num RX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get UT orientations got broadcasting
        rx_orientations = topology.rx_orientations
        rx_orientations = torch.unsqueeze(rx_orientations, 2)

        # Get antenna element positions in LCS and reshape for broadcasting
        rx_ant_pos_lcs = self._rx_array.ant_pos
        rx_ant_pos_lcs = torch.reshape(rx_ant_pos_lcs,
            [1,1]+list(rx_ant_pos_lcs.shape)+[1])

        # Compute antenna element positions in GCS
        rx_ant_pos_gcs = self._rot_pos(rx_orientations, rx_ant_pos_lcs)
        rx_ant_pos_gcs = torch.reshape(rx_ant_pos_gcs,
            rx_ant_pos_gcs.shape[:-1])

        d_bar_rx = rx_ant_pos_gcs

        return d_bar_rx

    def _step_10(self, shape):
        r"""
        Generate random and uniformly distributed phases for all rays and
        polarization combinations

        Input
        -----
        shape : Shape tensor
            Shape of the leading dimensions for the tensor of phases to generate

        Output
        ------
        phi : [shape] + [4], torch.float
            Phases for all polarization combinations
        """
        shape_tensor = torch.tensor(shape, dtype=torch.int64)
        phi_shape = torch.cat([shape_tensor, torch.tensor([4])], dim=0)
        # phi = (torch.rand(tuple(phi_shape.tolist()), dtype=self._real_dtype) * 2 * PI) - PI
        phi = np.random.uniform(low=-PI, high=PI, size=phi_shape.tolist())
        phi = torch.tensor(phi, dtype=torch.float32)
        return phi

    def _step_11_phase_matrix(self, phi, rays):
        # pylint: disable=line-too-long
        r"""
        Compute matrix with random phases in (7.5-22)

        Input
        -----
        phi : [batch size, num TXs, num RXs, num clusters, num rays, 4], torch.float
            Initial phases for all combinations of polarization

        rays : Rays
            Rays

        Output
        ------
        h_phase : [batch size, num TXs, num RXs, num clusters, num rays, 2, 2], torch.complex
            Matrix with random phases in (7.5-22)
        """
        xpr = rays.xpr
        print('xpr:-----',xpr)

        xpr_scaling = torch.complex(torch.sqrt(1/xpr.to(self._real_dtype)),
            torch.tensor(0., dtype=self._real_dtype))
        e0 = torch.exp(torch.complex(torch.tensor(0., dtype=self._real_dtype),
            phi[...,0]))
        e3 = torch.exp(torch.complex(torch.tensor(0., dtype=self._real_dtype),
            phi[...,3]))
        e1 = xpr_scaling*torch.exp(torch.complex(torch.tensor(0.,
                                dtype=self._real_dtype), phi[...,1]))
        e2 = xpr_scaling*torch.exp(torch.complex(torch.tensor(0.,
                                dtype=self._real_dtype), phi[...,2]))
        shape = torch.cat([torch.tensor(e0.shape), torch.tensor([2,2])], dim=-1).tolist()
        h_phase = torch.reshape(torch.stack([e0, e1, e2, e3], dim=-1), tuple(shape))

        return h_phase
    
    def _step_11_doppler_matrix(self, topology, aoa, zoa, t):
        # pylint: disable=line-too-long
        r"""
        Compute matrix with phase shifts due to mobility in (7.5-22)

        Input
        -----
        topology : Topology
            Topology of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], torch.float
            Azimuth angles of arrivals [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], torch.float
            Zenith angles of arrivals [radian]

        t : [number of time steps]
            Time steps at which the channel is sampled

        Output
        ------
        h_doppler : [batch size, num_tx, num rx, num clusters, num rays, num time steps], torch.complex
            Matrix with phase shifts due to mobility in (7.5-22)
        """
        lambda_0 = self._lambda_0
        velocities = topology.velocities

        # Add an extra dimension to make v_bar broadcastable with the time
        # dimension
        # v_bar [batch size, num tx or num rx, 3, 1]
        v_bar = velocities
        v_bar = torch.unsqueeze(v_bar, dim=-1)

        # Depending on which end of the channel is moving, tx or rx, we add an
        # extra dimension to make this tensor broadcastable with the other end
        if topology.moving_end == 'rx':
            # v_bar [batch size, 1, num rx, num tx, 1]
            v_bar = torch.unsqueeze(v_bar, 1)
        elif topology.moving_end == 'tx':
            # v_bar [batch size, num tx, 1, num tx, 1]
            v_bar = torch.unsqueeze(v_bar, 2)

        # v_bar [batch size, 1, num rx, 1, 1, 3, 1]
        # or    [batch size, num tx, 1, 1, 1, 3, 1]
        v_bar = torch.unsqueeze(torch.unsqueeze(v_bar, -3), -3)

        # v_bar [batch size, num_tx, num rx, num clusters, num rays, 3, 1]
        r_hat_rx = self._unit_sphere_vector(zoa, aoa)

        # Compute phase shift due to doppler
        # [batch size, num_tx, num rx, num clusters, num rays, num time steps]
        exponent = 2*PI/lambda_0*torch.sum(r_hat_rx*v_bar, -2)*t
        h_doppler = torch.exp(torch.complex(torch.tensor(0.,
                                    dtype=self._real_dtype), exponent))

        # [batch size, num_tx, num rx, num clusters, num rays, num time steps]
        return h_doppler
    
    def _step_11_array_offsets(self, topology, aoa, aod, zoa, zod):
        # pylint: disable=line-too-long
        r"""
        Compute matrix accounting for phases offsets between antenna elements

        Input
        -----
        topology : Topology
            Topology of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], torch.float
            Azimuth angles of arrivals [radian]

        aod : [batch size, num TXs, num RXs, num clusters, num rays], torch.float
            Azimuth angles of departure [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], torch.float
            Zenith angles of arrivals [radian]

        zod : [batch size, num TXs, num RXs, num clusters, num rays], torch.float
            Zenith angles of departure [radian]
        Output
        ------
        h_array : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas], torch.complex
            Matrix accounting for phases offsets between antenna elements
        """

        lambda_0 = self._lambda_0

        r_hat_rx = self._unit_sphere_vector(zoa, aoa)
        r_hat_rx = torch.squeeze(r_hat_rx, dim=len(r_hat_rx.shape)-1)
        r_hat_tx = self._unit_sphere_vector(zod, aod)
        r_hat_tx = torch.squeeze(r_hat_tx, dim=len(r_hat_tx.shape)-1)
        d_bar_rx = self._step_11_get_rx_antenna_positions(topology)
        d_bar_tx = self._step_11_get_tx_antenna_positions(topology)

        # Reshape tensors for broadcasting
        # r_hat_rx/tx have
        # shape [batch_size, num_tx, num_rx, num_clusters, num_rays,    3]
        # and will be reshaoed to
        # [batch_size, num_tx, num_rx, num_clusters, num_rays, 1, 3]
        r_hat_tx = torch.unsqueeze(r_hat_tx, -2)
        r_hat_rx = torch.unsqueeze(r_hat_rx, -2)

        # d_bar_tx has shape [batch_size, num_tx,          num_tx_antennas, 3]
        # and will be reshaped to
        # [batch_size, num_tx, 1, 1, 1, num_tx_antennas, 3]
        s = d_bar_tx.shape
        shape = list(s[:2]) + [1, 1, 1] + list(s[2:])
        d_bar_tx = torch.reshape(d_bar_tx, tuple(shape))

        # d_bar_rx has shape [batch_size,    num_rx,       num_rx_antennas, 3]
        # and will be reshaped to
        # [batch_size, 1, num_rx, 1, 1, num_rx_antennas, 3]
        s = d_bar_rx.shape
        shape = [s[0], 1, s[1], 1, 1] + list(s[2:])
        d_bar_rx = torch.reshape(d_bar_rx, shape)

        # Compute all tensor elements

        # As broadcasting of such high-rank tensors is not fully supported
        # in all cases, we need to do a hack here by explicitly
        # broadcasting one dimension:
        s = d_bar_rx.shape
        shape = [s[0], r_hat_rx.shape[1]] + list(s[2:])
        d_bar_rx = torch.broadcast_to(d_bar_rx, shape)
        exp_rx = 2*PI/lambda_0*torch.sum(r_hat_rx*d_bar_rx,
            dim=-1, keepdims=True)
        exp_rx = torch.exp(torch.complex(torch.tensor(0.,
                                    dtype=self._real_dtype), exp_rx))

        # The hack is for some reason not needed for this term
        # exp_tx = 2*PI/lambda_0*torch.sum(r_hat_tx*d_bar_tx,
        #     dim=-1, keepdims=True)
        exp_tx = 2*PI/lambda_0*torch.sum(r_hat_tx*d_bar_tx,
            dim=-1)
        exp_tx = torch.exp(torch.complex(torch.tensor(0.,
                                    dtype=self._real_dtype), exp_tx))
        exp_tx = torch.unsqueeze(exp_tx, -2)

        h_array = exp_rx*exp_tx

        return h_array
    
    def _step_11_field_matrix(self, topology, aoa, aod, zoa, zod, h_phase):
        # pylint: disable=line-too-long
        r"""
        Compute matrix accounting for the element responses, random phases
        and xpr

        Input
        -----
        topology : Topology
            Topology of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], torch.float
            Azimuth angles of arrivals [radian]

        aod : [batch size, num TXs, num RXs, num clusters, num rays], torch.float
            Azimuth angles of departure [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], torch.float
            Zenith angles of arrivals [radian]

        zod : [batch size, num TXs, num RXs, num clusters, num rays], torch.float
            Zenith angles of departure [radian]

        h_phase : [batch size, num_tx, num rx, num clusters, num rays, num time steps], torch.complex
            Matrix with phase shifts due to mobility in (7.5-22)

        Output
        ------
        h_field : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas], torch.complex
            Matrix accounting for element responses, random phases and xpr
        """

        tx_orientations = topology.tx_orientations
        rx_orientations = topology.rx_orientations

        # Transform departure angles to the LCS
        s = tx_orientations.shape
        shape = torch.cat([torch.tensor(s[:2]), torch.tensor([1,1,1,s[-1]])], 0).tolist()
        tx_orientations = torch.reshape(tx_orientations, tuple(shape))
        zod_prime, aod_prime = self._gcs_to_lcs(tx_orientations, zod, aod)

        # Transform arrival angles to the LCS
        s = rx_orientations.shape
        shape = torch.cat([torch.tensor([s[0],1]),torch.tensor([s[1],1,1,s[-1]])], 0).tolist()
        rx_orientations = torch.reshape(rx_orientations, tuple(shape))
        zoa_prime, aoa_prime = self._gcs_to_lcs(rx_orientations, zoa, aoa)

        # Compute transmitted and received field strength for all antennas
        # in the LCS  and convert to GCS
        f_tx_pol1_prime = torch.stack(self._tx_array.ant_pol1.field(zod_prime,
                                                            aod_prime), dim=-1)
        f_rx_pol1_prime = torch.stack(self._rx_array.ant_pol1.field(zoa_prime,
                                                            aoa_prime), dim=-1)

        f_tx_pol1 = self._l2g_response(f_tx_pol1_prime, tx_orientations,
            zod, aod)

        f_rx_pol1 = self._l2g_response(f_rx_pol1_prime, rx_orientations,
            zoa, aoa)

        if self._tx_array.polarization == 'dual':
            f_tx_pol2_prime = torch.stack(self._tx_array.ant_pol2.field(
                zod_prime, aod_prime), dim=-1)
            f_tx_pol2 = self._l2g_response(f_tx_pol2_prime, tx_orientations,
                zod, aod)

        if self._rx_array.polarization == 'dual':
            f_rx_pol2_prime = torch.stack(self._rx_array.ant_pol2.field(
                zoa_prime, aoa_prime), dim=-1)
            f_rx_pol2 = self._l2g_response(f_rx_pol2_prime, rx_orientations,
                zoa, aoa)

        # Fill the full channel matrix with field responses
        pol1_tx = torch.matmul(h_phase, torch.complex(f_tx_pol1,
            torch.tensor(0., dtype=self._real_dtype)))
        if self._tx_array.polarization == 'dual':
            pol2_tx = torch.matmul(h_phase, torch.complex(f_tx_pol2, torch.tensor(0.,
                                             dtype=self._real_dtype)))

        num_ant_tx = self._tx_array.num_ant
        if self._tx_array.polarization == 'single':
            # Each BS antenna gets the polarization 1 response
            dim_array = torch.cat([torch.tensor([num_ant_tx]), torch.ones([pol1_tx.dim()], dtype=torch.int32)], dim=0).tolist()
            f_tx_array = torch.tile(torch.unsqueeze(pol1_tx, 0),dims=tuple(dim_array))
        else:
            # Assign polarization reponse according to polarization to each
            # antenna
            pol_tx = torch.stack([pol1_tx, pol2_tx], 0)
            ant_ind_pol2 = self._tx_array.ant_ind_pol2
            num_ant_pol2 = ant_ind_pol2.shape[0]
            # O = Pol 1, 1 = Pol 2, we only scatter the indices for Pol 1,
            # the other elements are already 0

            gather_ind = scatter_nd_torch(torch.reshape(ant_ind_pol2, [-1,1]),
                torch.ones([num_ant_pol2], torch.int32), [num_ant_tx])
            f_tx_array = gather_pytorch(pol_tx,indices=gather_ind,dim=0)

        num_ant_rx = self._rx_array.num_ant
        if self._rx_array.polarization == 'single':
            # Each UT antenna gets the polarization 1 response
            dim_arr = torch.cat([torch.tensor([num_ant_tx]), torch.ones([pol1_tx.dim()], dtype=torch.int32)], dim=0).tolist()
            f_rx_array = torch.tile(torch.unsqueeze(f_rx_pol1, 0),tuple(dim_arr))
            f_rx_array = torch.complex(f_rx_array,
                                    torch.tensor(0.,  dtype=self._real_dtype))
        else:
            # Assign polarization response according to polarization to each
            # antenna
            pol_rx = torch.stack([f_rx_pol1, f_rx_pol2], 0)
            ant_ind_pol2 = self._rx_array.ant_ind_pol2
            num_ant_pol2 = ant_ind_pol2.shape[0]
            # O = Pol 1, 1 = Pol 2, we only scatter the indices for Pol 1,
            # the other elements are already 0
            gather_ind = scatter_nd_torch(torch.reshape(ant_ind_pol2, [-1,1]),
                torch.ones(size=[num_ant_pol2], dtype=torch.int32), [num_ant_rx])
            f_rx_array = torch.complex(gather_pytorch(pol_rx, indices=gather_ind,dim=0),
                            torch.tensor(0.,  dtype=self._real_dtype))

        # Compute the scalar product between the field vectors through
        # sum and transpose to put antenna dimensions last
        h_field = torch.sum(torch.unsqueeze(f_rx_array, 1)*torch.unsqueeze(
            f_tx_array, 0), [-2,-1])
        rolled_dims = torch.roll(torch.arange(h_field.dim()),-2, 0)
        h_field = h_field.permute(rolled_dims.tolist())

        return h_field
    
    def _step_11_nlos(self, phi, topology, rays, t):
        """
        Compute the full NLOS channel matrix (7.5-28)

        Input
        -----
        phi: [batch size, num TXs, num RXs, num clusters, num rays, 4], torch.float
            Random initial phases [radian]

        topology : Topology
            Topology of the network

        rays : Rays
            Rays

        t : [num time samples], torch.float
            Time samples

        Output
        ------
        h_full : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas, num time steps], torch.complex
            NLoS channel matrix
        """
        h_phase = self._step_11_phase_matrix(phi, rays)
        h_field = self._step_11_field_matrix(topology, rays.aoa, rays.aod, 
                                             rays.zoa, rays.zod, h_phase)
        h_array = self._step_11_array_offsets(topology, rays.aoa, rays.aod, 
                                              rays.zoa, rays.zod)
        h_doppler = self._step_11_doppler_matrix(topology, rays.aoa, 
                                                 rays.zoa, t)
        
        h_full = (h_field * h_array).unsqueeze(-1) * h_doppler.unsqueeze(-2).unsqueeze(-2)

        real = torch.sqrt(rays.powers / h_full.size(4)).to(self._real_dtype)
        img = torch.tensor(0.,dtype=self._real_dtype)
        power_scaling = torch.complex(real,img)
        shape = torch.cat([torch.tensor(power_scaling.shape, dtype=torch.int32),
                           torch.ones(h_full.dim() - power_scaling.dim(), dtype=torch.int32)], 0).tolist()
        h_full *= torch.reshape(power_scaling, tuple(shape))      
        return h_full

    def _step_11_reduce_nlos(self, h_full, rays, c_ds):
        # pylint: disable=line-too-long
        r"""
        Compute the final NLOS matrix in (7.5-27)

        Input
        ------
        h_full : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas, num time steps], torch.complex
            NLoS channel matrix

        rays : Rays
            Rays

        c_ds : [batch size, num TX, num RX], torch.float
            Cluster delay spread

        Output
        -------
        h_nlos : [batch size, num_tx, num rx, num clusters, num rx antennas, num tx antennas, num time steps], torch.complex
            Paths NLoS coefficients

        delays_nlos : [batch size, num_tx, num rx, num clusters], torch.float
            Paths NLoS delays
        """

        if self._subclustering:

            powers = rays.powers
            delays = rays.delays

            # Sort all clusters along their power
            strongest_clusters = torch.argsort(powers, dim=-1,
                descending=True)

            # Sort delays according to the same ordering
            delays_sorted = gather_pytorch(delays, strongest_clusters,
                batch_dims=3, dim=3)

            # Split into delays for strong and weak clusters
            delays_strong = delays_sorted[...,:2]
            delays_weak = delays_sorted[...,2:]

            # Compute delays for sub-clusters
            offsets = torch.reshape(self._sub_cl_delay_offsets,
                (delays_strong.dim()-1)*[1]+[-1]+[1])
            delays_sub_cl = (torch.unsqueeze(delays_strong, -2) +
                offsets*torch.unsqueeze(torch.unsqueeze(c_ds, dim=-1), dim=-1))
            delays_sub_cl = torch.reshape(delays_sub_cl,
                list(delays_sub_cl.shape[:-2])+[-1])

            # Select the strongest two clusters for sub-cluster splitting
            h_strong = gather_pytorch(h_full, strongest_clusters[...,:2],
                batch_dims=3, dim=3)

            # The other clusters are the weak clusters
            h_weak = gather_pytorch(h_full, strongest_clusters[...,2:],
                batch_dims=3, dim=3)

            # Sum specific rays for each sub-cluster
            h_sub_cl_1 = torch.sum(gather_pytorch(h_strong,
                self._sub_cl_1_ind, dim=4), dim=4)
            h_sub_cl_2 = torch.sum(gather_pytorch(h_strong,
                self._sub_cl_2_ind, dim=4), dim=4)
            h_sub_cl_3 = torch.sum(gather_pytorch(h_strong,
                self._sub_cl_3_ind, dim=4), dim=4)

            # Sum all rays for the weak clusters
            h_weak = torch.sum(h_weak, dim=4)

            # catenate the channel and delay tensors
            h_nlos = torch.cat([h_sub_cl_1, h_sub_cl_2, h_sub_cl_3, h_weak],
                dim=3)
            delays_nlos = torch.cat([delays_sub_cl, delays_weak], dim=3)
        else:
            # Sum over rays
            h_nlos = torch.sum(h_full, dim=4)
            delays_nlos = rays.delays

        # Order the delays in ascending orders
        delays_ind = torch.argsort(delays_nlos, dim=-1,
            descending=False)
        delays_nlos = gather_pytorch(delays_nlos, delays_ind, batch_dims=3,
            dim=3)

        # # Order the channel clusters according to the delay, too
        h_nlos = gather_pytorch(h_nlos, delays_ind, batch_dims=3, dim=3)

        return h_nlos, delays_nlos

    def _step_11_los(self, topology, t):
        # pylint: disable=line-too-long
        r"""Compute the LOS channels from (7.5-29)

        Intput
        ------
        topology : Topology
            Network topology

        t : [num time samples], torch.float
            Number of time samples

        Output
        ------
        h_los : [batch size, num_tx, num rx, 1, num rx antennas, num tx antennas, num time steps], torch.complex
            Paths LoS coefficients
        """

        aoa = topology.los_aoa
        aod = topology.los_aod
        zoa = topology.los_zoa
        zod = topology.los_zod

         # LoS departure and arrival angles
        aoa = torch.unsqueeze(torch.unsqueeze(aoa, dim=3), dim=4)
        zoa = torch.unsqueeze(torch.unsqueeze(zoa, dim=3), dim=4)
        aod = torch.unsqueeze(torch.unsqueeze(aod, dim=3), dim=4)
        zod = torch.unsqueeze(torch.unsqueeze(zod, dim=3), dim=4)

        # Field matrix
        h_phase = torch.reshape(torch.tensor([[1.,0.],
                                         [0.,-1.]],
                                         dtype=self._dtype),
                                         [1,1,1,1,1,2,2])
        h_field = self._step_11_field_matrix(topology, aoa, aod, zoa, zod,
                                                                    h_phase)

        # Array offset matrix
        h_array = self._step_11_array_offsets(topology, aoa, aod, zoa, zod)

        # Doppler matrix
        h_doppler = self._step_11_doppler_matrix(topology, aoa, zoa, t)

        # Phase shift due to propagation delay
        d3d = topology.distance_3d
        lambda_0 = self._lambda_0
        h_delay = torch.exp(torch.complex(torch.tensor(0.,
                        dtype=self._real_dtype), 2*PI*d3d/lambda_0))

        # Combining all to compute channel coefficient
        h_field = torch.unsqueeze(torch.squeeze(h_field, dim=4), dim=-1)
        h_array = torch.unsqueeze(torch.squeeze(h_array, dim=4), dim=-1)
        h_doppler = torch.unsqueeze(h_doppler, dim=4)
        h_delay = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
            torch.unsqueeze(h_delay, dim=3), dim=4), dim=5), dim=6)

        h_los = h_field*h_array*h_doppler*h_delay
        return h_los

    def _step_11(self, phi, topology, k_factor, rays, t, c_ds):
        # pylint: disable=line-too-long
        r"""
        Combine LOS and LOS components to compute (7.5-30)

        Input
        -----
        phi: [batch size, num TXs, num RXs, num clusters, num rays, 4], torch.float
            Random initial phases

        topology : Topology
            Network topology

        k_factor : [batch size, num TX, num RX], torch.float
            Rician K-factor

        rays : Rays
            Rays

        t : [num time samples], torch.float
            Number of time samples

        c_ds : [batch size, num TX, num RX], torch.float
            Cluster delay spread
        """
        # ?rays
        h_full = self._step_11_nlos(phi, topology, rays, t)
        # print('h_full:---------',h_full)
        h_nlos, delays_nlos = self._step_11_reduce_nlos(h_full, rays, c_ds)
        # print('h_nlos:---------',h_nlos)
        ####  LoS scenario

        h_los_los_comp = self._step_11_los(topology, t)
        shape = torch.cat([torch.tensor(k_factor.shape), torch.ones([h_los_los_comp.dim() - k_factor.dim()],dtype = torch.int32)], 0).tolist()
        k_factor = torch.reshape(k_factor,tuple(shape) )

        k_factor = torch.complex(k_factor, torch.tensor(0.,
                                            dtype=self._real_dtype))

        # Scale NLOS and LOS components according to K-factor
        h_los_los_comp = h_los_los_comp*torch.sqrt(k_factor/(k_factor+1))
        h_los_nlos_comp = h_nlos*torch.sqrt(1/(k_factor+1))

        # Add the LOS component to the zero-delay NLOS cluster
        h_los_cl = h_los_los_comp + torch.unsqueeze(
            h_los_nlos_comp[:,:,:,0,...], 3)

        # Combine all clusters into a single tensor
        h_los = torch.cat([h_los_cl, h_los_nlos_comp[:,:,:,1:,...]], dim=3)

        #### LoS or NLoS CIR according to link configuration
        shape =torch.cat([torch.tensor(topology.los.shape), torch.tensor([1,1,1,1])], dim=0).tolist()
        los_indicator = torch.reshape(topology.los, tuple(shape)).bool()

        h = torch.where(los_indicator, h_los, h_nlos)

        return h, delays_nlos

