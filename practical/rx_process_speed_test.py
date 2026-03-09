import time
import threading
import queue
import numpy as np
import torch


import os
gpu_num = 3 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import sionna
try:
    import sionna
except ImportError as e:
    # Install sionna if package is not already installed
    import os
    import sys
    print("Current directory:", os.getcwd())
    sys.path.append("/home/wzs/project/sionna-main/")
    # os.system("pip install sionna")
    import sionna
# Load the required sionna components
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from sionna.mimo import StreamManagement

from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers

from sionna.channel.tr38901 import AntennaArray, CDL, Antenna
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel

from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

from sionna.mapping import Mapper, Demapper

from sionna.utils import BinarySource, ebnodb2no, sim_ber
from sionna.utils.metrics import compute_ber
import tensorflow as tf
# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)
print(f"可用的GPU数量: {len(gpus)}")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
sionna.config.xla_compat=True

# if torch.cuda.is_available():
#     num_gpus = torch.cuda.device_count()
#     print(f"检测到 {num_gpus} 块 GPU:")
#     for i in range(num_gpus):
#         print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
#     device = torch.device("cuda:3")  
# else:
#     print("未检测到 GPU,使用 CPU")
#     device = torch.device("cpu")

# ========================
# 参数配置
# ========================
sample_rate = 600e6 / 16  # 假设网口总速率600Mbps，I/Q各16位=4字节/sample，约37.5MS/s
send_interval = 1e-3       # producer 每 1 ms 发送一次（可以改）
samples_per_chunk = int(sample_rate * send_interval)

# 形状配置
rx_ant = 1
symbols_per_slot = 1135    # 每帧 1135 个复样点
batch_size = 15000             # 堆叠成 [batch_size, 1, rx_ant, 1135]
frame_shape = (rx_ant, symbols_per_slot, 2)  # 最后一维 [I, Q]
dtype = np.int16
# 队列容量
max_queue = 5000
data_queue = queue.Queue(maxsize=max_queue)



# The number of transmitted streams is equal to the number of UT antennas
# in both uplink and downlink
num_streams_per_tx = 1

# Create an RX-TX association matrix
# rx_tx_association[i,j]=1 means that receiver i gets at least one stream
# from transmitter j. Depending on the transmission direction (uplink or downlink),
# the role of UT and BS can change. However, as we have only a single
# transmitter and receiver, this does not matter:
rx_tx_association = np.array([[1]])

# Instantiate a StreamManagement object
# This determines which data streams are determined for which receiver.
# In this simple setup, this is fairly easy. However, it can get more involved
# for simulations with many transmitters and receivers.
sm = StreamManagement(rx_tx_association, num_streams_per_tx)
rg = ResourceGrid(num_ofdm_symbols=14,
                fft_size=64,
                subcarrier_spacing=15e3,
                num_tx=1,
                num_streams_per_tx=num_streams_per_tx,
                cyclic_prefix_length=16,
                num_guard_carriers=[5,6],
                dc_null=True,
                pilot_pattern="kronecker",
                pilot_ofdm_symbol_indices=[2,11])

num_bits_per_symbol = 2 # QPSK modulation
coderate = 0.5 # Code rate
n = int(rg.num_data_symbols*num_bits_per_symbol) # Number of coded bits
k = int(n*coderate) # Number of information bits
# The encoder maps information bits to coded bits

ebno_db = 10
no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
encoder = LDPC5GEncoder(k, n)

l_min = torch.tensor(0, dtype=torch.int32)
demodulator = OFDMDemodulator(rg.fft_size, l_min, rg.cyclic_prefix_length)
# The LS channel estimator will provide channel estimates and error variances
ls_est = LSChannelEstimator(rg, interpolation_type="nn")
# The LMMSE equalizer will provide soft symbols together with noise variance estimates
lmmse_equ = LMMSEEqualizer(rg, sm)
# The demapper produces LLR for all coded bits
demapper = Demapper("app", "qam", num_bits_per_symbol)
# The decoder provides hard-decisions on the information bits
decoder = LDPC5GDecoder(encoder, hard_out=True)


stop_flag = threading.Event()
# ========================
# 模拟 Producer（发送端）
# ========================
def producer():
    frame_id = 0
    rng = np.random.default_rng()
    print("[Producer] start")
    while not stop_flag.is_set():
        # 模拟 16-bit I/Q 数据 [-32768, 32767]
        data = rng.integers(-32768, 32768, size=frame_shape, dtype=dtype)
        try:
            data_queue.put_nowait((frame_id, data))
        except queue.Full:
            print(f"[WARN] Queue full, drop frame {frame_id}")
        frame_id += 1
        time.sleep(send_interval)  # 模拟实时速率
    print("[Producer] exit")

# ========================
# 模拟 Consumer（接收机）
# ========================
def consumer():
    buffer = []
    processed = 0
    
    print("[Consumer] start")
    while not stop_flag.is_set():
        try:
            t_start = time.time()
            frame_id, data = data_queue.get(timeout=0.1)
            # -------------------------
            # 数据转换：int16 → float32 → complex
            # # -------------------------
            data_f32 = data.astype(np.float32) / 32768.0
            complex_tensor = data_f32[...,0] + 1j * data_f32[...,1]
            buffer.append(complex_tensor)

            # 当累积 batch_size 帧时堆叠处理
            if len(buffer) == batch_size:
                batch = np.stack(buffer, axis=0)   # [8,16,1120]
                buffer.clear()

                # 转成 torch 张量并添加 [8, 1, 16, 1120] 维度
                rx_batch = tf.expand_dims(tf.convert_to_tensor(batch, dtype=tf.complex64), 1) 
                # rx_batch = torch.from_numpy(batch).to(torch.complex64).unsqueeze(1).to(device)  # [batch_size,1,16,1120]
                # 模拟接收机处理（例如FFT或NN推理）
                # 这里简单地做一个FFT代替
                # rx_out = torch.fft.fft(rx_batch, dim=-1)
                y = demodulator(rx_batch)
                h_hat, err_var = ls_est ([y, no])
                x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])
                llr = demapper([x_hat, no_eff])
                b_hat = decoder(llr)
                # torch.cuda.synchronize() if device == 'cuda' else None

                processed += 1
                elapsed = time.time() - t_start
                throughput = batch_size * rx_ant * symbols_per_slot * 4 / (1024*1024*elapsed)
                print(f"[INFO] processed {processed:4d} batches | throughput={throughput:.2f} MB/s")
                if processed >= 1e5:
                    break
        except queue.Empty:
            continue
    print("[Consumer] exit")

# ========================
# 启动线程
# ========================

try:
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)
    producer_thread.start()
    consumer_thread.start()
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping threads...")
    stop_flag.set()
    producer_thread.join()
    consumer_thread.join()
    print("All threads stopped cleanly.")

# def process_data(rx_data):
    
#     # The number of transmitted streams is equal to the number of UT antennas
#     # in both uplink and downlink
#     num_streams_per_tx = 4
#     ebno_db = 30
#     no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate, rg)
#     # Create an RX-TX association matrix
#     # rx_tx_association[i,j]=1 means that receiver i gets at least one stream
#     # from transmitter j. Depending on the transmission direction (uplink or downlink),
#     # the role of UT and BS can change. However, as we have only a single
#     # transmitter and receiver, this does not matter:
#     rx_tx_association = np.array([[1]])

#     # Instantiate a StreamManagement object
#     # This determines which data streams are determined for which receiver.
#     # In this simple setup, this is fairly easy. However, it can get more involved
#     # for simulations with many transmitters and receivers.
#     sm = StreamManagement(rx_tx_association, num_streams_per_tx)
#     rg = ResourceGrid(num_ofdm_symbols=14,
#                     fft_size=64,
#                     subcarrier_spacing=15e3,
#                     num_tx=1,
#                     num_streams_per_tx=num_streams_per_tx,
#                   cyclic_prefix_length=16,
#                   num_guard_carriers=[5,6],
#                   dc_null=True,
#                   pilot_pattern="kronecker",
#                   pilot_ofdm_symbol_indices=[2,11])

#     num_bits_per_symbol = 2 # QPSK modulation
#     coderate = 0.5 # Code rate
#     n = int(rg.num_data_symbols*num_bits_per_symbol) # Number of coded bits
#     k = int(n*coderate) # Number of information bits
#     # The encoder maps information bits to coded bits
#     encoder = LDPC5GEncoder(k, n)

#     l_min = torch.tensor(-6, dtype=torch.int32)
#     demodulator = OFDMDemodulator(rg.fft_size, l_min, rg.cyclic_prefix_length)
#     # The LS channel estimator will provide channel estimates and error variances
#     ls_est = LSChannelEstimator(rg, interpolation_type="nn")
#     # The LMMSE equalizer will provide soft symbols together with noise variance estimates
#     lmmse_equ = LMMSEEqualizer(rg, sm)
#     # The demapper produces LLR for all coded bits
#     demapper = Demapper("app", "qam", num_bits_per_symbol)
#     # The decoder provides hard-decisions on the information bits
#     decoder = LDPC5GDecoder(encoder, hard_out=True)

#     y = demodulator(rx_data)

#     h_hat, err_var = ls_est ([y, no])
#     x_hat, no_eff = lmmse_equ([y, h_hat, err_var, no])
#     llr = demapper([x_hat, no_eff])
#     b_hat = decoder(llr)
#     return b_hat