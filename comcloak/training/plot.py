import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# log1 = pd.read_csv("/home/wzs/Project/sionna-main/comcloak/training/train_log/nrx_log/RX_log_v7_ResFiLM_correct.txt", header=None, names=["iter", "loss", "val_loss"])
# # log2 = pd.read_csv("/home/wzs/Project/sionna-main/comcloak/training/train_log/nrx_log/RX_log_v0_batch_var_snr_without_FiLM.txt", header=None, names=["iter", "loss"])
# plt.plot(log1["iter"], log1["loss"], label="loss")
# plt.plot(log1["iter"], log1["val_loss"], label="val_loss")
# # plt.plot(log2["iter"], log2["loss"], label="loss without FiLM  ")
# plt.title("Loss vs Iteration")
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid()
# plt.savefig("/home/wzs/Project/sionna-main/comcloak/training/train_log/nrx_log/RX_log_v7.png")
# plt.show()



# data1 = np.load("CDL-C_ber_without_film.npy", allow_pickle=True).item()
data2 = np.load("/home/wzs/Project/sionna-main/comcloak/training/results/data/CDL-B_ber_condconv_FiLM.npy", allow_pickle=True).item()
data3 = np.load("/home/wzs/Project/sionna-main/comcloak/training/results/data/TDL-C_ber_condconv_FiLM.npy", allow_pickle=True).item()
plt.figure(figsize=(7,5))
# plt.semilogy(data1["ebno_db"], data1["ber1"], "-o", label="4QAM")
# plt.semilogy(data1["ebno_db"], data1["ber2"], "-s", label="16QAM")
# plt.semilogy(data1["ebno_db"], data1["ber3"], "-^", label="64QAM")

plt.semilogy(data2["ebno_db"], data2["ber1"], "-o", label="4QAM_CDLB")
plt.semilogy(data2["ebno_db"], data2["ber2"], "-s", label="16QAM_CDLB")
plt.semilogy(data2["ebno_db"], data2["ber3"], "-^", label="64QAM_CDLB")

plt.semilogy(data3["ebno_db"], data3["ber1"], "-o", label="4QAM_TDL-C")
plt.semilogy(data3["ebno_db"], data3["ber2"], "-s", label="16QAM_TDL-C")
plt.semilogy(data3["ebno_db"], data3["ber3"], "-^", label="64QAM_TDL-C")
plt.title("BER vs Eb/N0 Compare")
plt.xlabel("Eb/N0 (dB)")
plt.ylabel("BER")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("/home/wzs/Project/sionna-main/comcloak/training/results/ber_compare_2_28.png")
plt.close()