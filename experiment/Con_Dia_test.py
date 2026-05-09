import pickle
import numpy as np
import matplotlib.pyplot as plt

# 載入資料集
file_path = r"C:\Users\李希均\Desktop\專題\code\RML2016.10a_dict.pkl"
print(f"正在載入資料庫，請稍候...")

with open(file_path, 'rb') as f:
    dataset = pickle.load(f, encoding='latin1')

# 定義 11 種調變格式
modulations = ['BPSK', 'QPSK', '8PSK', 'PAM4', 'QAM16', 'QAM64', 
               'CPFSK', 'GFSK', 'AM-DSB', 'AM-SSB', 'WBFM']

# 設定「理想狀況」SNR
target_snr = 18 

# 準備畫布 (設定 3 行 4 列的格子)
fig, axes = plt.subplots(3, 4, figsize=(15, 10))
fig.suptitle(f"RadioML 2016.10a Constellation Diagrams (SNR = {target_snr}dB)", fontsize=16)

# 將 2D 的 axes 矩陣壓平，方便我們用迴圈依序畫圖
axes = axes.flatten()

# 開始依序畫出 11 種調變
for i, mod in enumerate(modulations):
    # 抓取特定調變與 SNR 的資料
    key = (mod, target_snr)
    data = dataset[key]  # shape: (1000, 2, 128)
    
    # 為了讓星狀圖更明顯，我們取前 20 筆訊號片段，並把時間點全部攤平
    # 這樣就會有 20 * 128 = 2560 個點
    I_channel = data[:20, 0, :].flatten()
    Q_channel = data[:20, 1, :].flatten()
    
    # 在對應的格子裡畫散點圖
    ax = axes[i]
    ax.scatter(I_channel, Q_channel, s=1, color='b', alpha=0.5) # s=1 代表點的大小，alpha 是透明度
    
    # 設定圖表外觀
    ax.set_title(mod)
    ax.set_xlim([-0.02, 0.02]) # 根據 RadioML 的數值範圍稍微固定座標軸
    ax.set_ylim([-0.02, 0.02])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

# 第 12 個格子用不到，把它隱藏起來
axes[11].axis('off')

# 調整排版並顯示
plt.tight_layout()
plt.subplots_adjust(top=0.92) # 留一點空間給大標題
print("圖表繪製完成！請查看彈出的視窗。")
plt.show()