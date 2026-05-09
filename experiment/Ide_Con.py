import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def generate_all_ideal_constellations():
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Theoretical Ideal Constellation Diagrams (11 Modulations)", fontsize=18, fontweight='bold')
    axes = axes.flatten()

    # ==========================================
    # 第一組：數位點陣 (離散座標)
    # ==========================================
    
    # 1. BPSK
    axes[0].scatter([-1, 1], [0, 0], color='blue', s=80, zorder=3)
    axes[0].set_title('BPSK')

    # 2. QPSK
    qpsk_pts = np.array([-1-1j, -1+1j, 1-1j, 1+1j]) / np.sqrt(2)
    axes[1].scatter(qpsk_pts.real, qpsk_pts.imag, color='blue', s=80, zorder=3)
    axes[1].set_title('QPSK')

    # 3. 8PSK (半徑r=1，相位theta=0, 45, 90...)
    theta_8psk = np.linspace(0, 2*np.pi, 8, endpoint=False)
    axes[2].scatter(np.cos(theta_8psk), np.sin(theta_8psk), color='blue', s=80, zorder=3)
    axes[2].set_title('8PSK')

    # 4. PAM4 (振幅r變化，相位0或180)
    pam4_pts = np.array([-3, -1, 1, 3]) / np.sqrt(5)
    axes[3].scatter(pam4_pts, np.zeros(4), color='blue', s=80, zorder=3)
    axes[3].set_title('PAM4')

    # 5. QAM16
    x16 = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    xv16, yv16 = np.meshgrid(x16, x16)
    axes[4].scatter(xv16.flatten(), yv16.flatten(), color='blue', s=60, zorder=3)
    axes[4].set_title('QAM16')

    # 6. QAM64
    x64 = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42)
    xv64, yv64 = np.meshgrid(x64, x64)
    axes[5].scatter(xv64.flatten(), yv64.flatten(), color='blue', s=30, zorder=3)
    axes[5].set_title('QAM64')

    # ==========================================
    # 第二組：連續軌跡 (頻率與類比調變)
    # ==========================================
    
    # 準備一段連續時間 t
    t = np.linspace(0, 1, 2000)

    # 7, 8, 9. 頻率調變家族 (CPFSK, GFSK, WBFM) -> 完美的圓
    # 物理意義：振幅 r 永遠等於 1，相位 theta 連續旋轉
    theta_continuous = 2 * np.pi * 5 * t
    for i, title in zip([6, 7, 8], ['CPFSK', 'GFSK', 'WBFM']):
        axes[i].plot(np.cos(theta_continuous), np.sin(theta_continuous), color='red', linewidth=2)
        axes[i].set_title(title)

    # 10. AM-DSB (雙邊帶振幅調變)
    # 物理意義：相位 theta 恆定為 0 (落在 I 軸上)，振幅 r 隨聲音信號變化
    message = np.sin(2 * np.pi * 3 * t) # 模擬單音聲音
    carrier_amplitude = 1.5
    am_dsb_I = carrier_amplitude + message
    am_dsb_Q = np.zeros_like(t)
    axes[9].plot(am_dsb_I, am_dsb_Q, color='red', linewidth=2)
    axes[9].set_title('AM-DSB')

    # 11. AM-SSB (單邊帶振幅調變)
    # 物理意義：使用希爾伯特轉換 (Hilbert Transform) 產生複雜的螺旋軌跡
    complex_message = message + 0.5 * np.sin(2 * np.pi * 7 * t) # 模擬稍微複雜的聲音
    analytic_signal = hilbert(complex_message)
    am_ssb_I = carrier_amplitude + np.real(analytic_signal)
    am_ssb_Q = np.imag(analytic_signal)
    axes[10].plot(am_ssb_I, am_ssb_Q, color='red', linewidth=1, alpha=0.8)
    axes[10].set_title('AM-SSB')

    # ==========================================
    # 畫布排版與座標軸設定
    # ==========================================
    axes[11].axis('off') # 隱藏第 12 個無用的格子

    for ax in axes[:11]:
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_aspect('equal', adjustable='box') # 強制 X 軸與 Y 軸比例為 1:1，圓形才不會變橢圓

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

generate_all_ideal_constellations()