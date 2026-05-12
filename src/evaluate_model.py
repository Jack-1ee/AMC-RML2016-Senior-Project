import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "outputs" / "test_data"
MODEL_DIR = BASE_DIR / "outputs" / "models"
PLOT_DIR = BASE_DIR / "outputs" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)      #畫圖資料卡     

X_test = np.load(DATA_DIR / 'X_test.npy')
y_test = np.load(DATA_DIR / 'y_test.npy')
snr_test = np.load(DATA_DIR / 'snr_test.npy')
classes = np.load(DATA_DIR / 'classes.npy')

y_true = np.argmax(y_test, axis=1)  #把One-hot轉回普通數字標籤

model_cnn = tf.keras.models.load_model(MODEL_DIR / 'best_model_cnn.h5')
model_mlp = tf.keras.models.load_model(MODEL_DIR / 'best_model_mlp.h5')

#CNN與MLP預測
y_pred_prob_cnn = model_cnn.predict(X_test, batch_size=1024)    #predict列出各自機率
y_pred_cnn = np.argmax(y_pred_prob_cnn, axis=1)                 #argmax找出機率最大
y_pred_prob_mlp = model_mlp.predict(X_test, batch_size=1024)
y_pred_mlp = np.argmax(y_pred_prob_mlp, axis=1)

#依據難易度SNR計算準確率
snr_levels = np.unique(snr_test)    #挑選不重複數字
snr_levels = np.sort(snr_levels)    #從小到大排列
accuracies_cnn = []
accuracies_mlp = [] 

for snr in snr_levels:
    indices = np.where(snr_test == snr)[0]  #找出這個snr的考卷

    correct_cnn = np.sum(y_pred_cnn[indices] == y_true[indices])#正確題數
    acc_cnn = correct_cnn / len(indices)#準確率
    accuracies_cnn.append(acc_cnn)

    correct_mlp = np.sum(y_pred_mlp[indices] == y_true[indices])    #正確題數
    acc_mlp = correct_mlp / len(indices)    #準確率
    accuracies_mlp.append(acc_mlp)

    print(f"SNR {snr:>3}dB : accuracy {acc_cnn*100:>5.2f}% (CNN), {acc_mlp*100:>5.2f}% (MLP)")

#繪製折線圖
plt.figure(figsize=(10, 6))
# 畫上 CNN 的紅線
plt.plot(snr_levels, accuracies_cnn, marker='o', linestyle='-', color='#d62728', linewidth=2.5, label='CNN Model')
# 畫上 MLP 的藍線作對照
plt.plot(snr_levels, accuracies_mlp, marker='o', linestyle='-', color='#1f77b4', linewidth=2.5, label='MLP Model')

#圖表美化與標籤
plt.title('Modulation Classification Accuracy vs SNR (CNN vs MLP)', fontsize=16, fontweight='bold')
plt.xlabel('Signal-to-Noise Ratio (dB)', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(snr_levels) 
plt.yticks(np.arange(0, 1.1, 0.1)) 
plt.ylim([0, 1.05])
plt.legend(fontsize=12, loc='lower right')

#儲存圖片
save_path_curve = PLOT_DIR / 'snr_accuracy_comparison.png'
plt.savefig(save_path_curve, dpi=300, bbox_inches='tight')
print(f"The chart has been successfully saved：{save_path_curve}")

#繪製混淆矩陣
target_snrs = [18, 6, -6, -20]
#模型打包成字典 一次迴圈處裡
models_to_plot = {
    'CNN': y_pred_cnn,
    'MLP': y_pred_mlp
}

for model_name, y_pred_current in models_to_plot.items():
    for current_snr in target_snrs:
        #挑出特定SNR
        indices = np.where(snr_test == current_snr)[0] 
        #防呆
        if len(indices) == 0:
            print(f"Can't find data for SNR = {current_snr}dB, skipped.")
            continue

        y_pred_specific = y_pred_current[indices]
        y_true_specific = y_true[indices]

        #計算並正規劃混淆矩陣
        conf_mat = confusion_matrix(y_true_specific, y_pred_specific) #計算混淆矩陣
        conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

        #畫圖
        plt.figure(figsize=(10, 8)) 
        current_cmap = 'Reds' if model_name == 'CNN' else 'Blues'
        sns.heatmap(conf_mat_norm, annot=True, fmt=".2f", cmap=current_cmap, 
                    xticklabels=classes, yticklabels=classes)

        plt.title(f'{model_name} Confusion Matrix at SNR = {current_snr}dB', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)

        # 儲存第與顯示
        save_path_cm = PLOT_DIR / f'{model_name.lower()}_confusion_matrix_{current_snr}dB.png'
        plt.savefig(save_path_cm, dpi=300, bbox_inches='tight')
    
        print(f"The chart has been successfully saved：{save_path_cm.name}")
        plt.close()

print("CNN ", model_cnn.count_params())
print("MLP ", model_mlp.count_params())