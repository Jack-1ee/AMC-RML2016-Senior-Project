import tensorflow as tf
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent #回到根目錄CODE

#載入訓練好的模型與樣本
model = tf.keras.models.load_model(BASE_DIR / 'best_model.h5') #模型
X_test_sub = np.load(BASE_DIR / 'X_test_sample.npy') #10筆測試樣本 (128, 2)
y_test_sub = np.load(BASE_DIR / 'y_test_sample.npy') #10筆測試標籤 (one-hot格式)

def export_to_header(model, X_sample, y_sample, filename=BASE_DIR / 'model_params.h'):
    with open(filename, 'w') as f:
        #避免變數重複定義
        f.write("#ifndef MODEL_PARAMS_H\n#define MODEL_PARAMS_H\n\n")
        
        #匯出權重與偏差 (加上const以節省STM32 RAM)
        for layer in model.layers:
            weights = layer.get_weights() #抓取該層參數
            if not weights: #沒參數跳過
                continue
            
            w, b = weights[0], weights[1] #權重 偏差 分離
            layer_name = layer.name.replace("/", "_").replace(":", "_")
            
            f.write(f"// Layer: {layer_name}\n") #寫註解
            f.write(f"const float {layer_name}_w[] = {{ " + ", ".join([f"{v:.8f}f" for v in w.flatten()]) + " };\n")   #寫const float _w[]
            f.write(f"const float {layer_name}_b[] = {{ " + ", ".join([f"{v:.8f}f" for v in b.flatten()]) + " };\n\n") #寫const float _b[]
        
        #匯出第一筆測試資料
        #取出 X_test_sub 的第一筆資料 (128, 2)
        sample_0 = X_sample[0]
        f.write("// Test Input Sample (128x2)\n")
        f.write("const float test_input[128][2] = {\n")
        
        #python轉C陣列排版
        for i in range(128):
            f.write(f"    {{{sample_0[i, 0]:.8f}f, {sample_0[i, 1]:.8f}f}}") #{{}} = {} C陣列格式
            if i < 127: f.write(",")
            f.write("\n")
        f.write("};\n\n")

        # 匯出對應的正確標籤 (Index)
        true_label = np.argmax(y_sample[0]) #找one-hot最大 確定答案
        f.write(f"const int test_target_label = {true_label};\n\n") #寫入標準答案
        
        f.write("#endif\n")

# 執行
export_to_header(model, X_test_sub, y_test_sub)
print("model_params.h update completed.")