import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#1.I/Q 訊號預處理 (正規化) 

def load_and_preprocess(file_path, snr_threshold = 10): #SNR門檻10dB
    with open(file_path, 'rb') as f: 
        raw_data = pickle.load(f, encoding='latin1') #py3讀py2的pickle
    
    X, lbl = [], [] #x:訊號,lbl:標籤
    for (mod, snr) in raw_data.keys():
        if snr >= snr_threshold:  #僅選取SNR>=10dB
            X.append(raw_data[(mod, snr)])
            for _ in range(raw_data[(mod, snr)].shape[0]): #取得樣本總數
                lbl.append(mod)
    
    X = np.vstack(X)               #(樣本數, IQ通道, 128採樣點)
    X = np.transpose(X, (0, 2, 1)) #轉為 (樣本數, 128, 2)
    
    #L2 F範數正規化：確保每個樣本能量一致
    for i in range(X.shape[0]):
        norm = np.linalg.norm(X[i], 'fro') #Frobenius
        X[i] = X[i] / (norm + 1e-8) #避免除以零
    
    return X, lbl

#2. 資料分割 (80% Train / 20% Test)

print("正在讀取並處理資料...")
X, lbl = load_and_preprocess('RML2016.10a_dict.pkl')

#標籤轉換
le = LabelEncoder() #初始空間
y = tf.keras.utils.to_categorical(le.fit_transform(lbl)) #轉為one-hot

#執行分割 X=訓練數據, y=正確標籤
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 #固定隨機種子
)
print(f"訓練樣本: {X_train.shape}, 測試樣本: {X_test.shape}")

#3. Keras Training 與 基準準確率

# 1D CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 2)), #接收 長度128 IQ通道2
    tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same'), #特徵提取 64個濾波器, 大小3
    tf.keras.layers.MaxPooling1D(2), #降採樣 資料*1/2 增加穩定性
    tf.keras.layers.Flatten(), #轉成一維向量
    tf.keras.layers.Dense(32, activation='relu'), #全連接層 32神經元, ReLU激活
    tf.keras.layers.Dense(len(le.classes_), activation='softmax') #輸出 softmax(100%) 分類數量 = 標籤種類數量
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #adam 自動調整學習率, 計算損失函數

print("開始訓練基準模型...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1) #學習10遍 32筆修正一次 10%訓練資料作驗證

#獲取最終準確率
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n 基準準確率 (Benchmark Accuracy): {accuracy*100:.2f}%")

#python train_amc.py