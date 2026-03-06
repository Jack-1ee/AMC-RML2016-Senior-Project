import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_utils import load_and_preprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent #回到根目錄CODE

#資料分割 (80% Train / 20% Test)

X, lbl = load_and_preprocess('RML2016.10a_dict.pkl')

#標籤轉換
le = LabelEncoder() #初始空間
y = tf.keras.utils.to_categorical(le.fit_transform(lbl)) #轉為one-hot

#執行分割 X=訓練數據, y=正確標籤
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 #固定隨機種子
)
print(f"訓練樣本: {X_train.shape}, 測試樣本: {X_test.shape}")

#Keras Training與基準準確率
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
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1) #學習10遍 32筆修正一次 10%訓練資料作驗證

#儲存產出物
model.save(BASE_DIR / 'best_model.h5') #儲存模型
np.save(BASE_DIR / 'X_test_sample.npy', X_test[:10]) #儲存10筆測試樣本
np.save(BASE_DIR / 'y_test_sample.npy', y_test[:10]) #儲存10筆測試標籤 (one-hot格式)
print("訓練完成，已儲存模型與測試樣本。")

