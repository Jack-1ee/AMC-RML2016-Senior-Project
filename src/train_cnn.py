import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

BASE_DIR = Path(__file__).resolve().parent.parent #回到根目錄CODE
DATA_DIR = BASE_DIR / "outputs" / "test_data"
MODEL_DIR = BASE_DIR / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_train = np.load(DATA_DIR / 'X_train.npy')
y_train = np.load(DATA_DIR / 'y_train.npy')
X_val = np.load(DATA_DIR / 'X_val.npy')
y_val = np.load(DATA_DIR / 'y_val.npy')

#Keras Training與基準準確率
# 2D CNN 
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2, 128, 1)), 
    tf.keras.layers.Conv2D(256, (1, 3), activation='relu', padding='same'), #獨立看I Q
    tf.keras.layers.Dropout(0.5), #加入 Dropout 避免過擬合 強迫關掉0.5神經元
    tf.keras.layers.Conv2D(80, (2, 3), activation='relu', padding='valid'), #同時看I Q
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),                                        
    tf.keras.layers.Dense(256, activation='relu'),                     
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(11, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #adam 自動調整學習率, 計算損失函數

model.summary()

#加入EarlyStopping避免過擬合
early_stop = EarlyStopping(
    monitor='val_loss',         
    patience=10,                
    restore_best_weights=True   #停機後，倒回10圈內最好
)

#加入ModelCheckpoint保留最優
checkpoint = ModelCheckpoint(
    filepath=MODEL_DIR / 'best_model_cnn.h5', 
    monitor='val_loss',                    
    save_best_only=True #只存檔覆蓋最優
)

history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val),
    epochs=100, 
    batch_size=1024, 
    callbacks=[early_stop, checkpoint], 
    verbose=2
) #學習上限100遍 1024筆修正一次

print("Training is complete; the model and full test data have been stored.")