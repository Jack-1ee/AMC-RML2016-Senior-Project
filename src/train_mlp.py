import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

BASE_DIR = Path(__file__).resolve().parent.parent 
DATA_DIR = BASE_DIR / "outputs" / "test_data"
MODEL_DIR = BASE_DIR / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_train = np.load(DATA_DIR / 'X_train.npy')
y_train = np.load(DATA_DIR / 'y_train.npy')
X_val = np.load(DATA_DIR / 'X_val.npy')
y_val = np.load(DATA_DIR / 'y_val.npy')

#建立MLP模型
model = tf.keras.Sequential([
    #MLP最大致命傷 展平破壞了I/Q結構
    tf.keras.layers.Flatten(input_shape=(2, 128, 1)), 
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(11, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  #adam 自動調整學習率, 計算損失函數

model.summary()

#加入EarlyStopping避免過擬合
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

#加入ModelCheckpoint保留最優
checkpoint = ModelCheckpoint(
    filepath=MODEL_DIR / 'best_model_mlp.h5', 
    monitor='val_loss', 
    save_best_only=True
)

#開始訓練
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=1024,
    callbacks=[early_stopping, checkpoint],
    verbose=2
)

print("Training is complete; the model and full test data have been stored.")