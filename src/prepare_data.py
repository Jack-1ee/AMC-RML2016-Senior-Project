import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

BASE_DIR = Path(__file__).resolve().parent.parent #回到根目錄CODE
DATA_PATH = BASE_DIR / "data" / "RML2016.10a_dict.pkl"
OUTPUT_DIR = BASE_DIR / "outputs" / "test_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#訊號預處理 
#"RML2016.10a_dict.pkl"為 [1000, 2, 128]=(樣本數, IQ通道, 採樣點數)
with open(DATA_PATH, 'rb') as f: 
    raw_data = pickle.load(f, encoding='latin1') #py3讀py2的pickle
    
X, lbl, snr_list = [], [], []
for (mod, snr) in raw_data.keys():
    data = raw_data[(mod, snr)] 
    X.append(data)
    for _ in range(data.shape[0]): #shape[0]=樣本數=1000
        lbl.append(mod)
        snr_list.append(snr)
    
X = np.vstack(X)               #(樣本數, IQ通道, 128採樣點)
snr_list = np.array(snr_list)   

#F範數正規化：確保每個樣本能量一致
norms = np.linalg.norm(X, axis=(1, 2), keepdims=True) #利用 axis=(1,2)
X = X / (norms + 1e-8) #避免除以零

X = np.expand_dims(X, axis=-1) #擴充維度給CNN(樣本數, 2, 128, 1)

#標籤轉換 le保留inverse機會
le = LabelEncoder() 
y = to_categorical(le.fit_transform(lbl)) #標籤轉為one-hot
    
#資料分割 
X_temp, X_test, y_temp, y_test, snr_temp, snr_test = train_test_split(
X, y, snr_list, test_size=0.5, random_state=42 #固定隨機種子
)

X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
X_temp, y_temp, snr_temp, test_size=0.1, random_state=42
)
print(f"train sample: {X_train.shape}, validation sample: {X_val.shape}, test sample: {X_test.shape}")

#儲存預處理後的資料
np.save(OUTPUT_DIR / 'X_train.npy', X_train)
np.save(OUTPUT_DIR / 'y_train.npy', y_train)
np.save(OUTPUT_DIR / 'X_val.npy', X_val)
np.save(OUTPUT_DIR / 'y_val.npy', y_val)
np.save(OUTPUT_DIR / 'X_test.npy', X_test)
np.save(OUTPUT_DIR / 'y_test.npy', y_test)
np.save(OUTPUT_DIR / 'snr_test.npy', snr_test)
np.save(OUTPUT_DIR / 'classes.npy', le.classes_)

print("Data preprocessing completed and saved to disk.")
