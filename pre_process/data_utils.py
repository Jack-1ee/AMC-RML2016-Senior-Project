import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent #parent.parent回到根目錄CODE
DATA_PATH = BASE_DIR / "RML2016.10a_dict.pkl" #在目錄找資料

#I/Q 訊號預處理 (正規化) 
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
    
    #F範數正規化：確保每個樣本能量一致
    for i in range(X.shape[0]):
        norm = np.linalg.norm(X[i], 'fro') #Frobenius
        X[i] = X[i] / (norm + 1e-8) #避免除以零
    
    return X, lbl