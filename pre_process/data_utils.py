import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent #parent.parent回到根目錄CODE
DATA_PATH = BASE_DIR / "RML2016.10a_dict.pkl" #在目錄找資料

#訊號預處理 
#"RML2016.10a_dict.pkl"為 [1000, 2, 128]=(樣本數, IQ通道, 採樣點數)
def load_and_preprocess(file_path, snr_threshold = 10): #SNR門檻10dB
    with open(file_path, 'rb') as f: 
        raw_data = pickle.load(f, encoding='latin1') #py3讀py2的pickle
    
    X, lbl = [], [] #x存訊號,lbl存標籤
    for (mod, snr) in raw_data.keys():
        if snr >= snr_threshold:
            X.append(raw_data[(mod, snr)])
            for _ in range(raw_data[(mod, snr)].shape[0]): #shape[0]=樣本數=1000
                lbl.append(mod)
    
    X = np.vstack(X)               #(樣本數, IQ通道, 128採樣點)
    X = np.transpose(X, (0, 2, 1)) #轉為 (樣本數, 128, 2)
    
    #Feature Scaling - F範數正規化：確保每個樣本能量一致
    for i in range(X.shape[0]):
        norm = np.linalg.norm(X[i], 'fro') #Frobenius Norm
        X[i] = X[i] / (norm + 1e-8) #避免除以零
    
    return X, lbl