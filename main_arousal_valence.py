# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # ERC using text, audio and physical signal

# ## 0) Read all anotation file

import os
from collections import Counter
os.chdir("G:/내 드라이브/2023aifactory")
os.getcwd()

# ### 0-1) Declare Global variable

epochs = 21
max_len = 128 # maximum number of conversation in each script
hidden_dim = 64 # number of hidden state
device = 'cpu'
batch_size = 6*2

# ### 0-2) read annotation file

# +
import numpy as np
import pandas as pd
import glob, copy, gc
from copy import deepcopy
import torch
from torch import nn
import math

def read_eda(num,j,base_="EDA"):
    temp_dir = os.getcwd() + "\\"+base_+"\\Session"+num
    try:    temp_dat = pd.read_csv(temp_dir + "\\"+ os.listdir(temp_dir)[j], sep="\t")
    except: return None

    eda = pd.DataFrame(columns=['val', 'id'])

    all_val = []
    for i in range(1,len(temp_dat)):
        temp_str_ = str.split(''.join(temp_dat.iloc[i].tolist()),",")
        if len(temp_str_)>2: 
            temp_df = pd.DataFrame({'val':[float(temp_str_[0])],'id':temp_str_[2]})
        else:
            temp_df = pd.DataFrame({'val':[float(temp_str_[0])],'id':"n"})
        eda = pd.concat([eda,temp_df],axis=0)

    eda = eda.reset_index(drop=True)
    # 변화값
    eda['val'] = eda.val - pd.concat([pd.Series(0),eda.val.iloc[:-1]]).reset_index(drop=True)
    ## normalize
    eda['val'] = (eda.val - np.mean(eda.val))/np.std(eda.val.tolist())
    ## filtering
    eda = eda.loc[eda.id!="n"]
    ## id 마다 mean eda value 지정
    idx = eda.id.unique().tolist()
    val_ = []
    for i in range(len(idx)):
        sum_ = np.mean(eda.loc[eda.id==idx[i]].val)
        val_.append(sum_)
    return pd.DataFrame({'id':idx,'val':val_})

def read_ibi(num,j):
    temp_dir = os.getcwd() + "\\IBI\\Session"+num
    try: temp_dat = pd.read_csv(temp_dir + "\\"+ os.listdir(temp_dir)[j], sep="\t")
    except: return None
    ibi = pd.DataFrame(columns=['val', 'id'])

    all_val = []
    for i in range(len(temp_dat)):
        temp_str_ = str.split(''.join(temp_dat.iloc[i].tolist()),",")
        if len(temp_str_)>3: 
            temp_df = pd.DataFrame({'val':[float(temp_str_[1])],'id':temp_str_[3]})
        else:
            try: temp_df = pd.DataFrame({'val':[float(temp_str_[1])],'id':"n"})
            except: continue
        ibi = pd.concat([ibi,temp_df],axis=0)

    ibi = ibi.reset_index(drop=True)

    ## normalize
    ibi['val'] = (ibi.val - np.mean(ibi.val))/np.std(ibi.val.tolist())
    ## filtering
    ibi = ibi.loc[ibi.id!="n"]
    ## id 마다 mean eda value 지정
    idx = ibi.id.unique().tolist()
    val_ = []
    for i in range(len(idx)):
        sum_ = np.mean(ibi.loc[ibi.id==idx[i]].val)
        val_.append(sum_)
    return pd.DataFrame({'id':idx,'val':val_})

### Session01 ~ Session32 : Train set
### Session33 ~ Session40 : Test set

dic_ = {'happy':0,'surprise':1,'angry':2,'neutral':3,'disqust':4,'fear':5,'sad':6} # for emotion labeling

dir_list = glob.glob("wav/Session*") # length == 40

# Session
#    6 scripts
#       conversation from male & female

script_ = [0 for i in range(6)]
id_list = [deepcopy(script_) for i in range(len(dir_list))]

## all annotation data: list -> torch.tensor
frame = torch.zeros(40,6,max_len) # session / script / conversation
gender_tensor = deepcopy(frame).to(device)
emotion_tensor = deepcopy(frame-1).to(device)
valence_tensor = deepcopy(frame).to(device)
arousal_tensor = deepcopy(frame).to(device)

eda_tensor = deepcopy(frame).to(device)
ibi_tensor = deepcopy(frame).to(device)
temp_tensor = deepcopy(frame).to(device)

for i in range(40):
    print(f"{i+1} processing...")
#     dirpath = os.path.join(os.getcwd(),dir_list[i])
    annot = pd.read_csv(os.getcwd() + "/annotation/Sess"+str(i+1).zfill(2)+"_eval.csv")
    id_ = annot.iloc[1:,3].tolist()
    gender = [1 for i in range(len(id_))]
    for j in range(len(id_)):
        if str.find(id_[j],"User"+str((i+1)*2).zfill(3)) > 0:
            gender[j] = 0 # User 구분

    emotion = annot.iloc[1:,4]
    emotion[emotion.str.contains(";")] = emotion[emotion.str.contains(";")].str.split(";").str[0] # Choose first emotion when there are two emotions.
    emotion = emotion.tolist()
    valence = [float(i) for i in annot.iloc[1:,5].tolist()]
    arousal = [float(i) for i in annot.iloc[1:,6].tolist()]
    

    for j in range(1,7):
        print(f"{j}..." ,end=" ")
        sess_num = str(i+1).zfill(2);
        eda = pd.concat([read_eda(sess_num,2*(j-1)), read_eda(sess_num,2*(j-1)+1)], axis=0)
        ibi = pd.concat([read_ibi(sess_num,2*(j-1)), read_ibi(sess_num,2*(j-1)+1)], axis=0)
        skin_temp = pd.concat([read_eda(sess_num,2*(j-1),'TEMP'), read_eda(sess_num,2*(j-1)+1,"TEMP")], axis=0)
        # EDA
        
        j_idx = [id_.index(s) for s in id_ if "script"+str(j).zfill(2) in s]
        id_list[i][j-1] = [id_[s] for s in j_idx]
        
        for k in range(len(id_list[i][j-1])):
            if eda.id.isin([id_list[i][j-1][k]]).sum() > 0:
                eda_tensor[i,j-1,k] = float(eda.val.loc[eda.id.isin([id_list[i][j-1][k]])])
            if ibi.id.isin([id_list[i][j-1][k]]).sum() > 0:
                ibi_tensor[i,j-1,k] = float(ibi.val.loc[ibi.id.isin([id_list[i][j-1][k]])])
            if skin_temp.id.isin([id_list[i][j-1][k]]).sum() > 0:
                temp_tensor[i,j-1,k] = float(skin_temp.val.loc[skin_temp.id.isin([id_list[i][j-1][k]])])
        # list -> tensor
        n_ = len([gender[s] for s in j_idx])
        gender_tensor[i,j-1,:n_] = torch.tensor([gender[s] for s in j_idx], dtype=torch.int32)
        emotion_tensor[i,j-1,:n_] = torch.tensor([dic_[emotion[s]] for s in j_idx], dtype=torch.int32)
        valence_tensor[i,j-1,:n_] = torch.tensor([valence[s] for s in j_idx], dtype=torch.float32)
        arousal_tensor[i,j-1,:n_] = torch.tensor([arousal[s] for s in j_idx], dtype=torch.float32)

    del id_, gender, emotion, valence, arousal
    print()
gc.collect()

# +
# np.save(os.getcwd()+"\\"+"eda_tensor",eda_tensor.detach().numpy())
# np.save(os.getcwd()+"\\"+"ibi_tensor",ibi_tensor.detach().numpy())
# np.save(os.getcwd()+"\\"+"temp_tensor",temp_tensor.detach().numpy())

eda_tensor = torch.tensor(np.load(os.getcwd()+"\\"+"eda_tensor.npy"))
ibi_tensor = torch.tensor(np.load(os.getcwd()+"\\"+"ibi_tensor.npy"))
temp_tensor = torch.tensor(np.load(os.getcwd()+"\\"+"temp_tensor.npy"))

# -

# ## 1) Input preprocessing

# ### 1-1) Text input by KoBERT (worked in colab should be loaded.)

# +
import torch
from torch import nn
# device = torch.device("cuda")
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    device = torch.device('cuda')

print(device)
device="cpu"

# +
# kobert_output = torch.zeros([32,6,max_len,768]).to(device) # (train session, scripts, max_conversation_length, output_dim_of_KoBERT)
# kobert_output = torch.zeros([32,6,max_len,768]).to(device) # (train session, scripts, max_conversation_length, output_dim_of_KoBERT)
kobert_output = torch.tensor(np.load("G:/내 드라이브/2023aifactory/wav_text/temp_session01.npy")) # read only train text
print(kobert_output.shape)

for i in range(1,32):
    kobert_output[i,:,:,:] = torch.tensor(np.load("G:/내 드라이브/2023aifactory/wav_text/temp_session"+str(i+1).zfill(2)+".npy"),
                                         dtype=torch.float32)[0,:,:,:]

test_kobert_output = torch.zeros(8,kobert_output.shape[1],kobert_output.shape[2],kobert_output.shape[3])
for i in range(32,40):
    test_kobert_output[i-32,:,:,:] = torch.tensor(np.load("G:/내 드라이브/2023aifactory/wav_text/temp_session"+str(i+1).zfill(2)+".npy"),
                                         dtype=torch.float32)[0,:,:,:]
gc.collect()

# for i in range(1,2): # call only 1 session temporalry
#     print(f"session {i+1}")
#     temp_export_dir = os.getcwd()+"/wav_text/Session"+str(i+1).zfill(2)
    
#     for j in range(6): # six scripts
#         print(f"script {j+1}", end=" ")
        
#         for k in range(len(id_list[i][j])):
#             temp = torch.tensor(np.load(temp_export_dir+"/"+id_list[i][j][k]+".npy"), dtype=torch.float32)
#             kobert_output[i][j][k] = temp.clone().detach()
#             del temp
#             gc.collect()
#     print()
# -

gc.collect()

# +
# np.save("G:/내 드라이브/2023aifactory/wav_text/temp_session01", kobert_output.cpu().detach().numpy() )
# kobert_output = torch.tensor(np.load("G:/내 드라이브/2023aifactory/wav_text/temp_session01.npy"))
# -


# ### 1-2) Audio by MFCC

# +
# # !pip install librosa
import librosa
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools

def get_librosa_mfcc(filepath, n_mfcc=hidden_dim):
    sample_rate = 16000
    hop_length = 160
    n_fft = 400
    
    sig, sr = librosa.core.load(filepath, sr=sample_rate)
    mfccs = librosa.feature.mfcc(y=sig, sr=sr, hop_length=hop_length,
                                 n_mfcc=n_mfcc, n_fft=n_fft)
    mfccs = torch.FloatTensor(mfccs).contiguous().transpose(0,1)
    return mfccs

# from sklearn.decomposition import PCA
# pca = PCA(n_components=1)


# -

scripts_ = [[] for i in range(6)]
mfcc_output = [deepcopy(scripts_) for i in range(32)]
test_mfcc_output = [deepcopy(scripts_) for i in range(8)]

# +
import _pickle as cPickle

lst_ = ["temp_session_1_17_list","temp_session_18_list","temp_session_19_26_list","temp_session_27_32_list","temp_session_33_40_list"]
# range_ = [[0,17],[18,26],[27,32]]
# read only train set
for i in range(5):
    print(lst_[i])
    with open(os.getcwd()+"/wav_audio/"+lst_[i], "rb") as fp:   # Unpickling
        temp = cPickle.load(fp)
        print(len(temp))
        
    if i==0: mfcc_output = deepcopy(temp)
    elif i < 4: 
        for j in range(len(temp)):
            mfcc_output.append(deepcopy(temp[j]))
    else:
        test_mfcc_output = deepcopy(temp)
    del temp; gc.collect()


# +
# import glob, copy
# from copy import deepcopy
# import os
# import glob
# import gc
# import pandas as pd
# import re

# np.load("wav_audio/temp_session")
# # for i in range(40): # call all session with train(32) and test(8)
# # for i in range(1): # call only 1 session temporalry
# for i in range(17,32): # call only 32 session for training
#     print(f"session {i+1}")
    
#     for j in range(6):
#         print(f"script {j+1}", end=" ")
#         for k in range(len(id_list[i][j])): 
#             filepath =  os.getcwd() + "/wav/Session"+str(i+1).zfill(2)+"/"+id_list[i][j][k]+".wav"
#             temp = get_librosa_mfcc(filepath)
#             temp = preprocessing.scale(temp, axis=0).T
#             mfcc_output[i][j].append(temp)
#             del temp
    
#     gc.collect()
#     print()

# +
# import pickle

# with open("wav_audio/temp_session_18_26_list", "wb") as fp:
#     pickle.dump(mfcc_output[18:26], fp)
# -

# ### 1-3) set same output shape

class inputProcessing(nn.Module):
    def __init__(self,output_dim=hidden_dim):
        super(inputProcessing,self).__init__()
        self.output_dim = output_dim

    def reduceDim(self,input):
        input_dim = input.shape[len(input.shape)-1]
        if input.count_nonzero() > 0:
            func = nn.Linear(input_dim,self.output_dim).to(device)
        else: ## audio vector with no sound ## fix all value to zero
            func = nn.Linear(input_dim,self.output_dim, bias=False).to(device)

        # func.weight.data.to(device)
        # func.bias.data.to(device)
    
        res = func(input)
        
        if len(input.shape) > 2: # for text
            res[input[:,:,:,:self.output_dim]==0] = 0 # set zero where all input row was zero
        else: # for audio
            res[input[:,:self.output_dim]==0] = 0

        return res

    def forward(self,x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x,dtype=torch.float32)
        return self.reduceDim(x.to(device))


# ## 2) GRU (time-series)

# ### 2-1) setting GRU

from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import math
from torch.nn import GRUCell
import copy
from copy import deepcopy


class GRUModel(nn.Module):
    def __init__(self, batch_dim, input_dim, hidden_dim, layer_dim, output_dim, dropout_p=0.2, bContext=False):
        super(GRUModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p) 
        self.layer_dim = layer_dim
        self.batch_dim = batch_dim
        self.bContext = bContext # bContext: Context level or not

        # 3 layers of gru
        self.gru_0 = nn.GRUCell(self.input_dim, self.hidden_dim)
        self.gru_1 = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.gru_2 = nn.GRUCell(self.hidden_dim, self.hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        if torch.cuda.is_available():
          # h0 = torch.randn(self.layer_dim, self.batch_dim, self.hidden_dim).cuda() # layer, batch, hid # for GRU
            self.h0 = torch.zeros(self.batch_dim, self.hidden_dim) # layer, hid
            self.h1 = torch.zeros(self.batch_dim, self.hidden_dim) # layer, hid
            self.h2 = torch.zeros(self.batch_dim, self.hidden_dim) # layer, hid
            self.h0_F = torch.zeros(self.batch_dim, self.hidden_dim) # for self-context level # Female
            self.h1_F = torch.zeros(self.batch_dim, self.hidden_dim) # for self-context level # Female
            self.h2_F = torch.zeros(self.batch_dim, self.hidden_dim) # for self-context level # Female
        else:
          # h0 = torch.randn(self.layer_dim, 1, self.hidden_dim)
            self.h0 = torch.zeros(self.batch_dim, self.hidden_dim)
            self.h1 = torch.zeros(self.batch_dim, self.hidden_dim)
            self.h2 = torch.zeros(self.batch_dim, self.hidden_dim)
            self.h0_F = torch.zeros(self.batch_dim, self.hidden_dim) # for self-context level # Female
            self.h1_F = torch.zeros(self.batch_dim, self.hidden_dim) # for self-context level # Female
            self.h2_F = torch.zeros(self.batch_dim, self.hidden_dim) # for self-context level # Female
        
    def forward(self, x, gender=None):
        
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
       
        outs = []

        for i in range(x.shape[1]): # 1~seq_len
            input_ = x[:,i,:]
            mask_ = torch.zeros_like(input_)
            mask_[~torch.logical_not(input_)] = 1
            
            if self.bContext==True: # context-level
                self.h0 = self.gru_0(input_,self.h0)
                self.h1 = self.gru_1(self.h0,self.h1)
                self.h2 = self.gru_2(self.h1,self.h2)
                self.h2 = self.h2 * mask_ ## skip zero vector
                outs.append(self.h2.unsqueeze(1))
            else: # self-context level
                if gender[i]==0: # speaker gender
                    self.h0 = self.gru_0(input_,self.h0)
                    self.h1 = self.gru_1(self.h0,self.h1)
                    self.h2 = self.gru_2(self.h1,self.h2)
                    self.h2 = self.h2 * mask_ ## skip zero vector
                    outs.append(self.h2.unsqueeze(1))
                else: 
                    self.h0_F = self.gru_0(input_,self.h0_F)
                    self.h1_F = self.gru_1(self.h0_F,self.h1_F)
                    self.h2_F = self.gru_2(self.h1_F,self.h2_F)
                    self.h2_F = self.h2_F * mask_ ## skip zero vector
                    outs.append(self.h2_F.unsqueeze(1))
            
        outs = torch.cat(outs, dim = -2).squeeze()
        
        return outs


# ### 2-2) context GRU & self-context GRU

# +
# text model

class TimeSeriesModel(nn.Module):
    def __init__(self):
        super(TimeSeriesModel, self).__init__()

    def forward(self, text_input, audio_input, gender):
    
        model = GRUModel(batch_dim = text_input.shape[0], input_dim = hidden_dim, hidden_dim=hidden_dim, layer_dim=3, output_dim=hidden_dim, dropout_p=0.2, bContext=True)
        model_s = GRUModel(batch_dim = 1, input_dim = hidden_dim, hidden_dim=hidden_dim, layer_dim=3, output_dim=hidden_dim, dropout_p=0.2, bContext=False) 
#         if torch.cuda.is_available():
#             model = model.cuda() # contextual context-level
#             model_s = model_s.cuda() # speaker's self-context-level

        text_input_c = model(text_input.to(device)) # (198,128,64) => (session * script, seq_len, hidden_dim)
        audio_input_c = model(audio_input.to(device)) # (198,128,64) => (session * script, seq_len, hidden_dim)
        
        text_input_s = torch.zeros_like(text_input_c)
        audio_input_s = torch.zeros_like(audio_input_c)
        
        for i in range(text_input.shape[0]):
            text_input_s[i,:,:] = model_s(text_input[i,:,:].unsqueeze(dim=0).to(device), gender=gender[i,:].to(device))
            audio_input_s[i,:,:] = model_s(audio_input[i,:,:].unsqueeze(dim=0).to(device), gender=gender[i,:].to(device))
            
        gc.collect()

        return (text_input_c+text_input_s)/2, (audio_input_c+audio_input_s)/2


# -

# ## 3) Cross-Attention

# ### 3-1) Multi-head Attention

device='cpu'


# +
class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_embed, h):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_embed = d_embed
        self.h = h
        self.d_model = d_embed*h
        self.q_fc = nn.Linear(d_embed,self.d_model).to(device) # (d_embed, d_model)
        self.k_fc = nn.Linear(self.d_embed,self.d_model).to(device) # (d_embed, d_model)
        self.v_fc = nn.Linear(self.d_embed,self.d_model).to(device) # (d_embed, d_model)
        self.out_fc = nn.Linear(self.d_model,d_embed).to(device)          # (d_model, d_embed)

    def forward(self, *args, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)
        softmax = nn.Softmax(dim=-1)

        def transform(x, fc):  # (n_batch, seq_len, d_embed)
            out = fc(x)        # (n_batch, seq_len, d_model)
            out = out.contiguous().view(n_batch, -1, self.h, self.d_embed) # (n_batch, seq_len, h, d_k)
            out = out.contiguous().transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out

        def calculate_attention(query, key, value, mask, n_xor=0, mask_xor=None):
          # query, key, value: (n_batch, h, seq_len, d_k)
          # mask: (n_batch, 1, seq_len, seq_len)
            
            # n_xor==0: query and key have same number of zero
            # n_xor==1: key have more zero than query -> query win
            # n_xor==2: query have more zero than key -> key win
            
            ### mask_ 추가하기
            
            mask = mask.unsqueeze(dim=1).repeat(1,self.h,1,1)
            d_k = key.shape[-1]
            attention_score = torch.matmul(query, key.contiguous().transpose(-2, -1)) # Q x K^T, (n_batch, h, seq_len, seq_len)
            attention_score = attention_score / math.sqrt(d_k)
            
            if mask is not None:
                attention_score = attention_score.masked_fill(mask==0, -1e9)
            attention_prob = softmax(attention_score) # (n_batch, h, seq_len, seq_len)
            
            if mask is not None:
                attention_prob = attention_prob.masked_fill(mask==0, 0) # set zero to original zero index
            
            out = torch.matmul(attention_prob, value) # (n_batch, h, seq_len, d_k)
            
#             if n_xor==1:
#                 print('n_xor=1')
#                 for i in range(self.h):
#                     out[:,j,:,:][mask_xor] = query[:,j,:,:][mask_xor].contiguous()
#             elif n_xor==2:
#                 print('n_xor=2')
#                 for i in range(self.h):
#                     out[:,j,:,:][mask_xor] = key[:,j,:,:][mask_xor].contiguous()
            
            return out

        temp = torch.ne(torch.sum(torch.ne(query,0),dim=-1),0)
        mask_q = torch.stack([deepcopy(temp) for i in range(query.shape[-1])],axis=-1)
        temp = torch.ne(torch.sum(torch.ne(key,0),dim=-1),0)
        mask_k = torch.stack([deepcopy(temp) for i in range(key.shape[-1])],axis=-1)
        
#         mask_k = torch.ne(torch.sum(torch.ne(key,0),dim=-1))
#         mask_q = torch.zeros_like(query) # recognize zero vector
#         mask_q[~torch.logical_not(query)] = 1
#         mask_k = torch.zeros_like(key) # recognize zero vector
#         mask_k[~torch.logical_not(key)] = 1

        mask_xor = torch.logical_xor(mask_q,mask_k)
        if(mask_xor.sum()>0):
            n_q = mask_q.sum()
            n_k = mask_k.sum()
            if n_q > n_k: n_xor=1
            elif n_q < n_k: n_xor=2
            else: n_xor=0
        
        n_xor=0
        query = transform(query.to(device), self.q_fc) # (n_batch, h, seq_len, d_k)
        key = transform(key.to(device), self.k_fc)     # (n_batch, h, seq_len, d_k)
        value = transform(value.to(device), self.v_fc) # (n_batch, h, seq_len, d_k)

        out = calculate_attention(query, key, value, mask, n_xor, mask_xor) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed)
        
        return out


# -

class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, d_k, d_ff=None):
        super(PositionWiseFeedForwardLayer, self).__init__()
        if d_ff is None: d_ff = d_k
        self.fc1 = nn.Linear(d_k,d_ff).to(device)   # (d_embed, d_ff)
        self.relu = nn.ReLU().to(device)
        self.fc2 = nn.Linear(d_ff,d_k).to(device) # (d_ff, d_embed)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ResidualConnectionLayer(nn.Module):

    def __init__(self):
        super(ResidualConnectionLayer, self).__init__()

    def forward(self, x):
        lin = nn.Linear(x.shape[-1],x.shape[-1]).to(device)
        return lin(x) + x


# ### 3-2) Encoder

class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(2)]
        self.do = nn.Dropout(0.2).to(device)
        self.layer_norm = nn.LayerNorm(hidden_dim).to(device)
        
    def forward(self, trg, src, src_mask=None):
        
        out = self.layer_norm(trg)
        src = self.layer_norm(src)
        out = self.self_attention(query=out, key=src, value=src, mask=src_mask)
        out = self.residuals[0](self.layer_norm(self.do(out)))
        out = self.position_ff(self.layer_norm(self.do(out)))
        out = self.residuals[1](self.layer_norm(self.do(out)))
        out = self.do(out)

        return out


class Encoder(nn.Module):

    def __init__(self, encoder_layer, n_layer):  # n_layer: Encoder Layer의 개수
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_layer))

             
    def forward(self, trg,src,src_mask):
        out = trg
        for layer in self.layers:
            out = layer(out, src, src_mask)
        return out


class Transformer(nn.Module):
    def __init__(self, encoder):
        super(Transformer,self).__init__()
        self.encoder = encoder
  
    def forward(self, query, key, bMask=False):
        def make_pad_mask(query, key, pad_idx=0):
            query = query[:,:,0]
            key = key[:,:,0]
            # query: (n_batch, query_seq_len)
            # key: (n_batch, key_seq_len)

            mask_q = torch.ne(query,0).unsqueeze(dim=1).repeat(1,query.shape[1],1)
            mask_k = torch.ne(key,0).unsqueeze(dim=1).repeat(1,query.shape[1],1)
            mask = (mask_q * mask_k).transpose(1,2).contiguous()
            
            return mask

        def make_src_mask(trg, src):
            pad_mask = make_pad_mask(trg, src)
            return pad_mask

        src_mask = make_src_mask(query, key)

        if bMask:
            out = self.encoder(query, key, src_mask)
        else:
            out = self.encoder(query, key, None) 
    
        return out


class CrossModalTransformer(nn.Module):

    def __init__(self, d_embed=64, h=8, n_layer=5):
        super(CrossModalTransformer, self).__init__()
    # declare
        self.attention = MultiHeadAttentionLayer(d_embed = 64, h = 8)
        self.positionff = PositionWiseFeedForwardLayer(d_k=64, d_ff=64)
        self.residual = ResidualConnectionLayer()
        self.encoder_layer = EncoderBlock(self.attention, self.positionff)
        self.encoder = Encoder(self.encoder_layer,n_layer)
        self.model = Transformer(self.encoder)

    def forward(self, query, key, bMask):
        out = self.model(query, key, bMask)
        return out


class Attention(nn.Module):
    def __init__(self, d_embed):
        super(Attention, self).__init__()
        self.d_embed = d_embed
        self.q_fc = nn.Linear(self.d_embed,self.d_embed).to(device)
        self.k_fc = nn.Linear(self.d_embed,self.d_embed).to(device)
        self.v_fc = nn.Linear(self.d_embed,self.d_embed).to(device)
        self.out_fc = nn.Linear(self.d_embed,self.d_embed).to(device)

    def forward(self, src, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        softmax = nn.Softmax(dim=-1).to(device)
        query = src.clone().to(device)
        key = src.clone().to(device)
        value = src.clone().to(device)
        n_batch = query.size(0)
        
        def make_pad_mask(query, key, pad_idx=0):
            query = query[:,:,0]
            key = key[:,:,0]
            # query: (n_batch, query_seq_len)
            # key: (n_batch, key_seq_len)

            mask_q = torch.ne(query,0).unsqueeze(dim=1).repeat(1,query.shape[1],1)
            mask_k = torch.ne(key,0).unsqueeze(dim=1).repeat(1,query.shape[1],1)
            mask = (mask_q * mask_k).transpose(1,2).contiguous()
            mask.requires_grad = False
            
            return mask

        def make_src_mask(trg, src):
            pad_mask = make_pad_mask(trg, src)
            return pad_mask

        def transform(x, fc):  # (n_batch, seq_len, d_embed)
            out = fc(x)        # (n_batch, seq_len, d_model)
            return out

        def calculate_attention(query, key, value, mask=None):
            d_k = key.shape[-1]
            attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, seq_len, seq_len)
            attention_score = attention_score / math.sqrt(d_k)
            if mask is not None:
                attention_score = attention_score.masked_fill(mask==0, -1e9)
            attention_prob = softmax(attention_score) # (n_batch, seq_len, seq_len)
            if mask is not None:
                attention_prob = attention_prob.masked_fill(mask==0, 0) # set zero to original zero index
            out = torch.matmul(attention_prob, value) # (n_batch, seq_len, d_k)
            return out

        query = transform(query, self.q_fc) # (n_batch, seq_len, d_k)
        key = transform(key, self.k_fc)     # (n_batch, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, seq_len, d_k)

        src_mask = make_src_mask(query, key)
        out = calculate_attention(query, key, value, src_mask) # (n_batch, seq_len, d_k)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed)
        
        return out


# +
class last_attention(nn.Module):
    def __init__(self,model,attention,d_embed=64):
        super(last_attention,self).__init__()
        self.model=model
        self.attention = Attention(d_embed=d_embed)
        self.d_embed = d_embed
        self.linear = nn.Linear(d_embed,7).to(device)
        self.relu = nn.ReLU().to(device)
        self.softmax = nn.Softmax(dim=-1).to(device)
        self.do = nn.Dropout(0.2).to(device)
        self.layer_norm = nn.LayerNorm(hidden_dim).to(device)

    def forward(self, text, audio, bMask=False):
        if len(text.shape) < 3:
            text = text.unsqueeze(0)
            audio = audio.unsqueeze(0)
        ta = self.model(text, audio, bMask)
        at = self.model(audio, text, bMask)
        total = self.attention(ta+at, bMask)
        
#         total = self.attention( self.layer_norm(self.do(total)) + self.layer_norm(self.do(text)), bMask )

        # MLP
        total = self.relu(self.linear(total))
        vote = torch.argmax(self.softmax(total), dim=-1)

        return self.softmax(total), vote.squeeze()


# -

# ### 전체 모델링 진행

# +
class F1_score(nn.Module):
    def __init__(self):
        super(F1_score, self).__init__()
    
    def forward(self, pred, real):
            loss = 0
            confusion_matrix = torch.zeros(7,7)
            
            for i in range(len(pred)):
                confusion_matrix[real[i],pred[i]] += 1
            
            n = torch.nan_to_num( confusion_matrix.sum(axis=1), nan=0.0)
            n2 = torch.nan_to_num( confusion_matrix.sum(axis=0), nan=0.0)
#             weight_ = confusion_matrix.sum(axis=1) / confusion_matrix.sum()
            weight_ = n
            weight_[weight_ == float("Inf")] = 0
            weight_ = torch.nan_to_num(weight_, nan=0.0)
            weight_ = weight_ / weight_.sum() # 출현 갯수만큼 가중치 주기
            
            recall_ = torch.diag(confusion_matrix,0) / n
            precision_ = torch.diag(confusion_matrix,0) / n2
            
            # na to zero
            recall_ = torch.nan_to_num(recall_, nan=0.0)
            precision_ = torch.nan_to_num(precision_, nan=0.0)
            
#             loss = (recall_ * weight_).sum()
#             loss = (recall_).mean()
    
            loss = 2*precision_*recall_ / (precision_ + recall_)
            loss = torch.nan_to_num(loss,nan=0.0).mean()
            loss = ( torch.nan_to_num(loss,nan=0.0) * weight_).sum()
            
            loss = 1-loss
            
            if n2.eq(0).sum() > 5:
                loss += loss*weight_[int(n2.nonzero())]*100
            
            return loss


# +
class BuildModels(nn.Module):
    def __init__(self, inputprocessing_text, inputprocessing_audio, tsmodel, cmt, attention, last_attention):
        super(BuildModels, self).__init__()
        self.ip_text = inputprocessing_text
        self.ip_audio = inputprocessing_audio
        self.tsmodel = tsmodel
        self.cmt = cmt
        self.attention = attention
        self.la = la
        
    
    def forward(self, kobert_input_tensor, mfcc_input_list, gender_tensor, emotion_tensor, valence_tensor, arousal_tensor, bTest=False):
        
        def data_(kobert_input_tensor, mfcc_input_list, gender_tensor, emotion_tensor, valence_tensor, arousal_tensor):

            ## kobert -> pp_text
            pp_text = self.ip_text(kobert_input_tensor.to(device))

            ## mfcc -> pp_audio
            pp_audio = torch.zeros(len(mfcc_input_list),6,max_len,hidden_dim)

            for i in range(len(mfcc_input_list)):
                for j in range(6):
                    for k in range(len(mfcc_input_list[i][j])):
                        temp = torch.tensor(mfcc_input_list[i][j][k], dtype=torch.float32)
                        pp_audio[i,j,k,:] = self.ip_audio(temp).T
                        del temp
                torch.cuda.empty_cache()

            pp_audio = pp_audio.contiguous().reshape(-1,max_len,hidden_dim) # (192, 128, 64) => (batch size, max_len, hidden_dim)
            pp_text = pp_text.contiguous().reshape(-1,max_len,hidden_dim) # (192, 128, 64) => (batch size, max_len, hidden_dim)
            tr_gender_tensor = gender_tensor.contiguous().reshape(-1,max_len) # (192, 128) => (batch size, max_len) 
            tr_emotion_tensor = emotion_tensor.contiguous().reshape(-1,max_len) # (192, 128) => (batch size, max_len) 
            tr_valence_tensor = valence_tensor.contiguous().reshape(-1,max_len) # (192, 128) => (batch size, max_len) 
            tr_arousal_tensor = arousal_tensor.contiguous().reshape(-1,max_len) # (192, 128) => (batch size, max_len) 

            return pp_audio.to(device), pp_text.to(device), tr_gender_tensor.to(device), tr_emotion_tensor.to(device), tr_valence_tensor.to(device), tr_arousal_tensor.to(device)
        
        ###############
        
        def focal_loss(prob,real_emo,alpha=0.25,gamma=-6):
            prob = prob.squeeze()
            s_p = prob[real_emo]
            f_l = -alpha * torch.pow(1-s_p,gamma) * torch.log(s_p)
#             f_l = -alpha * torch.pow(1-s_p,gamma) * torch.log(s_p)
            
            return f_l
        
#         def f1_score(pred, real):
#             loss = 0
#             confusion_matrix = torch.zeros(7,7)
            
#             for i in range(len(pred)):
#                 confusion_matrix[real[i],pred[i]] += 1
            
#             n = torch.nan_to_num( confusion_matrix.sum(axis=1), nan=0.0)
#             n2 = torch.nan_to_num( confusion_matrix.sum(axis=0), nan=0.0)
# #             weight_ = confusion_matrix.sum(axis=1) / confusion_matrix.sum()
#             weight_ = 1/n
#             weight_[weight_ == float("Inf")] = 0
#             weight_ = torch.nan_to_num(weight_, nan=0.0)
#             weight_ = weight_ / weight_.sum() # 출현 갯수만큼 역가중치 주기
            
#             recall_ = torch.diag(confusion_matrix,0) / n
#             precision_ = torch.diag(confusion_matrix,0) / n2
            
#             # na to zero
#             recall_ = torch.nan_to_num(recall_, nan=0.0)
#             precision_ = torch.nan_to_num(precision_, nan=0.0)
            
# #             loss = (recall_ * weight_).sum()
# #             loss = (recall_).mean()
    
#             loss = 2*precision_*recall_ / (precision_ + recall_)
#             loss = torch.nan_to_num(loss,nan=0.0).mean()
#             loss = ( torch.nan_to_num(loss,nan=0.0) * weight_).sum()
            
#             loss = 1-loss
            
#             if n2.eq(0).sum() > 5:
#                 loss += loss*weight_[int(n2.nonzero())]*100
            
#             return loss
        
        def weighted_cross_entropy(prob, real_emo, alpha=1):
            prob = prob.squeeze()
            
            # 1. neutral or not
            s_p = 0
            
            if real_emo==3: s_p = prob[real_emo]
            else: s_p = 1 - prob[3]
            
            loss = -torch.log(s_p)
            
            # 2. not neutral, then focal loss
            
            if real_emo != 3:
                s_p = prob[real_emo]
                loss += -torch.log(s_p)
                
            return torch.tensor(loss,dtype=torch.long)
        
        ###############
        loss = 0
        n_cnt = 0
        accur = 0
        f1_score = F1_score()
        
        torch.manual_seed(1234)
        tr_audio, tr_text, tr_gender_tensor, tr_emotion_tensor, tr_valence_tensor, tr_arousal_tensor = data_(kobert_input_tensor, mfcc_input_list,
                                                                                                        gender_tensor, emotion_tensor,
                                                                                                        valence_tensor, arousal_tensor)
        tr_text, tr_audio = self.tsmodel(tr_text, tr_audio, tr_gender_tensor)
        
        ### resampling class start
        
        if bTest==False:
            tr_text2 = tr_text.reshape(-1,64).detach().numpy()
            tr_audio2 = tr_audio.reshape(-1,64).detach().numpy()
            tr_emotion_tensor2 = tr_emotion_tensor.reshape(-1).detach().numpy()

            print(tr_text2.shape)
            print(tr_audio2.shape)
            check_cnt = 0
            check = [i for i in range(7) if Counter(tr_emotion_tensor2)[i] < 4]
            if len(check) > 0:
                check_cnt = sum([Counter(tr_emotion_tensor2)[i] for i in range(7) if Counter(tr_emotion_tensor2)[i] < 4])
                check = ~np.isin(tr_emotion_tensor2,check)

                tr_text2 = tr_text2[check]
                tr_audio2 = tr_audio2[check]
                tr_emotion_tensor2 = tr_emotion_tensor2[check]

            temp_te = CustomDataset(tr_text2, tr_emotion_tensor2, n=3)
            temp_ae = CustomDataset(tr_audio2, tr_emotion_tensor2, n=3)

            tr_text2 = torch.zeros(len(temp_te),hidden_dim)
            tr_audio2 = torch.zeros(len(temp_te),hidden_dim)
            tr_emotion_tensor2 = torch.zeros(len(temp_te))

            for i in range(len(temp_te)):
                tr_text2[i] = torch.tensor(temp_te[i][0], dtype=torch.long)
                tr_audio2[i] = torch.tensor(temp_ae[i][0], dtype=torch.long)
                tr_emotion_tensor2[i] = torch.tensor(temp_te[i][1].astype(np.int32), dtype=torch.int32)

            div_ = len(temp_te)%(max_len)
            remove_idx = tr_emotion_tensor2.eq(-1).nonzero()
            
            if div_ > 0:
                remove_idx = remove_idx[range(div_)]
                remove_idx = set(torch.cat(list(set(remove_idx))).tolist())
                remove_idx = torch.tensor(list(set(range(tr_emotion_tensor2.shape[0])) - remove_idx))
                tr_text2 = torch.index_select(tr_text2, dim=0, index=remove_idx)
                tr_audio2 = torch.index_select(tr_audio2, dim=0, index=remove_idx)
                tr_emotion_tensor2 = torch.index_select(tr_emotion_tensor2, dim=0, index=remove_idx)
                
                
#                 tr_text2 = torch.cat([tr_text2,tr_text2[range(div_)]], dim=0)
#                 tr_audio2 = torch.cat([tr_audio2,tr_audio2[range(div_)]], dim=0)
#                 tr_emotion_tensor2 = torch.cat([tr_emotion_tensor2,tr_emotion_tensor2[range(div_)]], dim=0)
            
            tr_text = tr_text2.reshape(-1,max_len,hidden_dim)
            tr_audio = tr_audio2.reshape(-1,max_len,hidden_dim)
            tr_emotion_tensor = tr_emotion_tensor2.reshape(-1,max_len)
        
        ### resampling end
        
        prob, vote = self.la(tr_text.to(device),tr_audio.to(device), bMask=True)
    
        pred_emo_list = []
        real_emo_list = []
        pred_prob_list = []
        
        criterion = nn.CrossEntropyLoss()
        
        for i in range(prob.shape[0]):
            idx = (tr_emotion_tensor[i,:] != -1).nonzero() # 실제 감정 레이블 존재하는 행만 추출
            
            n_cnt += len(idx)
            temp_pred = []; temp_real = [];
            temp_prob = prob[i,idx,:].squeeze()
            temp_loss = 0
            for j in idx:
                ##### 손실함수 수정 필요. 가중치 적용하는 방식으로 직접 구현하기.
                loss += criterion(prob[i,j,:].to(device), tr_emotion_tensor[i,j].clone().detach().type(torch.long).to(device))
#                 loss += focal_loss(prob[i,j,:].to(device), torch.tensor(tr_emotion_tensor[i,j],dtype=torch.long).to(device),0.25,0.8)
#                 loss += weighted_cross_entropy(prob[i,j,:].to(device), torch.tensor(tr_emotion_tensor[i,j],dtype=torch.long).to(device), 1)
                accur += int(vote[i,j]==tr_emotion_tensor[i,j])
                temp_pred.append(int(vote[i,j].detach().numpy()))
                temp_real.append(int(tr_emotion_tensor[i,j].detach().numpy()))                
                pred_emo_list = pred_emo_list + temp_pred
                real_emo_list = real_emo_list + temp_real
#                 print(Counter(temp_real))
            
#             weight_ = 1 / torch.bincount(torch.tensor(temp_real,dtype=torch.int32))
#             weight_[weight_ == float("Inf")] = 0
#             if len(weight_) < 7: 
#                 tt = [ 0 for i in range(7-len(weight_))]
#                 weight_ = torch.tensor(weight_.tolist() + tt, dtype=torch.float32)
                    
#             weight_ = weight_ / weight_.sum()
            
#             criterion = nn.CrossEntropyLoss(weight=weight_).to(device)
#             loss += criterion(temp_prob, torch.tensor(temp_real, dtype=torch.long))
#             loss += f1_score(torch.tensor(temp_pred, dtype=torch.long),torch.tensor(temp_real, dtype=torch.long))
        
        pred_tensor = torch.tensor(pred_emo_list,dtype=torch.long)
        real_tensor = torch.tensor(real_emo_list,dtype=torch.long)
        
        print(Counter(pred_emo_list))
        print(Counter(real_emo_list))
        
#         loss += f1_score(pred_tensor, real_tensor)
        loss /= n_cnt
        accur /= n_cnt
        print(f"loss: {np.round(loss.detach().numpy(),4)}, accuracy: {round(accur,3)}")
        torch.cuda.empty_cache()
        gc.collect()
        
        return {'loss': loss, 'accur':accur, 'pred_emo':pred_emo_list, 'real_emo':real_emo_list}
# -


len( np.array([0,1]) )

# +
from itertools import chain

def evaluate(model, text_input_tensor, audio_input_list, gender_tensor, emotion_tensor, valence_tensor, arousal_tensor, step, jump=32):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        
        pred_emo = []
        real_emo = []
        for i in range(int(text_input_tensor.shape[0]/step)):
            n_start = 2*(i); n_end = 2*(i+1)
            ret = model(text_input_tensor[n_start:n_end,:,:,:], audio_input_list[n_start:n_end], 
                              gender_tensor[n_start:n_end], emotion_tensor[n_start:n_end],  
                              valence_tensor[n_start:n_end],  arousal_tensor[n_start:n_end], bTest=True)

            total_loss += ret['loss'].data
            pred_emo.append(ret['pred_emo'])
            real_emo.append(ret['real_emo'])
            
            torch.cuda.empty_cache()
            gc.collect()
        
        total_loss = round( float(total_loss.detach().numpy()), 3)
        pred_emo = torch.tensor(list(chain(*pred_emo)))
        real_emo = torch.tensor(list(chain(*real_emo)))
        
        accur = (pred_emo==real_emo).sum()/len(pred_emo)
        accur = round( float(accur.detach().numpy()), 3)
        
#         print(f'total_loss: {total_loss}, accuracy: {accur}')
        return {'loss': total_loss, 'accur': accur}


# +
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

step = int(batch_size/6)

ip_text = inputProcessing(output_dim=hidden_dim)
ip_audio = inputProcessing(output_dim=1)
tsmodel = TimeSeriesModel()
cmt = CrossModalTransformer()
attention = Attention(d_embed=64)
la = last_attention(cmt,attention,d_embed=64)

build_model = BuildModels(ip_text,ip_audio,tsmodel,cmt,attention,la)
optimizer = optim.AdamW(list(build_model.parameters()), lr = 1e-3)

# train

prog = tqdm(range(epochs))

for epoc in prog:
    
    total_loss = 0
    build_model.train()
    
    prog.set_description(f"Epoch - {epoc+1}")
    
    for i in range(int(32/step)):
        print(i)
        optimizer.zero_grad()
        n_start = step*(i); n_end = step*(i+1)
        ret = build_model(kobert_output[n_start:n_end,:,:,:], mfcc_output[n_start:n_end], gender_tensor[n_start:n_end], 
                          emotion_tensor[n_start:n_end],  valence_tensor[n_start:n_end],  arousal_tensor[n_start:n_end])

        total_loss += ret['loss'].data
        ret['loss'].requires_grad_(True)
        ret['loss'].backward()
        optimizer.step()

        torch.cuda.empty_cache()
        gc.collect()
        
    eval_ = evaluate(build_model,test_kobert_output, test_mfcc_output, gender_tensor[32:40],
                    emotion_tensor[32:40], valence_tensor[32:40], arousal_tensor[32:40], step, 32)
    
    prog.set_postfix(loss=round(eval_['loss']), acc=eval_['accur'])

# test


# -

torch.cuda.empty_cache()
gc.collect()


# #### 230403 월요일 to do list
# 0. audio all input 가져오기
# 1. all 0 이면 어떻게 처리할지 구성
# 2. loss ftn 정의 및 모델링 전체 구성
# 3. softmax 문제 해결
# 4. physical signal 우째할지
#
# 나중에 할것
# 1. input output 순서 맞는지 체크 -> 덜 중요
# 2. audio input check -> 덜 중요

# #### 230406 목요일 to do list
# 0. neutral 이외의 값을 얼마나 잘 맞추는지 가중치 줄 필요 있어보임 (W 따로 학습 시키도록? 하는것도 고려해서 가중치 주기.)
#     -> cross entropy loss ftn 직접 정의해서 구현하는게 좋을듯
# 1. physical signal 우째할지
# 2. 전체 모델링
#

# ## arousal, valence 예측

# +
class last_attention_ar_va(nn.Module):
    def __init__(self,model,attention,d_embed=64):
        super(last_attention_ar_va,self).__init__()
        self.model=model
        self.attention = Attention(d_embed=d_embed)
        self.d_embed = d_embed
        
        # define function
        self.linear = nn.Linear(d_embed,32).to(device)
        self.linear2 = nn.Linear(32,1).to(device)
        self.multi_linear = nn.Linear(4,1)
        self.relu = nn.ReLU().to(device)
        self.do = nn.Dropout(0.2).to(device)
        self.layer_norm = nn.LayerNorm(hidden_dim).to(device)
        self.sigmoid = nn.Softmax()

    def forward(self, text, audio, eda, ibi, skin_temp, bMask=False):
        if len(text.shape) < 3:
            text = text.unsqueeze(0)
            audio = audio.unsqueeze(0)
        ta = self.model(text, audio, bMask)
        at = self.model(audio, text, bMask)
        total = self.attention(ta+at, bMask)
        
#         total = self.attention( self.layer_norm(self.do(total)) + self.layer_norm(self.do(text)), bMask )

        # MLP
        total = self.relu(self.linear(total))
        total = self.relu(self.linear2(self.do(total)))
        total = torch.stack([total.squeeze(),eda,ibi,skin_temp], dim=2)
        total = self.multi_linear(self.do(total))

        return total


# +
# Concordance Correlation Coefficient

def ccc(y_true, y_pred):
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    var_true = torch.var(y_true)
    var_pred = torch.var(y_pred)
    sd_true = torch.std(y_true)
    sd_pred = torch.std(y_pred)

    rho = torch.mean((y_true - mean_true) * (y_pred - mean_pred)) / (sd_true * sd_pred)
    mean_cent_prod = torch.mean(y_true * y_pred) - mean_true * mean_pred
    return 2 * rho * sd_true * sd_pred / (var_true + var_pred + mean_cent_prod)


# -

class BuildModels_ar_va(nn.Module):
    def __init__(self, inputprocessing_text, inputprocessing_audio, tsmodel, cmt, attention, last_attention, bValence=False):
        super(BuildModels_ar_va, self).__init__()
        self.ip_text = inputprocessing_text
        self.ip_audio = inputprocessing_audio
        self.tsmodel = tsmodel
        self.cmt = cmt
        self.attention = attention
        self.la = last_attention
        self.bVal = bValence
    
    def forward(self, kobert_input_tensor, mfcc_input_list, gender_tensor, emotion_tensor, valence_tensor, arousal_tensor, 
                eda_tensor, ibi_tensor, temp_tensor, bTest=False):
        
        def data_(kobert_input_tensor, mfcc_input_list, gender_tensor, emotion_tensor, valence_tensor, arousal_tensor,
                  eda_tensor, ibi_tensor, temp_tensor):

            ## kobert -> pp_text
            pp_text = self.ip_text(kobert_input_tensor.to(device))

            ## mfcc -> pp_audio
            pp_audio = torch.zeros(len(mfcc_input_list),6,max_len,hidden_dim)

            for i in range(len(mfcc_input_list)):
                for j in range(6):
                    for k in range(len(mfcc_input_list[i][j])):
                        temp = torch.tensor(mfcc_input_list[i][j][k], dtype=torch.float32)
                        pp_audio[i,j,k,:] = self.ip_audio(temp).T
                        del temp
                torch.cuda.empty_cache()

            pp_audio = pp_audio.contiguous().reshape(-1,max_len,hidden_dim) # (192, 128, 64) => (batch size, max_len, hidden_dim)
            pp_text = pp_text.contiguous().reshape(-1,max_len,hidden_dim) # (192, 128, 64) => (batch size, max_len, hidden_dim)
            tr_gender_tensor = gender_tensor.contiguous().reshape(-1,max_len) # (192, 128) => (batch size, max_len) 
            tr_emotion_tensor = emotion_tensor.contiguous().reshape(-1,max_len) # (192, 128) => (batch size, max_len) 
            tr_valence_tensor = valence_tensor.contiguous().reshape(-1,max_len) # (192, 128) => (batch size, max_len) 
            tr_arousal_tensor = arousal_tensor.contiguous().reshape(-1,max_len) # (192, 128) => (batch size, max_len) 
            tr_eda_tensor = eda_tensor.contiguous().reshape(-1,max_len) # (192, 128) => (batch size, max_len) 
            tr_ibi_tensor = ibi_tensor.contiguous().reshape(-1,max_len) # (192, 128) => (batch size, max_len) 
            tr_temp_tensor = temp_tensor.contiguous().reshape(-1,max_len) # (192, 128) => (batch size, max_len) 

            return pp_audio.to(device), pp_text.to(device), tr_gender_tensor.to(device), tr_emotion_tensor.to(device), tr_valence_tensor.to(device), tr_arousal_tensor.to(device), tr_eda_tensor.to(device), tr_ibi_tensor.to(device), tr_temp_tensor.to(device)
        
        def my_loss(output, target,criterion):
            loss = criterion(output, target)  # 기본적인 MSE 손실
            loss += 100 * torch.mean(torch.clamp(output - 5, min=0) ** 2)  # 출력값이 5를 넘어서면 큰 패널티 부여
            loss += 100 * torch.mean(torch.clamp(-output, min=0) ** 2)  # 출력값이 0보다 작으면 큰 패널티 부여
            return loss
        
        ###############
        loss = 0
        n_cnt = 0
        accur = 0
        
        torch.manual_seed(1234)
        tr_audio, tr_text, tr_gender_tensor, tr_emotion_tensor, tr_valence_tensor, tr_arousal_tensor, tr_eda_tensor, tr_ibi_tensor, tr_temp_tensor = data_(kobert_input_tensor, mfcc_input_list,gender_tensor, emotion_tensor,valence_tensor, arousal_tensor, eda_tensor, ibi_tensor, temp_tensor)
        tr_text, tr_audio = self.tsmodel(tr_text, tr_audio, tr_gender_tensor)
        
        pred = self.la(tr_text,tr_audio, tr_eda_tensor, tr_ibi_tensor, tr_temp_tensor, bMask=True)
    
        pred_list = []
        real_list = []
        eval_fun = ccc
        criterion = nn.MSELoss()
        
        cnt = 0
        for i in range(pred.shape[0]):
            idx = (tr_emotion_tensor[i,:] != -1).nonzero() # 실제 감정 레이블 존재하는 행만 추출
            cnt += 1
            temp_pred = []
            temp_real = []
            
            for j in idx:
                temp_pred = temp_pred + pred[i,j].clone().tolist()
                if self.bVal: temp_real = temp_real + tr_valence_tensor[i,j].clone().tolist()
                else: temp_real = temp_real + tr_arousal_tensor[i,j].clone().tolist()

            loss += my_loss(torch.tensor(temp_pred).flatten(), torch.tensor(temp_real).flatten(), criterion)
            penalty = 5 * (torch.clamp(torch.tensor(temp_pred), min=0) ** 2 + torch.clamp(torch.tensor(temp_pred) - 5, max=0) ** 2)
            loss += penalty.mean()
#             loss += criterion(torch.tensor(temp_pred).flatten(), torch.tensor(temp_real).flatten())
            print(eval_fun(torch.tensor(temp_pred).flatten(), torch.tensor(temp_real).flatten()))

            
            pred_list.append(temp_pred)
            real_list.append(temp_real)
        
        loss /= cnt
        
        pred_tensor = torch.tensor(list(chain(*pred_list)))
        real_tensor = torch.tensor(list(chain(*real_list)))
        
        print(f'pred max: {pred_tensor.max()}, min: {pred_tensor.min()}')
        print(f'real max: {real_tensor.max()}, min: {real_tensor.min()}')
        
        print(f"loss: {round(loss.detach().tolist(),4)}")
        
        return {'loss': loss, 'pred':pred_list, 'real':real_list}

# +
from itertools import chain

def evaluate_ar_va(model, text_input_tensor, audio_input_list, gender_tensor, emotion_tensor, valence_tensor, arousal_tensor, 
                   eda_tensor, ibi_tensor, temp_tensor, step, jump=32):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        
        pred = []
        real = []
        for i in range(int(text_input_tensor.shape[0]/step)):
            n_start = step*(i); n_end = step*(i+1)
            ret = model(text_input_tensor[n_start:n_end,:,:,:], audio_input_list[n_start:n_end], 
                              gender_tensor[n_start:n_end], emotion_tensor[n_start:n_end],  
                              valence_tensor[n_start:n_end],  arousal_tensor[n_start:n_end], 
                              eda_tensor[n_start:n_end],  ibi_tensor[n_start:n_end], 
                              temp_tensor[n_start:n_end], bTest=True)

            total_loss += ret['loss'].data
            pred = pred + ret['pred']
            real = real + ret['real']
            
            torch.cuda.empty_cache()
            gc.collect()
        
        total_loss = round( float(total_loss.detach().numpy()), 3)
        pred = torch.tensor(list(chain(*pred)))
        real = torch.tensor(list(chain(*real)))
        
        return {'loss': total_loss,'pred':pred, 'real':real}


# +
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

step = int(batch_size/6)

ip_text = inputProcessing(output_dim=hidden_dim)
ip_audio = inputProcessing(output_dim=1)
tsmodel = TimeSeriesModel()
cmt = CrossModalTransformer()
attention = Attention(d_embed=64)
la = last_attention_ar_va(cmt,attention,d_embed=64)

build_model = BuildModels_ar_va(ip_text,ip_audio,tsmodel,cmt,attention,la)
optimizer = optim.AdamW(list(build_model.parameters()), lr = 1e-5)

# train

prog = tqdm(range(epochs))

for epoc in prog:
    
    total_loss = 0
    build_model.train()
    
    prog.set_description(f"Epoch - {epoc+1}")
    
    for i in range(int(32/step)):
        print(i)
        optimizer.zero_grad()
        n_start = step*(i); n_end = step*(i+1)
        ret = build_model(kobert_output[n_start:n_end,:,:,:], mfcc_output[n_start:n_end], gender_tensor[n_start:n_end], 
                          emotion_tensor[n_start:n_end],  valence_tensor[n_start:n_end],  arousal_tensor[n_start:n_end],
                          eda_tensor[n_start:n_end],  ibi_tensor[n_start:n_end],  temp_tensor[n_start:n_end])

        total_loss += ret['loss'].data
        ret['loss'].requires_grad_(True)
        ret['loss'].backward()
        optimizer.step()

        torch.cuda.empty_cache()
        gc.collect()
        
        print(list(build_model.parameters()))
        
    print("\n Evaluate Start")
    eval_ = evaluate_ar_va(build_model,test_kobert_output, test_mfcc_output, gender_tensor[32:40],
                    emotion_tensor[32:40], valence_tensor[32:40], arousal_tensor[32:40], 
                    eda_tensor[32:40], ibi_tensor[32:40], temp_tensor[32:40], 
                           step, 32)
    
    prog.set_postfix(loss=round(eval_['loss']))
    print("\n Evaluate End\n Train Start\n")

# test
