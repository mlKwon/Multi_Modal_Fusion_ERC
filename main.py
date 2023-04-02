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

# +
import os

os.chdir("G:/내 드라이브/2023aifactory")
os.getcwd()
# -

# ### 0-1) Declare Global variable

max_len = 128 # maximum number of conversation in each script
hidden_dim = 64 # number of hidden state
device = 'cuda'

# ### 0-2) read annotation file

# +
import numpy as np
import pandas as pd
import glob, copy, gc
from copy import deepcopy
import torch
from torch import nn

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
frame = torch.zeros(40,6,max_len)
gender_tensor = deepcopy(frame).to(device)
emotion_tensor = deepcopy(frame).to(device)
valence_tensor = deepcopy(frame).to(device)
arousal_tensor = deepcopy(frame).to(device)

for i in range(40):
#     dirpath = os.path.join(os.getcwd(),dir_list[i])
    annot = pd.read_csv(os.getcwd() + "/annotation/Sess"+str(i+1).zfill(2)+"_eval.csv")
    id_ = annot.iloc[1:,3].tolist()
    gender = [1 for i in range(len(id_))]
    for j in range(len(id_)):
        if str.find(id_[j],"M") > 0:
            gender[j] = 0 # 남자

    emotion = annot.iloc[1:,4]
    emotion[emotion.str.contains(";")] = emotion[emotion.str.contains(";")].str.split(";").str[0] # Choose first emotion when there are two emotions.
    emotion = emotion.tolist()
    valence = [float(i) for i in annot.iloc[1:,5].tolist()]
    arousal = [float(i) for i in annot.iloc[1:,6].tolist()]

    for j in range(1,7):
        j_idx = [id_.index(s) for s in id_ if "script"+str(j).zfill(2) in s]
        id_list[i][j-1] = [id_[s] for s in j_idx]
        # list -> tensor
        n_ = len([gender[s] for s in j_idx])
        gender_tensor[i,j-1,:n_] = torch.tensor([gender[s] for s in j_idx], dtype=torch.int32)
        emotion_tensor[i,j-1,:n_] = torch.tensor([dic_[emotion[s]] for s in j_idx], dtype=torch.int32)
        valence_tensor[i,j-1,:n_] = torch.tensor([valence[s] for s in j_idx], dtype=torch.float32)
        arousal_tensor[i,j-1,:n_] = torch.tensor([arousal[s] for s in j_idx], dtype=torch.float32)

    del id_, gender, emotion, valence, arousal

gc.collect()
# -

# ## 1) Input preprocessing

# ### 1-1) Text input by KoBERT (worked in colab should be loaded.)

# +
import torch
from torch import nn
device = torch.device("cuda")
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    device = torch.device('cuda')

print(device)

# +
# kobert_output = torch.zeros([32,6,max_len,768]).to(device) # (train session, scripts, max_conversation_length, output_dim_of_KoBERT)
# kobert_output = torch.zeros([32,6,max_len,768]).to(device) # (train session, scripts, max_conversation_length, output_dim_of_KoBERT)
kobert_output = torch.tensor(np.load("G:/내 드라이브/2023aifactory/wav_text/temp_session01.npy")) # read only train text
print(kobert_output.shape)

for i in range(1,32):
    kobert_output[i,:,:,:] = torch.tensor(np.load("G:/내 드라이브/2023aifactory/wav_text/temp_session"+str(i+1).zfill(2)+".npy"),
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

# np.save("G:/내 드라이브/2023aifactory/wav_text/temp_session01", kobert_output.cpu().detach().numpy() )
# kobert_output = torch.tensor(np.load("G:/내 드라이브/2023aifactory/wav_text/temp_session01.npy"))


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
    mfccs = torch.FloatTensor(mfccs).transpose(0,1)
    return mfccs

# from sklearn.decomposition import PCA
# pca = PCA(n_components=1)


# -

scripts_ = [[] for i in range(6)]
mfcc_output = [deepcopy(scripts_) for i in range(40)]

# +
import glob, copy
from copy import deepcopy
import os
import glob
import gc
import pandas as pd
import re

# for i in range(40): # call all session with train(32) and test(8)
# for i in range(1): # call only 1 session temporalry
for i in range(17,32): # call only 32 session for training
    print(f"session {i+1}")
    
    for j in range(6):
        print(f"script {j+1}", end=" ")
        for k in range(len(id_list[i][j])): 
            filepath =  os.getcwd() + "/wav/Session"+str(i+1).zfill(2)+"/"+id_list[i][j][k]+".wav"
            temp = get_librosa_mfcc(filepath)
            temp = preprocessing.scale(temp, axis=0).T
            mfcc_output[i][j].append(temp)
            del temp
    
    gc.collect()
    print()
# -

mfcc_output[17][1]


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



# ### model로 전체 빌딩 되어야 하는 부분 ### 수정 필요

# +

input_p_text = inputProcessing().to(device)
input_p_audio = inputProcessing(output_dim=1).to(device)

## kobert -> pp_text
pp_text = input_p_text(kobert_output.to(device))

## mfcc -> pp_audio
pp_audio = torch.zeros(32,6,max_len,hidden_dim)

for i in range(17):
    print(f"session {i+1}")
    for j in range(6):
        print(f"script {j+1}", end=" ")
        for k in range(len(mfcc_output[i][j])):
            temp = torch.tensor(mfcc_output[i][j][k], dtype=torch.float32)
            pp_audio[i,j,k,:] = input_p_audio(temp).T
            del temp
    print()
    torch.cuda.empty_cache()

gc.collect()

# pp_text

# aggregate to 3-dimension all tensors

pp_audio = pp_audio.contiguous().view(-1,max_len,output_dim) # (192, 128, 64) => (batch size, max_len, hidden_dim)
pp_text = pp_text.contiguous().view(-1,max_len,output_dim) # (192, 128, 64) => (batch size, max_len, hidden_dim)
tr_gender_tensor = gender_tensor[:32,:,:].contiguous().view(-1,max_len) # (192, 128) => (batch size, max_len) 
tr_emotion_tensor = emotion_tensor[:32,:,:].contiguous().view(-1,max_len) # (192, 128) => (batch size, max_len) 
tr_valence_tensor = valence_tensor[:32,:,:].contiguous().view(-1,max_len) # (192, 128) => (batch size, max_len) 
tr_arousal_tensor = arousal_tensor[:32,:,:].contiguous().view(-1,max_len) # (192, 128) => (batch size, max_len) 

# -

np.save("G:/내 드라이브/2023aifactory/wav_audio/temp_session1_17",pp_audio.cpu().detach().numpy())

torch.cuda.empty_cache()
gc.collect()
torch.cuda.memory_reserved()

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
            if self.bContext==True: # context-level
                if input_.count_nonzero() > 0: # 정상 utterance vector
                    self.h0 = self.gru_0(input_,self.h0)
                    self.h1 = self.gru_1(self.h0,self.h1)
                    self.h2 = self.gru_2(self.h1,self.h2)
                    outs.append(self.h2.unsqueeze(1))
                else: # utterance vector 값 없을 때 (ex. audio in session1)
                    outs.append(torch.zeros_like(self.h1.unsqueeze(1)))
            else: # self-context level
                if input_.count_nonzero() > 0: # 정상 utterance vector
                    if gender[i]==0: # speaker gender
                        self.h0 = self.gru_0(input_,self.h0)
                        self.h1 = self.gru_1(self.h0,self.h1)
                        self.h2 = self.gru_2(self.h1,self.h2)
                    else: 
                        self.h0_F = self.gru_0(input_,self.h0_F)
                        self.h1_F = self.gru_1(self.h0_F,self.h1_F)
                        self.h2_F = self.gru_2(self.h1_F,self.h2_F)
                    outs.append(self.h2_F.unsqueeze(1))
                else: # utterance vector 값 없을 때 (ex. audio in session1)
                    outs.append(torch.zeros_like(self.h1.unsqueeze(1)))
            
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

        text_input_c = model(text_input.to('cpu')) # (198,128,64) => (session * script, seq_len, hidden_dim)
        audio_input_c = model(audio_input.to('cpu')) # (198,128,64) => (session * script, seq_len, hidden_dim)
        
        text_input_s = torch.zeros_like(text_input_c)
        audio_input_s = torch.zeros_like(audio_input_c)
        
        for i in range(text_input.shape[0]):
            text_input_s[i,:,:] = model_s(text_input[i,:,:].unsqueeze(dim=0).to('cpu'), gender=gender[i,:].to('cpu'))
            audio_input_s[i,:,:] = model_s(audio_input[i,:,:].unsqueeze(dim=0).to('cpu'), gender=gender[i,:].to('cpu'))
            
        gc.collect()

        return (text_input_c+text_input_s)/2, (audio_input_c+audio_input_s)/2


# -

# ### model로 전체 빌딩 되어야 하는 부분 ### 수정 필요

tsmodel = TimeSeriesModel()
pp_text, pp_audio = tsmodel(pp_text, pp_audio, tr_gender_tensor)

# ## 3) Cross-Attention

# ### 3-1) Multi-head Attention

device='cpu'


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
            out = out.view(n_batch, -1, self.h, self.d_embed) # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out

        def calculate_attention(query, key, value, mask):
          # query, key, value: (n_batch, h, seq_len, d_k)
          # mask: (n_batch, 1, seq_len, seq_len)
            d_k = key.shape[-1]
            attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, h, seq_len, seq_len)
            attention_score = attention_score / math.sqrt(d_k)
            if mask is not None:
                attention_score = attention_score.masked_fill(mask==0, -1e9)
            attention_prob = softmax(attention_score) # (n_batch, h, seq_len, seq_len)
            out = torch.matmul(attention_prob, value) # (n_batch, h, seq_len, d_k)
            return out

        query = transform(query.to(device), self.q_fc) # (n_batch, h, seq_len, d_k)
        key = transform(key.to(device), self.k_fc)     # (n_batch, h, seq_len, d_k)
        value = transform(value.to(device), self.v_fc) # (n_batch, h, seq_len, d_k)

        out = calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed)
        return out


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
        
    def forward(self, trg, src, src_mask=None):
        
        layer_norm = nn.LayerNorm(trg.shape[-1]).to(device)
        out = layer_norm(trg)
        src = layer_norm(src)
        out = self.self_attention(query=out, key=src, value=src, mask=src_mask)
        out = self.residuals[0](layer_norm(self.do(out)))
        out = self.position_ff(layer_norm(self.do(out)))
        out = self.residuals[1](layer_norm(self.do(out)))
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
        def make_pad_mask(query, key, pad_idx=1):
            # query: (n_batch, query_seq_len)
            # key: (n_batch, key_seq_len)
            query_seq_len, key_seq_len = query.size(1), key.size(1)

            key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
            key_mask = key_mask.repeat(1, 1, query_seq_len, 1)    # (n_batch, 1, query_seq_len, key_seq_len)

            query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
            query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

            mask = key_mask & query_mask
            mask.requires_grad = False
            
            return mask

        def make_src_mask(trg, src):
            pad_mask = make_pad_mask(trg, src)
            return pad_mask

        # src_mask = make_src_mask(query, key)
        src_mask = None
        if bMask:
            out = self.encoder(query, key, src_mask)
        else:
            out = self.encoder(query, key, None) 
    
        return out


class Model(nn.Module):

    def __init__(self, d_embed=64, h=8, n_layer=5):
        super(Model, self).__init__()
    # declare
        self.attention = MultiHeadAttentionLayer(d_embed = 64, h = 8)
        self.positionff = PositionWiseFeedForwardLayer(d_k=64, d_ff=64)
        self.residual = ResidualConnectionLayer()
        self.encoder_layer = EncoderBlock(self.attention, self.positionff)
        self.encoder = Encoder(self.encoder_layer,n_layer)
        self.model = Transformer(self.encoder)

    def forward(self, query, key):
        out = self.model(query, key)
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
            out = torch.matmul(attention_prob, value) # (n_batch, seq_len, d_k)
            return out

        query = transform(query, self.q_fc) # (n_batch, seq_len, d_k)
        key = transform(key, self.k_fc)     # (n_batch, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, seq_len, d_k)

        out = calculate_attention(query, key, value, mask) # (n_batch, seq_len, d_k)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed)

        return out


class last_attention(nn.Module):
    def __init__(self,model,attention,d_embed=64):
        super(last_attention,self).__init__()
        self.model=model
        self.attention = Attention(d_embed=d_embed)
        self.d_embed = d_embed
        self.linear = nn.Linear(d_embed,7).to(device)
        self.relu = nn.ReLU().to(device)
        self.softmax = nn.Softmax(dim=-1).to(device)

    def forward(self,text, audio):
        if len(text.shape) < 3:
            text = text.unsqueeze(0)
            audio = audio.unsqueeze(0)
        ta = self.model(text, audio)
        at = self.model(audio, text)
        total = self.attention(ta+at)
        total = self.attention( total + text )

        # MLP
        total = self.relu(self.linear(total))
        vote = torch.argmax(self.softmax(total), dim=-1)

        return self.softmax(total), vote.squeeze()



# ### model로 전체 빌딩 되어야 하는 부분 ### 수정 필요

model = Model()
attention = Attention(d_embed=64)
la = last_attention(model,attention,d_embed=64)
output, vote = la(pp_text,pp_audio)
gc.collect()

vote[2,:]

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
