
'''
create by Nitrogen_pump 2022/2/25
'''

import math
import torch
from torch import nn
import numpy as np
import torch.optim as optim

def get_vocab(sentence):
    src_vocab = {'P':0}
    tg_vocab = {'S':0}
    temp = sentence[0].split()
    for j in range(len(temp)):
        if temp[j] not in src_vocab:
            src_vocab[temp[j]] = len(src_vocab)
    temp2 = sentence[1].split()
    for i in range(len(temp2)):
        if temp2[i] not in tg_vocab:
            tg_vocab[temp2[i]] = len(tg_vocab)
    tg_vocab['E'] = len(tg_vocab)
    return src_vocab,tg_vocab

def make_batch(sentence):
    Encoder_input = [[src_vocab[n] for n in sentence[0].split()]]
    Decoder_input = [[tg_vocab[n] for n in sentence[1].split()]]
    target = [[tg_vocab[n] for n in sentence[2].split()]]
    return torch.LongTensor(Encoder_input), torch.LongTensor(Decoder_input), torch.LongTensor(target)

def get_attn_mask_pad(seq):
    batch_size, len_data = seq.size()
    attn_pad = seq.data.eq(0).unsqueeze(1)
    attn_pad = attn_pad.expand(batch_size, len_data, len_data)
    return attn_pad #attn_pad:[1,6,6]

def get_attn_mask_pad_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask #subsequence_mask[1,6,6]

class PositionalEncoding(nn.Module):
    def __init__(self, Embedding_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, Embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, Embedding_size, 2).float() * (-math.log(10000.0) / Embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        ##pe形状是：[max_len*1*Embedding_size]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        # x = x.unsqueeze(-1).reshape(6 ,1, 512)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    def forward(self,s_q,s_k,s_v,attn_mask):
        score = torch.matmul(s_q,s_k.transpose(2,3)) / np.sqrt(K_size)
        score = score.masked_fill_(attn_mask, -1e9)
        score = nn.Softmax(dim=1)(score)
        score = torch.matmul(score,s_v)
        return score #score[1,8,6,64]

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(Embedding_size, Q_size*n_head)
        self.W_K = nn.Linear(Embedding_size, K_size*n_head)
        self.W_V = nn.Linear(Embedding_size, V_size*n_head)
        self.Linear = nn.Linear(Q_size*n_head, Embedding_size)
        self.norm = nn.LayerNorm(Embedding_size)
    def forward(self, Q, K, V, attn_mask):
        temp = Q
        batch_size, len_seq, _ = Q.size()
        s_q = self.W_Q(Q).unsqueeze(1).reshape(batch_size, n_head, len_seq, Q_size) #s_q:[1,8,6,64]
        s_k = self.W_K(K).unsqueeze(1).reshape(batch_size, n_head, len_seq, K_size) #s_k:[1,8,6,64]
        s_v = self.W_V(V).unsqueeze(1).reshape(batch_size, n_head, len_seq, V_size) #s_v:[1,8,6,64]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1) #attn_mask:[1,8,6,6]

        attn_score = ScaledDotProductAttention()(s_q,s_k,s_v,attn_mask) #attn_score:[1,8,6,64]
        attn_score = attn_score.transpose(1,2).squeeze(2).reshape(batch_size,len_seq,Q_size*n_head) #attn_score:[1,6,512]
        attn_score = self.Linear(attn_score) #attn_score:[1,6,512]
        attn_score = self.norm(attn_score+temp) #attn_score:[1,6,512]
        return attn_score #attn_score:[1,6,512]

class FeedForwardPosition(nn.Module):
    def __init__(self):
        super(FeedForwardPosition, self).__init__()
        self.conv1 = nn.Conv1d(Embedding_size,FF_dimension, kernel_size = 1)
        self.conv2 = nn.Conv1d(FF_dimension,Embedding_size, kernel_size = 1)
        self.norm = nn.LayerNorm(Embedding_size)
    def forward(self,attn_score):
        temp = attn_score
        output= self.conv1(attn_score.transpose(1,2))
        output = self.conv2(output).transpose(1,2)
        output = self.norm(output + temp)
        return output #output:[1,6,512]

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.Multi_attn = MultiHeadAttention()
        self.FeedForward = FeedForwardPosition()
    def forward(self,en_embed, attn_mask):
        attn_score = self.Multi_attn(en_embed, en_embed, en_embed, attn_mask) #attn_score:[1,6,512]
        output = self.FeedForward(attn_score) #output:[1,6,512]
        return output #output:[1,6,512]

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(length_srcv, Embedding_size)
        self.position = PositionalEncoding(Embedding_size)
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(n_layers))
    def forward(self, Encoder_input):
        en_embed = self.embed(Encoder_input) #en_embed:[1, 6, 512]
        en_embed = self.position(en_embed.transpose(0,1)).transpose(0,1) #en_embed:[1, 6, 512]

        attn_mask =get_attn_mask_pad(Encoder_input) #attn_mask:[1, 6, 6]
        for layer in self.layers:
            output = layer(en_embed, attn_mask) #output:[1,6,512]
        return output #output:[1,6,512]

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.mask_Mutil_attn = MultiHeadAttention()
        self.Mutil_attn = MultiHeadAttention()
        self.FeedForward = FeedForwardPosition()
    def forward(self,de_embed, Encoder_output, attn_mask, attn_mask_mask):
        all_mask = torch.gt((attn_mask+attn_mask_mask), 0)
        mask_attn = self.mask_Mutil_attn(de_embed, de_embed, de_embed, all_mask)
        attn_score =self.Mutil_attn(de_embed, Encoder_output,Encoder_output, attn_mask)
        output = self.FeedForward(attn_score)
        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(length_tgv, Embedding_size)
        self.position = PositionalEncoding(Embedding_size)
        self.layers = nn.ModuleList(DecoderLayer() for _ in range(n_layers))
    def forward(self,Encoder_output,Decoder_input):
        de_embed = self.embed(Decoder_input) #de_embed:[1,6,512]
        de_embed = self.position(de_embed.transpose(0,1)).transpose(0,1) #de_embed:[1,6,512]

        attn_mask = get_attn_mask_pad(Decoder_input) #attn_mask:[1,6,5]
        attn_mask_mask = get_attn_mask_pad_mask(Decoder_input) #attn_mask_mask:[1,6,5]

        for layer in self.layers:
            output = layer(de_embed, Encoder_output, attn_mask, attn_mask_mask) #output:[1,6,512]
        return output

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.Linear_projection = nn.Linear(Embedding_size, length_tgv)
    def forward(self, Encoder_input, Decoder_input):
        Encoder_output = self.encoder(Encoder_input)
        decoder_output = self.decoder(Encoder_output,Decoder_input)
        output = self.Linear_projection(decoder_output)
        return output.view(-1,output.size(-1)) #output:[6,7]


if __name__ == '__main__':
    #参数
    Embedding_size = 512 #词向量维度
    FF_dimension = 2048 #前馈层维度
    K_size = Q_size = V_size = 64 #注意力的QKV的维度
    n_layers = 6 #编码器与解码器的层数
    n_head = 8 #多头注意力的头数

    max_len = 5

    sentence = ['我 必 须 走 了 P', 'S i have to leave now', 'i have to leave now E']
    src_vocab, tg_vocab = get_vocab((sentence))
    length_srcv, length_tgv = len(src_vocab), len(tg_vocab)

    Encoder_input, Decoder_input, target = make_batch(sentence)

    model = Transformer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    for i in range(20):
        output = model(Encoder_input, Decoder_input)
        loss = criterion(output, target.view(-1))
        print('epoch:',i,',','loss:', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

