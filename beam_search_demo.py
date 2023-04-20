# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 00:17:51 2023

@author: lenovo
"""

import torch
import torch.nn.functional as F

def beam_search(LM_prob,beam_size = 3):
    batch,seqlen,vocab_size = LM_prob.shape
    #对LM_prob取对数
    log_LM_prob = LM_prob.log()
    #先选择第0个位置的最大beam_size个token，log_emb_prob与indices的shape为(batch,beam)
    log_beam_prob, indices = log_LM_prob[:,0,:].topk(beam_size,sorted = True)
    indices = indices.unsqueeze(-1)
    #对每个长度进行beam search
    for i in range(1,seqlen):
        #log_beam_prob (batch,beam,vocab_size),每个beam的可能产生的概率
        log_beam_prob = log_beam_prob.unsqueeze(-1) + log_LM_prob[:,i,:].unsqueeze(1).repeat(1,beam_size,1)
        #选择当前步概率最高的token
        log_beam_prob, index = log_beam_prob.view(batch,-1).topk(beam_size,sorted = True)
        #下面的计算：beam_id选出新beam来源于之前的哪个beam;index代表真实的token id
        #beam_id,index (batch,beam)
        beam_id = index//vocab_size
        index = index%vocab_size
        mid = torch.Tensor([])
        #对batch内每个样本循环，选出beam的同时拼接上新生成的token id
        for j,bid,idx in zip(range(batch),beam_id,index):
            x = torch.cat([indices[j][bid],idx.unsqueeze(-1)],-1)
            mid = torch.cat([mid,x.unsqueeze(0)],0)
        indices = mid
    return indices,log_beam_prob

if __name__=='__main__':
    # 建立一个语言模型 LM_prob (batch,seqlen,vocab_size)
    LM_prob = F.softmax(torch.randn([32,20,1000]),dim = -1)
    #最终返回每个候选，以及每个候选的log_prob，shape为(batch,beam_size,seqlen)
    indices,log_prob = beam_search(LM_prob,beam_size = 3)
    print(indices)