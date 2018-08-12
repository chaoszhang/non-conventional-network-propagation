import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def store_json(address, dic):
    with open(address, 'w') as fp:
        json.dump(dic, fp)

def load_json(address):
    with open(address, 'r') as f:
        return json.load(f)

class NLPropagation():
    def __init__(self, values, sources, targets, nIter = 10000, gpu = True, alpha = 0.56, beta = 0.0005, gamma = 1.0 / 3, shift = 3, power = 2):
        self.nIter = nIter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        n = len(values)
        p = (torch.t(torch.FloatTensor([values])) + shift) ** power
        
        count = [0] * n
        for v in sources:
            count[v] += 1
        weights = [1.0 / count[v] for v in sources]
        
        if gpu:
            self.p = p.cuda()
            self.a = torch.sparse.FloatTensor(torch.LongTensor([targets, sources]), torch.FloatTensor(weights)).cuda()
            self.b = torch.sparse.FloatTensor(torch.LongTensor([sources, targets]), torch.FloatTensor(weights)).cuda()    
        else:
            self.p = p
            self.a = torch.sparse.FloatTensor(torch.LongTensor([targets, sources]), torch.FloatTensor(weights))
            self.b = torch.sparse.FloatTensor(torch.LongTensor([sources, targets]), torch.FloatTensor(weights))
        
    def run(self):
        p = self.p
        p0 = p
        for i in range(self.nIter):
            p += self.beta * (self.alpha * (p0 - p) + (1 - self.alpha) * ((p ** self.gamma) * torch.matmul(self.a, p ** (1 - self.gamma)) - (p ** (1 - self.gamma)) * torch.matmul(self.b, p ** self.gamma)))
            print(p, sum(p.t().cpu().numpy().tolist()[0]))
        return p.t().cpu().numpy().tolist()[0]
    
if __name__ == '__main__':
    p_list, source_list, target_list = load_json(r'preprocessed_data.json')
    netprop = NLPropagation(p_list, source_list, target_list)
    output = netprop.run()
    store_json(r'non-linear-standard.json', output)
    print(output)