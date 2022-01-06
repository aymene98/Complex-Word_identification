import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import DistilBertModel

class BertClassifier(nn.Module):
    def __init__(self, device):
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.decision = nn.Linear(self.bert.config.hidden_size, 1)
        self.to(device)

    def forward(self, x, attention_map):
        output = self.bert(x, attention_mask = attention_map)
        return self.decision(torch.max(output[0], 1)[0])
