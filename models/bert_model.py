"""Implement bert model"""
from transformers import AutoModel
import torch.nn as nn


class ClassificationBert(nn.Module):
    """Bert model for classification"""

    def __init__(self, pretrained='vinai/phobert-base', num_classes=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained)
        hidden_size = self.bert.config.hidden_size
        self.sequential = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        """Feedforward step"""
        pooled = self.bert(input_ids, attention_mask)[1]
        logits = self.sequential(pooled)
        return logits
