import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):

    # usually, the first parameter would be batch size
    def __init__(self, d_model=2, row_dim=0, col_dim=1):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, token_encodings):
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)

        similarities = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_similarities = similarities / torch.tensor(k.size(self.col_dim) ** .5)

        attention_percents = F.softmax(scaled_similarities, dim=self.col_dim)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores

# testing
encodings_matrix = torch.tensor([[1.16, .23], [.57, 1.36], [4.41, -2.16]])

torch.manual_seed(42)

selfAttention = SelfAttention()

output = selfAttention(encodings_matrix)

print(output)
