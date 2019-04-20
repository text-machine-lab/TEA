import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from .gcl import EncapsulatedGCL

EMBEDDING_DIM = 300
DENSE_LABELS = True
MAX_LEN = 16  # max # of words on each branch of path

if DENSE_LABELS:
    LABELS = ["SIMULTANEOUS", "BEFORE", "AFTER", "IS_INCLUDED", "INCLUDES", "None"] # TimeBank Dense labels
else:
    LABELS = ["SIMULTANEOUS", "BEFORE", "AFTER", "IBEFORE", "IAFTER", "IS_INCLUDED", "INCLUDES",
              "BEGINS", "BEGUN_BY", "ENDS", "ENDED_BY", "None"]

def max_pool_time(x, bidirectinoal=True):
    """max pooling over time steps"""
    if bidirectinoal:
        b, t, d = x.shape
        x1, x2 = torch.split(x, d//2, dim=-1)
        x = x1 + x2

    return F.max_pool1d(x.permute(0, 2, 1), x.shape[-2]).squeeze()

def criterion(y_pred, y_true):
    return nn.CrossEntropyLoss()(y_pred, y_true)



class PairRelation(nn.Module):
    def __init__(self, vocab_size, nb_classes=6, input_dropout=0.5, word_embeddings=None):
        super().__init__()
        if word_embeddings is None:
            self.embedding = nn.Embedding(vocab_size + 1, EMBEDDING_DIM)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                torch.from_numpy(word_embeddings).type(torch.cuda.FloatTensor),freeze=True)

        self.emb_dropout = nn.Dropout(p=input_dropout)

        self.lstm_left = nn.LSTM(EMBEDDING_DIM, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_right = nn.LSTM(EMBEDDING_DIM, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_left_context = nn.LSTM(EMBEDDING_DIM, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_right_context = nn.LSTM(EMBEDDING_DIM, 128, num_layers=1, batch_first=True, bidirectional=True)

        # input size 128*4 + 1 + 3 = 516
        self.hidden1 = nn.Linear(516, 1024)
        self.hidden_dropout1 = nn.Dropout(p=0.5)
        self.hidden2 = nn.Linear(1028, 512)
        self.hidden_dropout2 = nn.Dropout(p=0.3)
        self.out = nn.Linear(512, nb_classes)

        self.optimizer = optim.RMSprop(self.parameters(), lr=0.001, weight_decay=0)

    def forward(self, X, truncate=False, **kwargs):
        left_input, right_input, type_markers, left_context_input, right_context_input, time_differences, *_ = X

        # print(left_input.shape)
        batch_size, chunk_size, time_steps = left_input.shape

        left_input = left_input.view(batch_size * chunk_size, -1)
        right_input = right_input.view(batch_size * chunk_size, -1)
        left_context_input = left_context_input.view(batch_size * chunk_size, -1)
        right_context_input = right_context_input.view(batch_size * chunk_size, -1)
        type_markers = type_markers.view(batch_size * chunk_size, -1).float()
        time_differences = time_differences.view(batch_size * chunk_size, -1).float()

        left_input_emb = self.emb_dropout(self.embedding(left_input))
        right_input_emb = self.emb_dropout(self.embedding(right_input))
        self.lstm_left.flatten_parameters()
        self.lstm_right.flatten_parameters()
        left_branch = max_pool_time(self.lstm_left(left_input_emb)[0])
        right_branch = max_pool_time(self.lstm_right(right_input_emb)[0])

        left_context_emb = self.emb_dropout(self.embedding(left_context_input))
        right_context_emb = self.emb_dropout(self.embedding(right_context_input))
        self.lstm_left_context.flatten_parameters()
        self.lstm_right_context.flatten_parameters()
        left_context = max_pool_time(self.lstm_left_context(left_context_emb)[0])
        right_context = max_pool_time(self.lstm_right_context(right_context_emb)[0])

        #XL, XR, type_markers, context_L, context_R, time_differences, labels, pair_index
        hidden_input = torch.cat((left_branch, right_branch, type_markers,
                                  left_context, right_context, time_differences), -1)
        hidden1 = self.hidden_dropout1(self.hidden1(hidden_input))
        hidden1 = F.relu(hidden1)
        if truncate:
            return hidden1, left_context, right_context

        hidden1 = torch.cat((hidden1, type_markers, time_differences), -1)
        hidden2 = self.hidden_dropout2(self.hidden2(hidden1))
        hidden2 = F.relu(hidden2)

        output = self.out(hidden2)

        return output

    def fit(self, X, y, epochs=1):
        self.train()
        y = y.view(-1, 1).squeeze()  # (batch_size * chunk_size, 1)

        for epoch in range(epochs):
            pred = self.forward(X)
            loss = criterion(pred, y)

            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        pred = pred.max(1)[1]
        # print(pred.eq(y)); sys.exit(1)
        acc = pred.eq(y).sum().float() / len(y)

        return loss.data.item(), acc.data.item()

    def predict(self, X, return_probs=True, **kwargs):
        self.eval()
        pred = self.forward(X)
        if return_probs:
            return pred.cpu().detach().numpy()

        return pred.max(1)[1].cpu().detach().numpy()


class GCLRelation(nn.Module):
    def __init__(self, pairwise_model, nb_classes=6, gcl_size=512, controller_size=512,
                 controller_layers=1, num_heads=1, num_slots=40, m_depth=512):
        super().__init__()
        self.pairwise_model = pairwise_model

        # freeze pre-trained model
        # for p in self.pairwise_model.parameters():
        #     p.requires_grad = False

        gcl_input_size = self.pairwise_model.hidden1.out_features + 1 + 3
        key_size = self.pairwise_model.lstm_left_context.hidden_size * 2
        print("GCL input size {}, key size {}".format(gcl_input_size, key_size))

        # (num_inputs, num_outputs, controller_size, controller_layers, num_heads, N, M, K)
        self.gcl = EncapsulatedGCL(gcl_input_size, gcl_size, controller_size,
                                   controller_layers, num_heads, num_slots, m_depth, key_size)

        self.hidden1 = nn.Linear(2 * gcl_size, 1024)
        self.drop1 = nn.Dropout(p=0.5)
        self.hidden2 = nn.Linear(1024, 512)
        self.drop2 = nn.Dropout(p=0.5)
        self.out = nn.Linear(512 + 4, nb_classes)

        self.optimizer = optim.RMSprop(self.parameters(), lr=0.001, weight_decay=0)

    def forward(self, X, save_attn=False):
        left_input, right_input, type_markers, left_context_input, right_context_input, time_differences, *_ = X
        type_markers = type_markers.float()
        time_differences = time_differences.float()

        batch_size, chunk_size, _ = X[0].shape

        self.gcl.init_sequence(batch_size)
        pairwise_feed, left_context, right_context = self.pairwise_model(X, truncate=True)
        pairwise_feed = pairwise_feed.view(batch_size, chunk_size, -1)

        pairwise_feed = torch.cat([pairwise_feed, type_markers, time_differences], -1)
        # print(pairwise_feed.shape, left_context.shape)

        key = torch.cat([left_context, right_context], -1).view(batch_size, chunk_size, -1)
        gcl_output = self.gcl(key, pairwise_feed, bidirectional=True, save_attn=save_attn)
        hidden1 = F.relu(self.drop1(self.hidden1(gcl_output)))
        hidden2 = F.relu(self.drop2(self.hidden2(hidden1)))
        hidden2 = torch.cat([hidden2, type_markers, time_differences], -1)

        output = self.out(hidden2)

        return output
        
    def fit(self, X, y, epochs=1):
        self.train()
        y = y.view(-1, 1).squeeze()  # (batch_size * chunk_size, 1)

        for epoch in range(epochs):
            pred = self.forward(X)
            pred = pred.view(-1, pred.shape[-1])
            # print("pred", pred.shape)
            loss = criterion(pred, y)

            # update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        pred = pred.max(1)[1]
        # print(pred.eq(y)); sys.exit(1)
        acc = pred.eq(y).sum().float() / len(y)

        return loss.data.item(), acc.data.item()

    def predict(self, X, return_probs=True, save_attn=False):
        self.eval()
        pred = self.forward(X, save_attn=save_attn)
        pred = pred.view(-1, pred.shape[-1])
        if return_probs:
            return pred.cpu().detach().numpy()

        return pred.max(1)[1].cpu().detach().numpy()