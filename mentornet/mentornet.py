import torch
import torch.nn as nn
import numpy as np

class MentorNet(nn.Module):
    def __init__(self, label_embedding_size=2, epoch_embedding_size=5, num_fc_nodes=20):
        super(MentorNet, self).__init__()
        self.label_embedding_size = label_embedding_size
        self.epoch_embedding_size = epoch_embedding_size
        self.num_fc_nodes = num_fc_nodes
        
        self.label_embedding = nn.Parameter(torch.Tensor(2, label_embedding_size))
        nn.init.xavier_uniform_(self.label_embedding)
        
        self.epoch_embedding = nn.Parameter(torch.Tensor(100, epoch_embedding_size))
        nn.init.xavier_uniform_(self.epoch_embedding)
        self.epoch_embedding.requires_grad = False
        
        self.forward_cell = nn.LSTM(2, 1, batch_first=True, bidirectional=True)
        self.fc_1 = nn.Linear(label_embedding_size + epoch_embedding_size + 2, num_fc_nodes)
        self.fc_2 = nn.Linear(num_fc_nodes, 1)
        self.activation = nn.Tanh()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def forward(self, input_features):
        batch_size = input_features.size(0)

        losses = input_features[:, 0].view(-1, 1)

        loss_diffs = input_features[:, 1].view(-1, 1)

        labels = input_features[:, 2].view(-1, 1).long()

        epochs = input_features[:, 3].view(-1, 1).long()
        #print(epochs.device)
        #epochs = torch.min(epochs, torch.ones([batch_size, 1], dtype=torch.int32).to(device) * 99)

        if losses.dim() <= 1:
            num_steps = 1
        else:
            num_steps = losses.size(1)

        lstm_inputs = torch.stack([losses, loss_diffs], dim=2)

        #lstm_inputs = lstm_inputs.squeeze(1)
        

        _, (h_n, c_n) = self.forward_cell(lstm_inputs)
        h_forward = h_n[0,:,:] 
        h_backward = h_n[1,:,:] 
        # print(h_forward.shape)
        # print(h_backward.shape)
        out = torch.concat((h_forward,h_backward),1)
        # print(out.shape)


        
        label_inputs = self.label_embedding[labels.squeeze()]
        epoch_inputs = self.epoch_embedding[epochs.squeeze()]

        feat = torch.cat([label_inputs, epoch_inputs, out], dim=1)


        fc_1 = self.activation(self.fc_1(feat))
        out_layer = self.fc_2(fc_1)

        return out_layer