import torch.nn as nn

#model definition
class Net(nn.Module):
    def __init__(self,n_channels, time_steps, ff_dim, n_head, n_classes, n_layers, dropout):
        super(Net, self).__init__()

        self.transformer_model = nn.Transformer(d_model=n_channels, nhead=n_head, num_encoder_layers=n_layers,
                                                num_decoder_layers=n_layers, dim_feedforward=ff_dim, batch_first=True,dropout=dropout,device=device)
        self.fc = nn.Linear(time_steps*n_channels, n_classes)
        self.time_steps = time_steps
        self.n_channels = n_channels

    def forward(self, x):
        
        x = self.transformer_model(x,x)
        x = x.view(-1,self.time_steps*self.n_channels)
        x = self.fc(x)

        return x
