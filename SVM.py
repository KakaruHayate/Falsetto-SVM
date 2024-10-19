import torch
import torch.nn as nn
import torch.nn.functional as F


class SVM(nn.Module):
    """
    Linear Support Vector Machine
    -----------------------------
    This SVM is a subclass of the PyTorch nn module that
    implements the Linear  function. The  size  of  each
    input sample is input_dim and output sample  is output_dim.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()  # Call the init function of nn.Module
        self.fully_connected = nn.Linear(input_dim, output_dim)  # Implement the Linear function

    def forward(self, x):
        fwd = self.fully_connected(x.transpose(1, 2))  # Forward pass
        return fwd


class HingeLoss(nn.Module):
	# HingeLoss for SVM
	# Is not 'nn.HingeEmbeddingLoss'
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_true = y_true.unsqueeze(-1)
        loss = F.relu(1 - y_true * y_pred)
        mean_loss = torch.mean(loss)
        return mean_loss
