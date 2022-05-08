import torch
import torch.nn as nn

# Methods to deal with imbalanced dataset
# 1. Oversampling
# 2. Class Weighting

# Class Weighting

loss = nn.CrossEntropyLoss(weight=torch.tensor([1, 50]))

# This will penalize classifying elkhound as retreiver 50 times more than otherwise
