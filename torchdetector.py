import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class FacialDetection(nn.Module):
	def __init__(self):
		super(FacialDetection, self).__init__()
		self.layer_1 = nn.Sequential(
        	nn.Conv2d(kernel_size=2, in_channels=3, out_channels=256),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU())
		self.layer_2 = nn.Sequential(
            nn.Conv2d(kernel_size=2, in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU())
		self.output = nn.Conv2d(kernel_size=2, in_channels=256, out_channels=128)


	def forward(self, x):
		self.first = self.layer_1(x)
		self.second = self.layer_2(self.first)
		self.third = self.output(self.second)
		return self.third

