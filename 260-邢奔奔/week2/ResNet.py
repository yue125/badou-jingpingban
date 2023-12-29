import torch
import matplotlib.pyplot as plt
import numpy as np


class Resnet(nn.Module):
    def __init__(self,input_size,output_size):
        super(Resnet,self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def Convblock(self,input_size,output_size):
        x = torch.conv2d()