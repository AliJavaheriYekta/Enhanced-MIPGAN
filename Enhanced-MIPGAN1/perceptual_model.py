import torch 
from torchvision import models


class VGG16_for_Perceptual(torch.nn.Module):
    def __init__(self,requires_grad=False,n_layers=None,device='cpu'):
        super(VGG16_for_Perceptual,self).__init__()
        vgg_pretrained_features=models.vgg16(pretrained=True).features.to(device)

        self.slices = []
        pre_layer = 0
        for i in n_layers:
            slice = torch.nn.Sequential()
            for j in range(pre_layer,i):
                slice.add_module(str(j),vgg_pretrained_features[j])
            self.slices.append(slice)
            pre_layer = i
        self.results = []
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad=False       
        self.device = device
    def forward(self,x):
        self.results = []
        x = x.to(self.device)
        for slice in self.slices:
            x = slice(x)
            self.results.append(x)
        return self.results      
