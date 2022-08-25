from models import (
    cifar10_resnet_74
)
from fvcore.nn import FlopCountAnalysis
import torch

model = cifar10_resnet_74()
print(model)
raise ""

inp = torch.zeros(128, 3, 32, 32).cpu()
flops = FlopCountAnalysis(model.cuda(), inp.cuda()).total()
print(flops * 64000.)
