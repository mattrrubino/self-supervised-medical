import torchvision.models as models

from cuda import device


encoder_2d = models.densenet121().to(device)

