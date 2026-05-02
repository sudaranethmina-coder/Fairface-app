from torchvision import transforms

# CIFAR-10 normalization values
mean = (0.4914, 0.4822, 0.4465)             # RGB
std = (0.2470, 0.2435, 0.2616)

num_classes = 9

train_transform = transforms.Compose([
    transforms.RandomCrop(32,padding=4),    # add 4 pixel padding arround the picture and randomly crop for 32 pixel picture
                                            # this is done to randomly place the image at a random place to help prediction accurate
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),                  # values map to 0,1 from 0,256 to make train more stable
    transforms.Normalize(mean,std)
])

# validation/ test
eval_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),                  
    transforms.Normalize(mean,std)
])

import torch.nn as nn

# CNN model
class CNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # feature extractor using the original simple pattern
        # Conv -> ReLU -> MaxPool
        self.features = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),       # [32,16,16]

            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),       # [64,8,8]

            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)        # [128,4,4]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self,x):
        return self.classifier(
            self.features(x)
        )