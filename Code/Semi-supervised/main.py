

use_pretrained = True
# Teacher model
model_ft = models.resnet18(pretrained=use_pretrained)
model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)