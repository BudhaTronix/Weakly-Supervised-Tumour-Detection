import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def defineModel():
    # Define Model
    model = models.resnet101(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_classes = 10
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def optimizer(model):
    params_to_update = model.parameters()
    optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
    return optimizer


def criterion():
    return nn.MSELoss()

def tempCal(X_train,X_test):
    model = defineModel()
    teacher_train_logits = model(X_train)
    teacher_test_logits = model(X_test)
        # This model directly gives the logits ( see the teacher_WO_softmax model above)

    # Perform a manual softmax at raised temperature
    train_logits_T = teacher_train_logits / temp
    test_logits_T = teacher_test_logits / temp

    Y_train_soft = nn.functional.softmax(train_logits_T)
    Y_test_soft = nn.functional.softmax(test_logits_T)

    # Concatenate so that this becomes a 10 + 10 dimensional vector
    Y_train_new = np.concatenate([Y_train, Y_train_soft], axis=1)
    Y_test_new = np.concatenate([Y_test, Y_test_soft], axis=1)

a = defineModel()
print(a)
c = torch.rand(1,1,100,100)