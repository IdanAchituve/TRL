import torch
import torchvision
import argparse
from torchvision import datasets, transforms
import math
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import copy
import pickle
from torch.utils.data.dataset import Dataset


class my_CNN(nn.Module):

    __slots__ = ('conv1', 'conv1_bn', 'conv1_drop', 'conv2', 'conv2_bn', 'conv2_drop', 'pool',
                 'out1_volume_size', 'out2_volume_size', 'fc1', 'dense1_bn', 'fc2','dense2_bn','fc3')

    def __init__(self, args):
        super(my_CNN, self).__init__()
        # conv layers
        self.conv1 = nn.Conv2d(3, args.K[0], kernel_size=args.F[0], stride= args.S[0], padding=args.P[0])  # 1st convolution
        self.conv1_bn = nn.BatchNorm2d(args.K[0])
        self.conv1_drop = nn.Dropout2d(p = args.dropout_prob_conv)
        self.conv2 = nn.Conv2d(args.K[0], args.K[1], kernel_size=args.F[1], stride= args.S[1], padding=args.P[1])  # 2nd convolution
        self.conv2_bn = nn.BatchNorm2d(args.K[1])
        self.conv2_drop = nn.Dropout2d(p = args.dropout_prob_conv)
        self.pool = nn.MaxPool2d(args.pool, args.pool)  # down-sampling
        # calc output size of conv layers
        self.out1_volume_size = ((32 - args.F[0] + 2*args.P[0])/args.S[0] + 1)/args.pool
        self.out2_volume_size = ((self.out1_volume_size - args.F[1] + 2*args.P[1])/args.S[1] + 1)/args.pool
        # FC layers
        self.fc1 = nn.Linear(args.K[1] * self.out2_volume_size * self.out2_volume_size, args.fc_layers[0])
        self.dense1_bn = nn.BatchNorm1d(args.fc_layers[0])
        self.fc2 = nn.Linear(args.fc_layers[0], args.fc_layers[1])
        self.dense2_bn = nn.BatchNorm1d(args.fc_layers[1])
        self.fc3 = nn.Linear(args.fc_layers[1], 10)
        # xavier initialization
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)


    def forward(self, x, args):
        x = self.pool(F.relu(self.conv1_drop(self.conv1_bn(self.conv1(x)))))
        x = self.pool(F.relu(self.conv2_drop(self.conv2_bn(self.conv2(x)))))
        x = x.view(-1, args.K[1] * self.out2_volume_size * self.out2_volume_size)
        x = F.relu(self.dense1_bn(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=args.dropout_prob_fc)
        x = F.relu(self.dense2_bn(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=args.dropout_prob_fc)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    
def read_data(args):
    
    cifar_mean = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
    cifar_std = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
    transformations = transforms.Compose([
                               transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(cifar_mean, cifar_std)
                           ]) if args.ResNet else transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(cifar_mean, cifar_std)
                                         ])
    # user python data loaders to create iterators over fashion mnist data
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transformations )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=True, transform=transformations),
        batch_size=args.test_batch_size, shuffle=False, num_workers=1)

    # get validation set from train set
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(math.floor(num_train*0.2))

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # define our samplers -- we use a SubsetRandomSampler because it will return
    # a random subset of the split defined by the given indices without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    # Create the train_loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, sampler=train_sampler,
                                               num_workers=1)
    # Create the validation loader
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.test_batch_size, sampler=validation_sampler,
                                                    num_workers=1)

    return train_loader, validation_loader, test_loader


def train(args, model, train_loader, validation_loader, optimizer, device):

    num_epochs = args.epochs
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        cumm_loss = correct = 0.0
        num_examples = 0
        model.train()

        for batch_idx, (inputs, labels) in enumerate(train_loader):

            # move to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # zero the gradients
            output = model(inputs, args) if not args.ResNet else model(inputs)# get prediction
            if args.ResNet:
                output = F.log_softmax(output, dim=1)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            loss = F.nll_loss(output, labels)  # the average batch loss
            loss.backward()
            optimizer.step()

            num_examples += len(labels)
            cumm_loss += loss  # get the average batch loss
            correct += pred.eq(labels.data.view_as(pred)).sum().item()  # sum correct predictions

        cumm_loss /= len(train_loader)  # get average loss over whole dataset
        correct /= num_examples
        _, val_cumm_loss, val_accuracy = test(args, model, validation_loader, device)  # run model on validation set

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

        print('train Epoch: {}\ttrain Loss: {:.6f}\ttrain accuracy: {:.6f}\t'
          'val loss: {:.6f}\tval accuracy: {:.6f}'.format(
        epoch, cumm_loss, correct,val_cumm_loss, val_accuracy))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc


def test(args, model, loader, device, is_val=True):

    cumm_loss = correct = 0.0
    model.eval()
    num_examples = 0

    for batch_idx, (inputs, labels) in enumerate(loader):

        # save true value
        y_true = np.copy(labels.numpy()) if batch_idx == 0 else np.append(y_true, labels.numpy())

        # move to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        output = model(inputs, args) if not args.ResNet else model(inputs)  # get prediction
        if args.ResNet:
            output = F.log_softmax(output, dim=1)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        cumm_loss += F.nll_loss(output, labels, size_average=False).item()  # sum losses
        correct += pred.eq(labels.view_as(pred)).sum().item()
        num_examples += len(labels)

        # save predictions
        y_pred = np.copy(pred.view(1,1,-1).cpu().numpy().flatten()) if batch_idx == 0 else np.append(y_pred, pred.view(1,1,-1).cpu().numpy().flatten())

    cumm_loss /= num_examples  # get average loss over whole dataset
    correct /= num_examples

    if not is_val:
        print('test Loss: {:.6f}\ttest accuracy: {:.6f}\t'.format(
            cumm_loss, correct))
        print(confusion_matrix(y_true, y_pred))

    return y_pred, cumm_loss, correct


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    #mean = np.array([0.485, 0.456, 0.406])
    #std = np.array([0.229, 0.224, 0.225])
    #inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


def run():
    # configurations
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--dropout-prob-conv', type=float, default=0.2, metavar='N',
                        help='dropout probability conv(default: 0.2)')
    parser.add_argument('--dropout-prob-fc', type=float, default=0.4, metavar='N',
                        help='dropout probability (default: 0.4)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--K', type=list, default=[64, 128], metavar='M',
                        help='number of kernels per conv layer (default: [64, 128])')
    parser.add_argument('--F', type=list, default=[3, 3], metavar='M',
                        help='filter size per conv layer (default: [3, 3])')
    parser.add_argument('--S', type=list, default=[1, 1], metavar='M',
                        help='stride size per conv layer (default: [1, 1])')
    parser.add_argument('--P', type=list, default=[1, 1], metavar='M',
                        help='zero padding per conv layer (default: [1, 1])')
    parser.add_argument('--pool', type=int, default=2, metavar='M',
                        help='pooling layer size (default: 2)')
    parser.add_argument('--fc-layers', type=list, default=[150, 100], metavar='M',
                        help='fully connected layer size (default: [150, 100])')
    parser.add_argument('--seed', type=int, default=111, metavar='S',
                        help='random seed (default: 111)')
    parser.add_argument('--ResNet', type=bool, default=False, metavar='S',
                        help='use ResNet-18 model as a feature extractor (default: False)')
    parser.add_argument('--image-show', type=bool, default=False, metavar='S',
                        help='whether to plot the data (default: False)')

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    # read CIFAR-10 to data loader objects
    train_loader, validation_loader, test_loader = read_data(args)

    # obtain class names
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    if args.image_show:
        # Get a batch of training data
        inputs, classes = next(iter(train_loader))
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[class_names[x] for x in classes])


    # initialize network
    if not args.ResNet:
        model = my_CNN(args)
    else:
        model = torchvision.models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        # replace last FC layer(model.fc) with the bellow
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

    # if there is a GPU train on it
    model = model.to(device)

    # set loss function and optimizer
    if args.momentum == 0.0:
        optimizer = optim.Adam(model.parameters(), lr=args.lr) if not args.ResNet else optim.Adam(model.fc.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) if not args.ResNet else optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)

    # train model
    model, dev_acc = train(args, model, train_loader, validation_loader, optimizer, device)

    # test model
    y_pred, cumm_loss, correct = test(args, model, test_loader, device, False)
    np.savetxt("./predictions/test.pred.{:.6f}".format(dev_acc), y_pred, fmt='%d')


if __name__ == '__main__':
    run()