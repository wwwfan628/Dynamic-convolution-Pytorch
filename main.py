import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import argparse
import torchvision.models.vgg
import torchvision.transforms as transforms
from dynamic_models.dy_vgg import vgg11_bn as dy_vgg11
from dynamic_models.dy_vgg_small import dy_vgg_small as dy_vgg_small
from dynamic_models.dy_lenet5 import dy_lenet5
from dynamic_models.raw_vgg import vgg11 as raw_vgg11
from dynamic_models.dy_resnet import resnet18 as dy_resnet18
from torchvision.models.resnet import resnet18 as raw_resnet18


def adjust_lr(optimizer, epoch):
    if epoch in [args.epochs*0.5, args.epochs*0.75, args.epochs*0.85]:
        for p in optimizer.param_groups:
            p['lr'] *= 0.1
            lr = p['lr']
        print('Change lr:'+str(lr))


def train(model, optimizer, trainloader, epoch):
    loss_func = nn.CrossEntropyLoss()
    model.train()
    avg_loss = 0.
    train_acc = 0.
    adjust_lr(optimizer, epoch)
    for batch_idx, (data, target) in enumerate(trainloader):

        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()

        optimizer.step()
    print('Train Epoch: {}, loss{:.6f}, acc{}'.format(epoch, loss.item(), train_acc/len(trainloader.dataset)), end='')
    if args.net_name.startswith('dy'):
        model.update_temperature()


def val(model, testloader, epoch):
    loss_func = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.
    correct=0.
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(args.device), label.to(args.device)
            output = model(data)
            test_loss += loss_func(output, label).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
    test_loss/=len(testloader.dataset)
    correct = int(correct)
    print('Test set:average loss: {:.4f}, accuracy{}'.format(test_loss, 100.*correct/len(testloader.dataset)))
    return correct/len(testloader.dataset)


def main(args):
    if args.dataset == 'mnist':
        numclasses = 10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        trainset = torchvision.datasets.MNIST(root='../datasets', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.MNIST(root='../datasets', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    elif args.dataset == 'cifar10':
        numclasses = 10
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                transform=transforms.Compose([
                                                    transforms.Pad(4),
                                                    transforms.RandomCrop(32),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                         (0.2023, 0.1994, 0.2010))
                                                ]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                        (0.2023, 0.1994, 0.2010))
                                               ]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    elif args.dataset == 'cifar100':
        numclasses = 100
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.Pad(4),
                                                     transforms.RandomCrop(32),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                          (0.2023, 0.1994, 0.2010))
                                                 ]))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                                transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                         (0.2023, 0.1994, 0.2010))
                                                ]))
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.net_name == 'dy_lenet5':
        model = dy_lenet5(numclasses=numclasses)
    elif args.net_name == 'dy_resnet18':
        model = dy_resnet18(num_classes=numclasses)
    elif args.net_name == 'raw_resnet18':
        model = raw_resnet18(num_classes=numclasses)
    elif args.net_name == 'raw_vgg11':
        model = raw_vgg11(num_classes=numclasses)
    elif args.net_name == 'dy_vgg11':
        model = dy_vgg11(num_classes=numclasses)
    elif args.net_name == 'dy_vgg_small':
        model = dy_vgg_small(num_classes=numclasses)
    model.to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print(str(args))

    best_val_acc = 0.
    for i in range(args.epochs):
        train(model, optimizer, trainloader, i + 1)
        temp_acc = val(model, testloader, i + 1)
        if temp_acc > best_val_acc:
            best_val_acc = temp_acc
        print("Epoch {:03d} | Test Acc {:.4f}% ".format(i + 1, temp_acc))
    print('Best acc{}'.format(best_val_acc))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dynamic convolution')
    parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--lr', type=float, default=0.1, )
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--net-name', default='dy_vgg11')

    args = parser.parse_args()
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(args)
    main(args)
    print("Finish!")




