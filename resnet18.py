import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
class VeRiDataset(Dataset):
    def __init__(self, directory, list_IDs, labels, transform=None):
        self.list_IDs = list_IDs
        self.labels = labels
        self.transform = transform
        self.dir = directory
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        img_path = join(self.dir, self.list_IDs[idx])
        
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        identity = int(self.labels[idx])
        #return the image and its identity
        return image, identity

def load_data(root_dir, file):
    list_IDs = []
    labels = []
    with open(join(root_dir, file)) as file_IDs:
        for ID in file_IDs:
            list_IDs.append(ID[:-1])
            labels.append(ID[:4])
    return list_IDs, labels

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
def prepare_data():
    
    root_dir = "/home/namnd/VeRi"
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
    
    #Dataset
    training_list_IDs, training_labels = load_data(root_dir, "name_train.txt")
    testing_list_IDs, testing_labels = load_data(root_dir, "name_test.txt")
    query_list_IDs, query_labels = load_data(root_dir, "name_query.txt")

    partition = {"training": training_list_IDs, "testing": testing_list_IDs, "query": query_list_IDs}# ID
    labels = {"training": training_labels, "testing": testing_labels, "query": query_labels}#Labels

    #Parameters
    params = {"batch_size": 128, "shuffle": True, "num_workers": 0}

    #Generators
    training_set = VeRiDataset(join(root_dir, "image_train"), partition["training"], labels["training"], transform=transform)
    training_generator = DataLoader(training_set, **params)

    testing_set = VeRiDataset(join(root_dir, "image_test"), partition["testing"], labels["testing"], transform=transform)
    testing_generator = DataLoader(testing_set, **params)

    query_set = VeRiDataset(join(root_dir, "image_query"), partition["query"], labels["query"], transform=transform)
    query_generator = DataLoader(query_set, **params)

    return training_generator, testing_generator, query_generator

if __name__ == '__main__':
    model = models.resnet18(pretrained=True)
    #print(model._forward_hooks.keys())
    layer = model._modules['fc']
    print(layer)
    for param in layer.parameters():
        print(param.size())
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # training_generator, testing_generator, query_generator = prepare_data()

    # num_classes = 776
    # max_epochs = 100
    
    # model = models.resnet18(pretrained=True)
    
    # in_features = model.fc.in_features
    # model.fc = nn.Linear(in_features, num_classes)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # model.to(device)
    # #Loop over epochs
    # for epoch in range(1):
    #     running_loss = 0.0
    #     #Training
    #     for i, data in enumerate(training_generator, 0):
    #         inputs, labels = data
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         output = model(inputs)
    #         loss = criterion(output, labels)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item()
    #         print(i, running_loss)
    #         running_loss = 0.0
    # print("Finished Training")

    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for i, data in enumerate(query_generator, 0):
    #         print(i)
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()

    # print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
    