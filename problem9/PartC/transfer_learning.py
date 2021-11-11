#imports
import os
import numpy as np 
import torch

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

#check if gpu is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('Cuda is not available. Training on CPU ')
else:
    print('Cuda is available, Training on GPU ')


#define directories
data_dir = 'flower_photos/'
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')

dirs = [data_dir, train_dir, test_dir]

for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)



#Match the classes to the folders in each directory with the flower names
classes = ['daisy', 'dandelion', 'roses', 'sunflower', 'tulips']

#load and transform data
#VGG-16 Takes 224x224 images as input, so we resize all of them
data_transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])

train_data = datasets.ImageFolder(train_dir, transform=data_transform)
test_data = datasets.ImageFolder(test_dir, transform=data_transform)

#print out some data stats
print("Num training images: ", len(train_data))
print("Num test images: ", len(test_data))

#Define the data loaders

batch_size = 20
num_workers = 0

#prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

#Investigate the data
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() #for displaying

#plot the batch of images as well as the labels
fig = plt.figure(figsize=(25, 4))
#for idx in np.arange(20):
#    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
#    plt.imsave("./sample_batch.png", np.transpose(images[idx], (1, 2, 0)))
#    ax.set_title(classes[labels[idx]])

for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])
plt.savefig("sample_batch")
    

#load pretrained models
vgg16 = models.vgg16(pretrained=True)

#print the model
print(vgg16)

#print the in and out values of the last layer)
print(vgg16.classifier[6].in_features)
print(vgg16.classifier[6].out_features)

#freeze the weights 
for param in vgg16.features.parameters():
    param.require_grad = False

#number of inputs to last layer
n_inputs = vgg16.classifier[6].in_features

#add last linear layer (this will be a classifer for the five flower classes)
last_layer = torch.nn.Linear(n_inputs, len(classes))

vgg16.classifier[6] = last_layer

#train on gpu if possible
if train_on_gpu:
    vgg16.cuda()

# check that last layer matches expected output
print(vgg16.classifier[6].out_features)

#loss and optimizer for the classifier part of the model
import torch.optim as optim

#specify loss function (categorical cross-entropy)
criterion = torch.nn.CrossEntropyLoss()

#specify optimizer (stochastic gradient descent) and learning rate = .001
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001)

#train
n_epochs = 2
for epoch in range (1, n_epochs+1):
    #keep track of training and validation loss
    train_loss = 0.0

    ############
    #train model
    ############
    for batch_i, (data, target) in enumerate(train_loader):
        #move to gpu if available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        #clear the gradient of all optimized variables
        optimizer.zero_grad()
        #forward_pass: compute predicted outputs by passing inputs to the model
        output = vgg16(data)
        #calculate the batch loss
        loss = criterion(output, target)
        #backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        #perform a single optimizer step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()

        # print mini-batch
        if batch_i % 20 == 19:
            print("Epoch %d, Batch %d loss: %.16f" %
                  (epoch, batch_i + 1, train_loss / 20))
            train_loss = 0.0

# track test loss
# over 5 flower classes
test_loss = 0.0
class_correct = list(0. for i in range(5))
class_total = list(0. for i in range(5))

#iterate over test data
for data, target in test_loader:

    if train_on_gpu:
        data, target = data.cuda(), target.cuda()

    #foward pass: compute predicted outputs by passing inputs to the model
    output = vgg16(data)

    #calculate the batch loss
    loss = criterion(output, target)

    #update test loss
    test_loss += loss.item()*data.size(0)

    #convert output probabilities to predicted class
    _, pred = torch.max(output, 1)

    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

#actually report accuracy
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))


for i in range(5):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100.0 * np.sum(class_correct) / np.sum(class_total), 
    np.sum(class_correct), np.sum(class_total)))



# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()

# get sample outputs
output = vgg16(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
images = images.cpu().numpy()

for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
plt.savefig("prdictions")
