import pickle
import os
from sklearn.model_selection import train_test_split
import numpy as np
import torch

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#read data from pkl file
def load_full_dataset(data_path='drive/MyDrive/data'):

    with open(os.path.join(data_path, 'motorway_dataset.pkl'), 'rb') as f:
        save = pickle.load(f,encoding='bytes')
        motorway_dataset = save[b'dataset']
        motorway_labels = save[b'labels']
        del save

    with open(os.path.join(data_path,'secondary_dataset.pkl'), 'rb') as f:
        save = pickle.load(f,encoding='bytes')
        secondary_dataset = save[b'dataset']
        secondary_labels = save[b'labels']
        del save

    dataset = np.concatenate((motorway_dataset,secondary_dataset), axis=0)
    labels = np.concatenate((motorway_labels,secondary_labels), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test

#test function
def test(model,testloader,lossfn,onehotencoding=True,n_class=3):
  predlist=torch.zeros(0,dtype=torch.long, device='cpu')
  lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
  running_loss = 0.0
  for i, data in enumerate(testloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data 
    #inputs = inputs.view(-1,1,64,9)  
    outputs = model(inputs)
    if onehotencoding == True:
      new_labels = torch.nn.functional.one_hot(labels.to(torch.int64),n_class).to(torch.float32)
    else:
      new_labels = labels
    loss = lossfn(outputs, new_labels)
    # print statistics
    running_loss += loss.item()
    predlist=torch.cat([predlist,torch.argmax(outputs, dim=1).view(-1).cpu()])
    lbllist=torch.cat([lbllist,labels.view(-1).cpu()])
  print()
  print("The test is: ",loss)
  conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
  print("The test confusion matrix is: ",conf_mat)
  class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
  print("the test accuracy acheived is: ")
  print(np.sum(class_accuracy)/n_class)
  print("classification report:")
  print(classification_report(lbllist.numpy(), predlist.numpy()))
  print()

#model training function
def train(model,n_epochs,trainloader,optimizer,lossfn,test_dataloader,onehotencoding=True,n_class=3):
  predlist=torch.zeros(0,dtype=torch.long, device='cpu')
  lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
  for epoch in range(n_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data   
      # zero the parameter gradients
      optimizer.zero_grad()
      # forward + backward + optimize
      #inputs = inputs.view(-1,1,64,9)
      outputs = model(inputs)
      #print(outputs.shape)
      if onehotencoding == True:
        #new_labels = torch.tensor(onehotvector[labels])
        #print(labels.shape)
        new_labels = torch.nn.functional.one_hot(labels.to(torch.int64),n_class).to(torch.float32)
      else:
        new_labels = labels
      loss = lossfn(outputs, new_labels)

      loss.backward()
      optimizer.step()
      # print statistics
      running_loss += loss.item()
      predlist=torch.cat([predlist,torch.argmax(outputs, dim=1).view(-1).cpu()])
      lbllist=torch.cat([lbllist,labels.view(-1).cpu()])
    print()
    print("The training loss for epoch ",epoch+1," is: ",loss)
    # Confusion matrix
    
    print("the training accuracy acheived is: ",accuracy_score(lbllist,predlist))
    test(model,test_dataloader,lossfn)
