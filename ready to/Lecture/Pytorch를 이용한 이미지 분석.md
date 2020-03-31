### Pytorch를 이용한 이미지 분석

- Pytorch

  ```py
  import torch
  
  import torch.nn as nn
  import torch.nn.functional as F
  import torch.optim as optim
  
  from torchvision import datasets, transforms
  
  seed = 1
  lr = 0.001
  momentum = 0.5
  batch_size = 64
  test_batch_size = 64
  epochs = 5
  no_cuda = False
  log_interval = 100
  ```

- Model

  ```py
  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
          self.conv1 = nn.Conv2d(3, 20, 5, 1)
          self.conv2 = nn.Conv2d(20, 50, 5, 1)
          self.fc1 = nn.Linear(4*4*50, 500)
          self.fc2 = nn.Linear(500, 10)
  
      def forward(self, x):
          x = F.relu(self.conv1(x))
          x = F.max_pool2d(x, 2, 2)
          x = F.relu(self.conv2(x))
          x = F.max_pool2d(x, 2, 2)
          x = x.view(-1, 4*4*50)
          x = F.relu(self.fc1(x))
          x = self.fc2(x)
          return F.log_softmax(x, dim=1)
  ```

- Preprocess

  ```py
  train_path = './dataset/mnist_png/training/'
  test_path = './dataset/mnist_png/testing/'
  
  torch.manual_seed(seed)
  
  use_cuda = not no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  
  
  train_dataset = datasets.ImageFolder(root=train_dir,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
  
  test_dataset = datasets.ImageFolder(root=test_dir,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
  
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=2,
                                           shuffle=True)
  
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=2
                                           )
  ```

- Optimization

  ```py
  model = Net().to(device)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
  ```