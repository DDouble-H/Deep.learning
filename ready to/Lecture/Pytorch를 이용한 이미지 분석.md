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
  
  
  train_dataset = datasets.ImageFolder(root=train_path,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
  
  test_dataset = datasets.ImageFolder(root=test_path,
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

- Custom dataset

  ```py
  class FaceLandmarksDataset(Dataset):
  
      def __init__(self, data_path, transform=None):
          self.data_path = data_path
          self.transform = transform
  
      def __len__(self):
          return len(self.data_path)
  
      def __getitem__(self, idx):
          path = self.data_path[idx]
          # read image
          image = Image.open(path).convert("L")
          # get label
          label = int(path.split('\\')[-2])
          
          if self.transform:
              sample = self.transform(sample)
  
          return image, label
  ```

  ```py
  torch.manual_seed(seed)
  
  use_cuda = not no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  
  train_loader = torch.utils.data.DataLoader(
      Dataset(train_path, 
              transforms.Compose([
                  transforms.RandomHorizontalFlip(), 
                  transforms.ToTensor(), 
                  transforms.Normalize(
                      mean=[0.406],
                      std=[0.225])])
             ),
      batch_size=batch_size, 
      shuffle=True, 
      **kwargs
  )
  
  test_loader = torch.utils.data.DataLoader(
      Dataset(test_path, 
              transforms.Compose([ 
                  transforms.ToTensor(), 
                  transforms.Normalize(
                       mean=[0.406],
                      std=[0.225])])
             ),
      batch_size=batch_size, 
      shuffle=False, 
      **kwargs
  )
  ```

  

