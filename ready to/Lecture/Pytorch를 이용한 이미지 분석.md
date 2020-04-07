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


- torchvision

  ```py
  from PIL import Image
  import numpy as np
  import torchvision
  
  path = 'test.jpg'
  image = Image.open(path)
  
  torchvision.transforms.CenterCrop(size=(300, 300))(image)
  torchvision.transforms.ColorJitter(brightness=1, contrast=0, saturation=0, hue=0)(image)
  torchvision.transforms.FiveCrop(size=(300, 300))(image)[3] # 5개로 크롭
  torchvision.transforms.Grayscale(num_output_channels=1)(image)
  torchvision.transforms.RandomAffine(degrees=90, translate=None, scale=None, shear=None, resample=False, fillcolor=0)(image) # fillcolor=빈 공간 채울 색
  
  
  transforms = [torchvision.transforms.Grayscale(num_output_channels=1),
                torchvision.transforms.CenterCrop(size=(600, 600)),
                torchvision.transforms.RandomAffine(degrees=90, translate=None, scale=None, shear=None, resample=False, fillcolor=0)]
                
  torchvision.transforms.RandomApply(transforms, p=0.5)(image) # p=확률
  torchvision.transforms.RandomChoice(transforms)(image)
  torchvision.transforms.RandomCrop(size=(600, 600), padding=None, pad_if_needed=False, fill=0, padding_mode='constant')(image)
  torchvision.transforms.RandomHorizontalFlip(p=0.7)(image)
  ```

- transform on Tensor

  ```py
  # normalization
  tensor_image = torchvision.transforms.ToTensor()(image) # totensor
  transform_image = torchvision.transforms.Normalize(mean=(0,0,0), std=(1,1,1), inplace=False)(tensor_image) # mean=dimension기준
  
  transform_image = transform_image.numpy()
  
  print('Min:', np.min(transform_image_image), 
        ', Max:', np.max(transform_image_image), 
        ', Mean:', np.mean(transform_image_image), 
        ', Std:', np.std(transform_image_image))
        
  transform_image = torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)(tensor_image)
  ```

- Learning Rate Scheduler

  ```py
  from torch.optim.lr_scheduler import ReduceLROnPlateau
  scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)
  ```

- Training

  ```py
  import torchvision
  from torch.utils.tensorboard import SummaryWriter
  
  writer = SummaryWriter() # log_dir 지정하지 않아도 생성, 지정가능
  
  for epoch in range(1, epochs + 1):
      # Train Mode
      model.train()
  
      for batch_idx, (data, target) in enumerate(train_loader):
          data, target = data.to(device), target.to(device)
  
          optimizer.zero_grad()
          output = model(data)
          loss = F.nll_loss(output, target)  # https://pytorch.org/docs/stable/nn.html#nll-loss
          loss.backward()
          optimizer.step()
  
          if batch_idx % log_interval == 0:
              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))
      
      # Test mode
      model.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data.to(device), target.to(device)
              output = model(data)
              test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
              pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
              correct += pred.eq(target.view_as(pred)).sum().item()
  
      test_loss /= len(test_loader.dataset)
      
      accuracy = 100. * correct / len(test_loader.dataset)
      
      scheduler.step(accuracy, epoch) # Learning rate scheduler
      
      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),
          accuracy))
      
      if epoch == 0:
          grid = torchvision.utils.make_grid(data)
          writer.add_image('images', grid, epoch)
          writer.add_graph(model, data)
      
      writer.add_scalar('Loss/train/', loss, epoch)
      writer.add_scalar('Loww/test/', test_loss, epoch)
      writer.add_scalar('Accuracy/test/', accuracy, epoch)
  writer.close()
  ```

- save model

  ```py
  # weight만 저장 : 다른 모델에 적용, 모델 수정 등 용이
  save_path = './model/model_weight.pt'
  torch.save(model.state_dict(), save_path)
  
  model = Net().to(device) # load a model
  weight_dict = torch.load(save_path)
  
  model.load_state_dict(weight_dict) # load a model, weight
  model.eval()
  
  # 전체모델 저장
  save_path = './model/model.pt'
  torch.save(model, save_path)
  model = torch.load(save_path)
  model.eval()
  
  #
  checkpoint_path = 'checkpoint.pt'
  torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': loss
              }, checkpoint_path)
  model = Net().to(device)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
  
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']
  model.train()
  ```