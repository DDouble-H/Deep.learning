## pytorch 기초 사용법 

- torch : 텐서 등의 다양한 수학 함수가 포함되어 있으며 numpy와 유사

  ```py
  import numpy as np
  import torch
  
  arr = torch.arange(9)
  arr.numpy()
  arr.reshape(3,3)
  ```

- oprations : pytorch로 수학연산

  ```py
  arr * 3 # broad cast 가능
  arr + arr
  result = arr.add(10)
  ```

- view : 원소의 수를 유지하면서 텐서의 크기를 변경함(numpy의 reshape과 같은 역할)

  ```py
  arr = torch.arange(9).reshape(3,3)
  arr.view(-1)
  arr.view(1, 9)
  ```

- slice and index : numpy와 동일

  ```py
  arr[1]
  arr[1,1]
  arr[1:, 1:]
  ```

- comile : numpy를 torch tensor로 불러오기

  ```py
  arr = np.arry([1,1,1])
  arr_torch = torch.from_numpy(arr)
  ```

- cuda tensors : torch.device를 사용해 tensor를 gpu 안팎으로 이동

  ```py
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  arr_torch.to(device)
  ```

- autograd : 자동 미분을 위한 함수가 포함

  ```py
  x = torch.ones(2, 2, requires_grad=True)
  y = x + 2
  print(y.grad_fn)
  z = y * y * 3
  out = z.mean()
  out.backward()
  print(x.requires_grad)
  with torch.no_grad():
      print((x ** 2).requires_grad)
  ```

torch.cat : 두 텐서를 이어붙임(concatenate), 데이터 복사

torch.unsqueeze : 데이터 참조

## pytorch data preprocess

```py
import torch
from torchvision import datasets, transforms
```

torch.utils.data.DataLoader : 데이터 로드시 DataLoader 클래스 제공

```py
batch_size = 32
test_batch_size = 32

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=(0.5,), std=(0.5,))
                   ])),
    batch_size=batch_size,
    shuffle=True)
    
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('dataset', train=False, 
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5))
                   ])),
    batch_size=test_batch_size,
    shuffle=True)
```

pytorch는 (# batch_size, channel, height, width)