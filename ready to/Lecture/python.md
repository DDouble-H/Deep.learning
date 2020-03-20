python 문자열 관련 함수

- os

  ```py
  import os
  
  os.listdir(path) # 파일 리스트
  os.path.basename(path) # 파일 이름
  os.path.dirname(path) # 파일 경로
  os.path.exists(path) # 파일 존재 여부
  os.mkdir(path) # 폴더 생성
  ```

- glob

  ```py
  from glob import glob
  test_path = glob('../dataset/cifar/test/*.png')
  ```

- replace

  ```py
  path.replace('test', '/')
  ```

- split

  ```py
  path.split('/')
  ```

- join

  ```py
  os.path.join('data', 'cifar','train)
  ```

- strip

  ```py
  '    test'.strip()
  ```

Dataframe 생성

- pandas

  ```py
  import pandas as pd
  
  data = {'a':[1, 2, 3], 'b':[10, 20, 30], 'c':[100, 200, 300]}
  
  df = pd.DataFrame(data_ex)
  df.head()
  
  df.to_csv(path, index=False)
  
  _df = pd.read_csv('.csv') # file path 
  ```

