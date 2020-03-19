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