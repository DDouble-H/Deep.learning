### tensor flow 2.0을 이용한 이미지 분석



```py
import os
from glob import glob

os.listdir('파일경로') # 해당 경로에 있는 파일 목록 가져오기
os.getcwd() # 현재 경로 확인
glob('os.listdir('./dataset/img/*.png) # 해당경로에서 확장자가 .png인 파일 목록 가져오기
```

- pillow 통해 이미지 데이터 확인

  ```py
  from PIL import Image
  import matplotlib.pyplot as plt
  
  data_path = glob('os.listdir('./dataset/img/*.png)
  img_pil = Image.open(path)
  img = np.array(img_pil) # img.shape = (28, 28)
  
  plt.imshow(img, 'gray')
  plt.show()
  ```

- tensorflow 통해 이미지 데이터 확인

  ```py
  import tensorflow as tf
  
  file = tf.io.read_file(path) # file open
  image = tf.io.decode_image(file) # image 가져오기
  image.shape # tensor로 넣기 위해서는 channel 필수적, 3차원
  # TensorShape([28, 28, 1])
  plt.imshow(image[:,:,0], 'gray')
  plt.show()
  ```

- 데이터 이미지 사이즈 알기

  ```py
  from tqdm import tqdm_notebook # 진행상황 progress bar로 보여주는 라이브러리
  # 이미지 데이터의 사이즈가 다양, input shape 맞추기 위해 resize 함
  # 무조건 resize를 하면 큰 이미지 > 작게 : 데이터의 정보 손실 또는 작은 이미지 > 크게 : 이미지 키우면 학습에 방해되기도 함
  
  for path in tqdm_notebook(data_path):
      img_pil = Image.open(path)
      image = np.array(img_pil)
      h, w = image.shape
      
      heights.append(h)
      widths.append(w)
  ```

- set data generator

  - Transformation
    - width_shift_range 
    - height_shift_range 
    - brightness_range 
    - zoom_range 
    - horizontal_flip 
    - vertical_flip 
    - rescale 
    - preprocessing_function

  ```py
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  
  # load image
  file = tf.io.read_file(path)
  image = tf.io.decode_image(file)
  
  # set data generator
  datagen = ImageDataGenerator(
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True)
  ```

