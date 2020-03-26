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

- data preprocess

  ```py
  train_path = glob(./dataset/cifar/train/*.png)
  test_path = glob(./dataset/cifar/test/*.png)
  
  path = train_path[0]
  
  def get_class_name(path):
      return path.split('_')[-1].replace('.png', '')
      
  train_label = [get_class_name(path) for path in train_path]
  
  class_name = np.unique(train_label)
  
  def get_label(path):
      fname = tf.strings.split(path, '_')[-1]
      label_name = tf.strings.regex_replace(fname, '.png', '')
      onehot = tf.cast(label_name == class_names, tf.uint8)
      return onehot
      
  def load_image_label(path):
  	  # read image
      gfile = tf.io.read_file(path)
      image = tf.io.decode_image(gfile)
      image = tf.cast(image, tf.float32) / 255.  # rescale
      
      # read label
      label = get_label(path)
      return image, label
  
  image, label = load_image_label(path)
  
  def image_preprocess(image, label):
      image = tf.image.random_flip_up_down(image)
      image = tf.image.random_flip_left_right(image)
      return image, label
      
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  
  # train_dataset
  train_dataset = tf.data.Dataset.from_tensor_slices(train_path)
  train_dataset = train_dataset.map(load_image_label, num_parallel_calls=AUTOTUNE)
  train_dataset = train_dataset.map(image_preprocess, num_parallel_calls=AUTOTUNE)
  train_dataset = train_dataset.batch(batch_size)
  train_dataset = train_dataset.shuffle(buffer_size=len(train_path))
  train_dataset = train_dataset.repeat()
  
  # test_dataset
  test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
  test_dataset = test_dataset.map(load_image_label, num_parallel_calls=AUTOTUNE)
  test_dataset = test_dataset.batch(batch_size)
  test_dataset = test_dataset.repeat()
  ```

- training

  ```py
  steps_per_epoch = len(train_path) // batch_size
  validation_steps = len(test_path) // batch_size
  
  model.fit_generator(
      train_dataset,
      steps_per_epoch=steps_per_epoch,
      validation_data=test_dataset,
      validation_steps=validation_steps,
      epochs=num_epochs
  )
  ```

- Callbacks

  ```py
  logdir = os.path.join('logs',  datetime.now().strftime("%Y%m%d-%H%M%S"))
  
  tensorboard = tf.keras.callbacks.TensorBoard(
      log_dir=logdir, 
      write_graph=True, 
      write_images=True,
      histogram_freq=1
  )
  ```

- Checkpoint

  ```py
  checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') 
  # save_best_only acc가 올라가면 저장, 아니면 저장하지 않음
  # monitor='loss'> mode ='min'
  
  ```

- Learning Rate Scheduler

  ```py
  def scheduler(epoch): # 에폭마다 러닝레이트를 변경함
      if epoch < 10 :
          return 0.001
      else:
          return 0.001 * math.exp(0.1 * (10-epoch))
          
  learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
  ```

  

