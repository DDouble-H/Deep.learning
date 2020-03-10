## Layer별 파라미터 정보

```py
input shape = [batch_size, height, width, channel]
```

##### Convolution

```py
tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation='relu')
```

- filters : 필터의 개수를 지정

- kernel_size : convolution filter의 사이즈를 지정

- strides : 필터를 순회하는 간격을 의미

- padding : convolutional layer에서  filter, stride 적용으로 feature map의 크기는 입력데이터보다 작아진다. 이 때 출력데이터가 줄어드는 것을 방지하는 방법, 일반적으로 0으로 채워넣음

- activation : activation function을 지정

- pooling : convolutional의 출력 데이터를 입력으로 받아 출력 데이터의 크기를 줄이거나 특정 데이터를 강조하는 용도로 사용됨

- flatten : convolutional layer에서 추출된 피쳐들을 1차원으로 변경시킴

- dense

  ```py
  tf.keras.layers.Dense(32, activation='relu')
  ```

- dropout

  ```py
  dropout = tf.keras.layers.Dropout(0.5)
  ```

- build model

  ```python
  from tensorflow.keras import layers
  input_shape  = (28,28,1)
  num_classes = 10
  
  inputs = layers.Input(shape=input_shape)
  
  #feature extraction 
  model = layers.Conv2D(32, 3, padding ='same')(inputs)
  model = layers.Activation('relu')(model)
  model = layers.Conv2D(32, 3, padding ='same')(model)
  model = layers.Activation('relu')(model)
  model = layers.MaxPool2D(2,2)(model)
  model = layers.Activation('relu')(model)
  model = layers.Dropout(0.5)(model)
  
  model = layers.Conv2D(64, 3, padding ='same')(model)
  model = layers.Activation('relu')(model)
  model = layers.MaxPool2D(2,2)(model)
  model = layers.Activation('relu')(model)
  model = layers.Conv2D(128, 3, padding ='same')(model)
  model = layers.Activation('relu')(model)
  model = layers.MaxPool2D(2,2)(model)
  model = layers.Activation('relu')(model)
  model = layers.Dropout(0.5)(model)
  
  # fully connected
  model = layers.Flatten()(model)
  model = layers.Dense(512)(model)
  model = layers.Activation('relu')(model)
  model = layers.Dropout(0.5)(model)
  model = layers.Dense(num_classes)(model)
  model = layers.Activation('softmax')(model)
  
  model = tf.keras.Model(inputs=inputs, outputs=model, name='basic_cnn')
  ```


##### Optimization

- Loss function : loss를 통해 정답라벨과 비교해 얼마나 틀렸는지 확인할 수 있으며, 학습을 통해 loss를 최대한 줄이는 것이 목표

  - binary : 분류하고자 하는  class가 2개일 때
  - categorical :  분류하고자 하는 class가 2개 이상일 때

  ```py
  tf.keras.losses.sparse_categorical_crossentropy
  tf.keras.losses.categorical_crossentropy
  tf.keras.losses.binary_crossentropy
  ```

- Optimization : 학습을 통해 얻은 loss값을 최소화하기 위해 최적화된 값들을 반환하고 반환된 값이 적용하며 점차 모델의 성능을 높이는 방식으로 진행되며,  이때 최적화된 값만큼 즉각적으로 변화가 있는 것이 아니라 learning rate만큼 변한 값이 적용 됨

  - sgd, rmsprop, adam

  ```py
  tf.keras.optimizers.SGD()
  tf.keras.optimizers.RMSprop()
  tf.keras.optimizers.Adam()
  ```

- Metrics : 모델을 평가하는 방법

  ```python
  tf.keras.metrics.Accuracy()
  tf.keras.metrics.Precision()
  tf.keras.metrics.Recall()
  ```

##### compile

```py
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss ='sparse_categorical_crossentropy',
              metrics = [tf.keras.metrics.Accuracy()])
```

##### train

- hyperparameter 설정
  - epochs, batch_size

```py
model.fit(x_train, y_train, 
          batch_size=32, 
          shuffle=True, 
          epochs=50) 
```