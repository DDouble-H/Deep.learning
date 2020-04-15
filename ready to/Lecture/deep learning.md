###### 경사하강학습법

- 지도 학습(supervised learning)
  
- 입력과 정답을 알려주고 정답을 맞추도록 하는 학습 방법
  
- 비지도 학습(unsupervised learning)
  
  - 정답없이 학습데이터로부터 유용한 정보를 추출하는 학습방법
  
- 학습매개변수(trainable parameters)
  
  - 학습과정에서 값이 변화하는 매개변수, 값의 변화에 따라 알고리즘 출력이 변화
  
- 손실함수(loss function)
  - 알고리즘이 얼마나 잘못하고 있는지를 표현하는 지표로서 손실함수의 값이 낮을수록 학습이 잘 된 것. 정답과 알고리즘 출력을 비교하는데에 사용됨
  - 어떤 손실함수를 사용하는지에 따라서 학습이 어떤 식으로 이루어지는지 결정, 정답의 형태가 결정됨
  - MSE, cross entropy error
  
- - 손실함수를 최소로하는 입력값을 찾아내는 것 (y의 min or max 값을 얻게하는 x값)
  - 분석적 방법(Analytical method) : 함수의 모든 구간을 수식으로 알 때 사용하는 수식적인 해석방법
  - 수치적 방법(Numerical method): 함수의 형태와 수식을 알지 못할 때 사용하는 계산적인 해석방법, gradient descent가 있음
  - 전역 솔루션(global solution) : 정의역(Domain)에서 단 하나가 존재
  - 지역 솔루션(local solution) : 여러개일 수 있으며, 일반적으로 하나의 솔루션을 찾았을 때 local인지 global인지 확신할 수 없음
  - 딥러닝과 최적화 이론 :  딥러닝 네트워크의 학습은 손실함수가 최소가 되게하는 파라미터를 구하는 최적화 문제 

- 경사하강법(Gradient Descent)

  - 경사를 따라 여러번의 단계를 통해 최적점을 찾아내며 경사는 기울기(미분, gradient)를 이용해 계산

  - Learning rate
    - a에 비례해 이동함, a가 너무 작은 경우에 학습 속도가 느림, 적절한 학습률을 선택하는 것이 중요
  - 볼록함수(convex function) 
    - 볼록함수는 어디서 시작하더라도 경사하강법으로 최적값에 도달할 수 있음
  - 비볼록함수(non-convex fuction)
    - 시작위치에 따라 다른 최적값을 찾음 즉 지엽최적값(local minimum)에 빠질 위험이 있으며,  global minimum을 찾지 못할 수 있음
    - 딥러닝 학습의 경우 대부분 비볼록 함수이므로 local minimum에 빠질 위험이 있음
    - 안장점(saddle point) :기울기가 0이되지만 극값이 아닌 지점, 경사하강법은 안장점에서 벗어나지 못함
  - 관성(Momentum)
    - 이동벡터를 이용해 이전 기울기에 영향을 받도록 하는 방법(속도 유지, smooth한 반복)
    - 관성을 이용하면 local minimum과 잡음에 대처할 수 있으며, 이동벡터를 추가로 사용하므로, 경사하강법 대비 2배의 메모리를 사용
  - 적응적 기울기(Adaptive gradient; AdaGrad)
    - 여러개의 변수가 있을 때 변수별로 학습율이 달라지게 조절하는 알고리즘
    - 기울기가 커서 학습이 많이 된 변수는 학습율을 감소시켜, 다른 변수들이 학습이 잘 되도록 함
    - 학습이 오래 진행되면 더이상 학습이 이루어지지 않는 단점 존재
  - RMSProp
    - AdaGrad의 문제점을 개선한 방법으로, 합 대신 지수평균을 이용
    - 변수간의 상대적인 학습율 차이는 유지하면서 gt가 무한정 커지지 않아 학습을 오래 할 수 있음
  - Adam
    - RMSProp과 Momentum의 장점을 결합한 알고리즘으로 딥러닝에서 가장 많이 사용

- 경사하강법을 사용해 얕은 신경망 학습

  ```py
  import tensorflow as tf
  import numpy as np
  
  EPOCHS = 1000
  # 모델 구조
  class MyModel(tf.keras.Model):
      def __init__(self):
          super(MyModel, self).__init__()
          self.d1 = tf.keras.layers.Dense(128, input_dim=2, activation='sigmoid')
          self.d2 = tf.keras.layers.Dense(10, activation='softmax')
              
      def call(self, x, training=None, mask=None):
          x = self.d1(x)
          return self.d2(x)
  # 학습 루프 정의
  @tf.function
  def train_step(model, inputs, labels, loss_object, optimizer, train_loss, train_metric):
      with tf.GradientTape() as tape:
          predictions = model(inputs)
          loss = loss_object(labels, predictions)
      gradients = tape.gradient(loss, model.trainable_variables) # df(x)/dx
      
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      train_loss(loss)
      train_metric(labels, predictions)
  
  np.random.seed(0)
  
  pts = list()
  labels = list()
  
  center_pts = np.random.uniform(-8.0, 8.0, (10,2))
  for label, center_pt in enumerate(center_pts):
      for _ in range(100):
          pts.append(center_pt + np.random.randn(*center_pt.shape))
          labels.append(label)
  pts = np.stack(pts, axis = 0).astype(np.float32) 
  label = np.stack(labels, axis=0)
  
  train_ds = tf.data.Dataset.from_tensor_slices((pts, labels)).shuffle(10000).batch(32)
  # 모델 생성
  model = MyModel()
  
  # 손실함수, 최적화 알고리즘, 평가지표 설정
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
  optimizer = tf.keras.optimizers.Adam()
  train_loss = tf.keras.metrics.Mean(name = 'train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'train_accuracy')
  
  # 학습
  for epoch in range(EPOCHS):
      for x, label in train_ds:
          train_step(model, x, label, loss_object, optimizer, train_loss, train_accuracy)
          
      template = 'Epoch{}, loss{}, Acc{}'
      print(template.format(epoch+1, train_loss.result(), train_accuracy.result()*100))
      
  # 데이터셋, 학습 파라미터 저장
  np.savez_compressed('ch2_datset.npz', inputs=pts, labels=labels)
  w_h, b_h = model.d1.get_weights() # h : hidden layers
  w_out, b_out = model.d2.get_weights()
  w_h = np.transpose(w_h)
  w_out = np.transpose(w_out)
  np.savez_compressed('ch2_parameters.npz',
                     w_h = w_h,
                     b_h = b_h,
                     w_out = w_out,
                     b_out = b_out)
  ```

네트워크 구조

- 뉴런(Neuron)
  - 신경망은 뉴런을 기본 단위로 하며, 이를 조합해 복잡한 구조를 이룸
    - 입력 > 가중치와 편향 > 활성함수 > 출력
    - 두 벡터의 내적으로 표현
    - Fully connecte layer : 행렬곱 연산으로 표현
- 얕은 신경망(Shallow Neural Network)
  - 가장 단순하고 얕은(은닉 계층 1) 신경망 구조를 일컬음
- 심층 신경망(Deep Neural Network)
  - 은닉 계층이 많은 신경망으로 일반적으로 5개 이상의 계층이 있는 경우
  - 은닉 계층 추가 = 특징의 비선형 변환 추가
  - 학습 매개변수의 수가 계층 크기의 제곱에 비례함
  - Sigmoid 활성 함수의 동작이 원활하지 않아 ReLU 도입 필요	
  -  FC layer를 연쇄적으로 적용하면 심층 신경망의 수학적 표현이 됨
  - 심층신경망의 연쇄법칙 : 미분하고자하는 경로 사이에 있는 모든 미분값을 곱하면 원하는 미분을 구할 수 있음 (원하는 미분을 구하고자 하면 loss func의 미분을 알아야 함)

###### 역전파 학습법

- 동적계획법(dynamic programming)
  - 첫 계산시 값을 저장하므로 중복 계산 발생하지 않음
- 출력계층의 미분
  - 연쇄 법칙(chain rule)을 이용하려면 손실함수의 미분이 필요
- 은닉계층의 미분
  - 연쇄법칙을 이용하려면 손실함수, 출력계층 사이의 모든 은닉 계층의 미분이 필요
- 순방향 추론(forward inference)
  - 현재 매개변수에서의 손실값을 계산하기 위해 순차적인 연산을 수행하는 것
  - 학습을 마친 후 알고리즘을 사용할 때에는 순방향 추론을 사용함
- 역전파 학습법(back-propagation)
  - 심층신경망의 미분을 계산하기 위해, 연쇄법칙과 동적계획법을 이용해 효율적으로 계산
  - 순방향 추론과 반대로 이루어져 역전파 학습법이라고 함

