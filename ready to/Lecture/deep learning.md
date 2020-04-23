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
  - 학습 데이터로 정방향 연산을 통해 loss를 구하고 이 때 계층별로 역전파에 필요한 중간결과를 저장함 >  loss를 각 파라미터로 미분, 역방향 연산 이용, 중간결과를 저장해 메모리를 추가로 사용함
- 합성곱 연산 (Convolution)
  - 두 함수를 합성하는 합성곱 연산, 한 함수를 뒤집고 이동하면서 두 함수의 곱을 적분하여 계산
  - 영상(2d)의 합성곱 계산
    - 2D 디지털 신호의 합성곱은 필터를 한 칸씩 옮기면서 영상과 겹치는 부분을 모두 곱해 합치면 된다. 
  - 2d gaussian filter를 적용하면 영상이 흐려지지만 잡음 제거 효과가 있음
  -  미분필터
    - sobel filter를 적용하면 특정 방향으로 미분한 영상을 얻을 수 있으며 해당 방향의 edge 성분을 추출하는 특성이 있음
- 합성곱 계층(Convolutional layer)
  - 합성곱으로 이루어진 뉴런을 전결합 형태로 연결한 것이 합성곱 계층
  - 여러 채널에서 특별한 특징이 나타나는 위치를 찾아냄
  - 합성곱 계층에 의해서 추출된 결과는 공간적 특징이 있으며 이것이 특징맵(Feature map)
  - C(in) X C(out) 번의 합성곱 연산으로 이루어짐
- 합성곱 신경망
  - 입력 : 이미지,
  - 풀링 계층(pooling layer)
    - 이미지의 크기를 줄여줌
    - 여러 화소를 종합해 하나의 화소로 변환하는 계층으로 풀링 계층을 통과하면 이미지의 크기가 줄어들고, 정보가 종합됨
    - max pooling, average pooling이 있음
  - Flatten
    - 입력된 특징 맵의 모든 화소를 나열해 하나의 벡터로 만드는 것
    - 합성공 계층과 전결합 계층을 연결
  - receptive field
    - 같은 크기의 필터여도, 풀링에 의해 작아진 특징 맵에 적용되면 원본 이미지에서 차지하는 범위가 넓음
  - stride
    - 커널을 이동시키는 거리를 말하며, stride를 크게하면 출력의 크기가 줄어듦

- 일반 경사 하강법의 경우, gradient를 한 번 업데이트 하기 위해 모든 학습 데이터를 사용한다
- 확률적 경사 하강법은 (SGD:Stochastic gradient descent)
  - 데이터를 선택, batch size를 쪼개서 네트워크 모델 학습, 손실함수를 구해 batch size개의 gradient 평균을 사용함, 한 번 업데이트 하기 위해 일부의 데이터만을 사용
- 미니 배치 학습법
  - 학습 데이터 전체를 한 번 학습하는 것을 Epoch, 한 번의 gradient를 구하는 단위를 batch라고 함
- internal covariate shift
  - 학습과정에서 계층별로 입력의 데이터 분포가 달라지는 현상, 이를 해결할 수 있는 방법은 배치 정규화
- 배치 정규화
  - 학습과정에서 배치별로 평균과 분산을 이용해 정규화하는 계층을 배치 정규화 계층이라고 함
  - 학습단계에서 정규화로 인해, 모든 계층의 특징이 동일한 scale이 되어 학습률 결저에 유리하며 추가적인 scale, bias를 학습해 activation에 적합한 분포로 변환할 수 있음
  - 추론단계에서는 평균과 분산을 이동평균(지수평균)을 구하여 고정하며 정규화와 추가 scale, bias를 결합하여 단일 곱, 더하기 연산으로 줄일 수 있음
- GoogLeNet(Inception)
  - inception module
    - 다양한 크기의 합성곱 계층을 한 번에 계산하는 모듈
    - 연산량을 줄이기 위한 1x1 합성곱 계층을 적용, bottleneck 구조
    - Bottleneck구조 활용으로 receptive filed를 유지하면서 파라미터의 수와 연산량 감소하는 효과
    - 역전파에서 vanishing gradient 문제 발생을 방지하기 위해 추가 분류기 적용
- Residual Network (ResNet)
  - Skip-connection
    - feature를 추출하기 전 후를 더하는 특징이 있음
    - 기존에는 한 단위의 특징 맵을 추출하고 난 후에 활성함수를 적용했으나 개선된 구조에서는 identity mapping을 얻기 위해서 pre-activation을 제안
  - Pre-activation
    - Conv-BN-ReLU 구조를 BN-ReLU-Conv 구조로 변경 후 성능 개선
    - BN-ReLU-Conv의 경우 gradient highway가 형성되어 극적인 효과가 나타남
- Densely Connected ConvNets(DenseNet)
  - ResNet의 아이디어를 개선한 구조
  - Dense Block을 제안하며, Dense Block은 ResNet과 같이 Pre-Activaton 구조를 사용
  - Dense Block
    - 연결된 skip connection이 복잡해 보일 수 있으나 이전 특징 맵에 누적해 concatenate하는 결과와 같음
    - 연산량이 증가되는 것을 방지하기 위해 1x1 Conv를 이용한 Bottleneck 구조를 사용함