딥러닝의 전체 구조를 간단하게 정리.

Data : 모델을 학습시키는데 있어서 데이터가 필요하며 데이터가 모델의 입력값이 된다.

학습진행 전 데이터에 따라 다양한 전처리 방식이 요구되며 데이터는 정답라벨이 있다.

Model : CNN, ResNet 등 다양하게 설계된 모델이 있으며 Convolutional layer, pooling 등 다양한 layer 층들로 구성된다.

Prediction/logit : 각 클래스 별로 예측한 값을 나타내며 가장 높은 값이 모델에서 예측하는 클래스를 나타낸다.

Loss/cost : loss를 통해 정답라벨과 비교해 얼마나 틀렸는지 확인할 수 있다. 학습을 통해 loss를 최대한 줄이는 것이 목표

Optimization : 학습을 통해 얻은 loss값을 최소화하기 위해 최적화된 값들을 반환하고 반환된 값이 적용하며 점차 모델의 성능을 높이는 방식으로 진행된다.  이때 최적화된 값만큼 즉각적으로 변화가 있는 것이 아니라 learning rate만큼 변한 값이 적용이 된다.

Result : 반복적으로 학습을 진행한 후, 평가 시 예측된 값에서 argmax를 통해 가장 높은 값을 예측한 클래스라고 판단한다.