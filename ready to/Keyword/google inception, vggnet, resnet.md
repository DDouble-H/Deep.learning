google inception, vggnet, resnet

cnn은 네트워크가 깊어질수록 학습해야하는 파라미터의 수가 늘어날수록 연산량이 커진다. 이를 해결할 수 있는 방법이 inception이다. inception모듈에서는 이전의 입력을 1x1 conv layer를 거쳐  3x3, 5x5의 conv layer를 거치며 확장하는 방식으로 진행해 연산량을 줄이게 된다. 또한, softmax를 통해 값을 뽑아내는 부분이 중간 중간에 위치하고 있어 gradient를 적절하게 역전파된다. inception 모델은 각 필터의 결과를 합쳐 표현하는 것으로 네트워크가 깊은 층을 가지지만, 1x1 conv layer를 사용해 연산량을 줄인다는 것이 특장점이다.

vggnet은 네트워크의 깊이가 어떤 영향을 주는지 연구하고자 설계된 네트워크로 convolutional kernel을 한 사이즈로 정하고 실험한다. 간단한 구조와 단일 네트워크에서 좋은 성능을 보여 많은 실험에 이용되고 있지만, 파라미터의 수가 너무 많다는 단점이 있다.

resnet은 20-layer, 56-layer를 가지고 실험을 진행했으며, 더 많은 레이어를 사용한 56-layer 모델의 결과가 더 나쁘게 나오는 것을 확인했다. 이는 네트워크가 깊어질수록 발생하는 문제로 네트워크가 깊어질수록 gradient vanishing/exploding 혹은 degradation이 발생한다. 이를 해결하기 위해 resnet은 layer의 입력을 출력에 바로 연결시키는 skip connection을 사용했다. 기존의 학습 방식이 weighte layer를 거쳐 학습을 통해 최적의 출력을 내는 것에 반해 resnet은 출력과 입력의 차를 얻을 수 있도록 학습을 한다. 이 때 skip connection은 파라미터 없이 바로 연결되는 구조이기 때문에 연산량이 늘어나지는 않는다. resnet 모델은 깊은 네트워크도 최적화가 가능하며, 깊은 네트워크로 인해 정확도를 개선할 수 있다는 장점이 있다.

