residual block

resnet의 구조는 resiual block + identity block으로 이루어진다. residual block의 gradient가 흐를 수 있도록 shortcut(skip connection)을 만들어주며, 이를 통해 gradient vanishing 문제를 해결한다. 또한, residual block의 skip connection 덕분에 입력데이터와 그래디언트가 오갈 수 있는  통로가 늘어나기 때문에 앙상블 모델과 유사한 효과를 낼 수 있다.

densenet

densenet은 resnet에서 조금 개선된 모델로 dense connectivity를 제안했다.  dense connectivity는 입력값을 계속해서 출력값의 채널 방향으로 합쳐주는 것이다. densenet은 이미지의 저수준의 특징들이 잘 보존되고, gradient vanishing 문제가 발생하지 않는다. 또한 깊이에 비해 파라미터 수가 적어 연산량이 줄어든다.