pooling layer는 convolutional의 출력 데이터를 입력으로 받아 출력 데이터의 크기를 줄이거나 특정 데이터를 강조하는 용도로 사용되며 max pooling, average pooling 등이 있다. maxpooling layer는 activation map을 통해 추출된 feature map 중 지정한 영역의 크기의 maxpooling map 안에서 가장 큰 값을 추출하는 방법이다. maxpooling을 통해서 전체 데이터 사이즈가 줄어들기 때문에 연산이 줄어들며, 데이터의 크기가 줄어들면서 소실이 발생하므로, 오버피팅을 방지할 수 있다. average pooling은 지정한 영역 중 평균값을 계산하는 방법이다. 이미지 인식 분야에서는 주로 max pooling을 사용한다.

