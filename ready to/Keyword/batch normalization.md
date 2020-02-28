배치정규화(Batch Normalization)는 네트워크의 학습 중에 이전 layer의 파라미터 변화로 인해 현재 layer의 입력의 분포가 바뀌는 현상인 Internal Covariate Shift 문제를 해결하기 위해 제안된 방법이다. 배치정규화는 Internal Covariate Shift 문제를 방지하기 위해 각 layer 입력의 분포를 평균이 0, 표준편차가 1이 되도록 정규화를 시킨다. 이 때 입력의 평균을 이동시키고 표준편자를 변화시킬 때 일부 파라미터의 영향이 반영되지 않을 수 있다. 이 문제점을 보완하기 위해서 scale, shift 연산을 추가적으로 진행하며, scale, shift하는 값은 일반 파라미터와 동일하게 네트워크 학습시 학습한다. 정규화를 시킬 때의 평균과 표준편차는 학습에서 사용하는 미니배치의 평균과 표준편차를 이용하며 테스트를 진행할때는 학습했을 때 사용한 미니배치들의 평균과 표준편차의 평균을 이용한다.