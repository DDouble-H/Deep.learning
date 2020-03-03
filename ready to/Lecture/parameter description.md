## TensorFlow 사용법

Tensor 생성

- Tf.constant()
  - List > tensor
  - tuple > tensor
  - array > tensor

```python
import tensorflow as tf
import numpy as np

tf.constant([1,2,3])
tf.constant(((1,2,3),(4,5,6))
arr = np.array([1,2,3])
tensor = tf.constant(arr) # np.array > tf
```

tensor 정보 확인

```py
# shape
tensor.shape
# data type
tensor = tf.constant([1, 2, 3], dtype=tf.float32)
tf.cast(tensor, dtype=tf.uint8)
```

Tensor에서  numpy 사용

```py
# .numpy()
tensor.numpy
# np.arry()
np.array(tensor)
```

 난수 생성

```py
# normal distribution: 연속적인 모양, uniform distribution: 불연속적이며 일정한 분포
np.random.randn(5) # random한 숫자 9개 생성
tf.random.normal([3,3]) # normal
tf.random.uniform([3,3]) # uniform
```

데이터 차원수 늘리기

```py
# numpy
np.expand_dims(train_x, -1)
# tensorflow
tf.expand_dims(train_x, -1)
train_x[..., tf.newaxis].shape
```

