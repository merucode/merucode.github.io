---
layout: default
title: ANN Code
parent: ANN
grand_parent: Deep Leaning
nav_order: 7
---

# ANN Code
{: .no_toc .d-inline-block }
ing
{: .label .label-green }
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

<!------------------------------------ STEP ------------------------------------>

### Step 1-1. 가중치/편향 초기화

```python
import numpy as np

def initialize_parameters(neurons_per_layer):	# 가중치/편향 초기화
    L = len(neurons_per_layer)- 1   # 층 개수 저장
    parameters = {}

    for l in range(1, L+1):		    # 1층 부터 L층까지 돌면서 가중치와 편향 초기화
        parameters['W' + str(l)] = np.random.randn(neurons_per_layer[l],neurons_per_layer[l-1]) * np.sqrt(1/neurons_per_layer[l])
        parameters['b' + str(l)] = np.random.randn(neurons_per_layer[l]) * np.sqrt(1/neurons_per_layer[l])
        
    return parameters

# 테스트 코드
neurons_per_layer = [10, 5, 5, 3]
initialize_parameters(neurons_per_layer)
```

* 가중치와 편향의 표준오차 조절
	* [image](https://www.codeit.kr/learn/4001)


### Step 1-2. Forward Propagation

```python
import numpy as np
import pandas as pd

dataset = pd.read_csv('./data/MNIST_preprocessed.csv', sep=',', header=None).values


X = dataset[:, 0:784]	# 입력, 목표 변수 데이터 셋 나누기
Y = dataset[:, 784:]
X_train, X_test = X[0:250,], X[250:,] # training, testing 데이터 셋 나누기
Y_train, Y_test = Y[0:250,], Y[250:,]

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def initialize_parameters(nodes_per_layer):
	...
    return parameters

def feed_forward(x, parameters):
    cache = {'a0': x}  			# 0 번째 층 출력 저장
    L = len(parameters) // 2  	# 층 수 저장(W, b 2개로 구성되어서 나누기 2 해줌)
    
    for l in range(1, L+1):
        # 전 층 뉴런의 출력, 현재 층 뉴런들의 가중치, 편향 데이터를 가지고 온다 (여기에 코드를 작성하세요)
        a_prev = cache['a' + str(l-1)]
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        
        # 가지고 온 데이터로 z와 a를 계산한다. (여기에 코드를 작성하세요)
        z = W @ a_prev + b
        a = sigmoid(z)

        # 결과 값을 캐시에 저장한다.
        cache['z' + str(l)] = z
        cache['a' + str(l)] = a
                
    return a, cache

# 테스트 코드
neurons_per_layer = [784, 128, 64, 10]
parameters = initialize_parameters(neurons_per_layer)
feed_forward(X_train[0], parameters)[0]
```

### Step 1-3. 