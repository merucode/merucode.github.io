---
layout: default
title: ANN Code
parent: Supervised Learning
grand_parent: Machine Leaning
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

### Step 1-3. Back propagation

```python
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

dataset = pd.read_csv('./data/MNIST_preprocessed.csv', sep=',', header=None).values

X = dataset[:, 0:784]	# 입력, 목표 변수 데이터 셋 나누기
Y = dataset[:, 784:]
X_train, X_test = X[0:250,], X[250:,]	# training, testing 데이터 셋 나누기
Y_train, Y_test = Y[0:250,], Y[250:,]

def sigmoid(x):
    """시그모이드 함수"""
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    """시그모이드 미분 함수"""
    return (np.exp(-x))/((np.exp(-x)+1)**2)

def initialize_parameters(neurons_per_layer):
    """신경망의 가중치와 편향을 초기화해주는 함수"""
    L = len(neurons_per_layer) - 1  # 입력층을 포함함 층 개수 저장
    parameters = {}
    
    # 1층 부터 L 층까지 돌면서 가중치와 편향 초기화
    for l in range(1, L+1):
        parameters['W' + str(l)] = np.random.randn(neurons_per_layer[l], neurons_per_layer[l-1]) * np.sqrt(1. / neurons_per_layer[l])
        parameters['b' + str(l)] = np.random.randn(neurons_per_layer[l]) * np.sqrt(1. / neurons_per_layer[l])
        
    return parameters

def feed_forward(x, parameters):
    """순전파 함수"""
    cache = {'a0': x}
    L = len(parameters) // 2
    
    for l in range(1, L+1):
        # 전 층 뉴런의 출력, 현재 층 뉴런들의 가중치, 편향 데이터를 가지고 온다
        a_prev = cache['a' + str(l-1)]
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        
        # 데이터로 z와 a를 계산한다
        z = W @ a_prev + b
        a = sigmoid(z)

        # 결과 값을 캐쉬에 저장한다
        cache['z' + str(l)] = z
        cache['a' + str(l)] = a
                
    return a, cache

def compute_accuracy(x_val, y_val, parameters):
    """테스트 데이터에서 예측값들의 성능을 계산하는 함수"""
    predictions = []

    for x, y in zip(x_val, y_val):
        output, _ = feed_forward(x, parameters)
        pred = np.argmax(output)
        predictions.append(pred == np.argmax(y))

    return np.mean(predictions)

def compute_loss(x_val, y_val, parameters):
    """학습 데이터에서 현재 모델의 손실을 계산하는 함수"""
    loss = 0
    
    for x, y in zip(x_val, y_val):
        output, _ = feed_forward(x, parameters)
        loss += np.mean((output - y)**2) / 2
        
    return loss / len(x_val)

def back_prop(prediction, y, cache, parameters):
    """역전파 함수"""
    gradients = {}
    L = len(cache) // 2
    da = (prediction - y) / y.shape[0]
    
    for layer in range(L, 0, -1):
        # 역전파 행렬 연산을 사용해서 각 요소에 대한 편미분 계산
        # 여기에 코드를 작성하세요
        db = d_sigmoid(cache['z' + str(layer)]) * da    # a와 b 배열 요소별 곱하기
        dW = np.outer(db, cache['a' + str(layer-1)])  # a와 b 배열 외적곱
        da = parameters['W' + str(layer)].T @ db
        # 계산한 편미분 값들을 저장
        gradients['dW' + str(layer)] = dW
        gradients['db' + str(layer)] = db
    
    # 계산한 편미분 값들 리턴
    return gradients

def update(parameters, gradients, alpha, m):
    """계산한 경사로 가중치와 편향을 업데이트 하는 함수"""
    L = len(parameters) // 2
    
    for layer in range(1, L+1):
        parameters['W'+str(layer)] -= alpha * gradients['dW'+str(layer)] / m
        parameters['b'+str(layer)] -= alpha * gradients['db'+str(layer)] / m
    
    return parameters

def train_nn(X_train, Y_train, X_test, Y_test, neurons_per_layer, epoch, alpha):
    """신경망을 학습시키는 함수"""
    parameters = initialize_parameters(neurons_per_layer)
    loss_list = []
    m = X_train.shape[0]
    
    # epoch 번 경사 하강을 한다
    for i in range(epoch):
        parameters_copy = parameters.copy()
        # 모든 이미지에 대해서 경사 계산 후 평균 계산
        for x, y in zip(X_train, Y_train):
            prediction, cache = feed_forward(x, parameters)
            gradients = back_prop(prediction, y, cache, parameters)
            parameters_copy = update(parameters_copy, gradients, alpha, m)
        
        # 가중치와 편향 실제로 업데이트
        parameters = parameters_copy
        loss_list.append(compute_loss(X_train, Y_train, parameters))
        print('{}번째 경사 하강, 테스트 셋에서 성능: {}'.format(i+1, round(compute_accuracy(X_test, Y_test, parameters), 2)))     
            
    return loss_list, parameters


"""디버깅을 위한 시각화 코드(쥬피터를 사용하시면 실행 코드 가장 아래줄에 넣어주세요!)
plt.plot(loss_list)
plt.show()
"""

# 테스트 코드
neurons_per_layer = [784, 128, 64, 10]
parameters = initialize_parameters(neurons_per_layer)

loss_list, parameters = train_nn(X_train, Y_train, X_test, Y_test, neurons_per_layer, 25, 300)

```

