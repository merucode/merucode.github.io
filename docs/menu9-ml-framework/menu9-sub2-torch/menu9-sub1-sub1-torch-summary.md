---
layout: default
title: Summary
parent: Torch
grand_parent: ML Framework
nav_order: 1
---

# Summary
{: .no_toc }

## STEP 1. Basic

* [파이토치 한국어 튜토리얼](https://tutorials.pytorch.kr/)
* [PyTorch로 시작하는 딥 러닝 입문](https://wikidocs.net/52460)

### Step 1-1. Tensor
```python
### Import
import torch
import numpy as np

### Create
data = [[1,2], [3,4]]       # directly create from data
x_data = torch.tensor(data)

arr = np.array(data)        # create from numpy
x_data = torch.tensor(arr)

# create from other tensor
x_one = torch.ones_like(x_data)                     # x_data 속성 유지
x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data 속성 덮어쓰기

# create with random or constant
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

### Attribute
tensor.shape
tensor.dtype
tensor.device

### Operation
if torch.cuda.is_available():   # GPU가 존재하면 텐서를 이동합니다
    tensor = tensor.to("cuda")

# 텐서 합치기
t1 = torch.cat([tensor, tensor, tensor], dim=1)

# 산술 연산(Arithmetic operations)
y1 = tensor @ tensor.T  # 두 텐서 간의 행렬 곱(matrix multiplication)
z1 = tensor * tensor    # 요소별 곱(element-wise product)을 계산

# 바꿔치기(in-place) 연산(_)

# 단일-요소(single-element) 텐서 값 반환
agg = tensor.sum()
agg_item = agg.item()   # float 타입 value 반환

### Numpy 변환
# CPU 상의 텐서와 NumPy 배열은 메모리 공간을 공유하기 때문에, 하나를 변경하면 다른 하나도 변경
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)     # tensor에도 반영
```


### Step 1-2. Dataset, Dataloader

```python
### Load dataset
# Dataset 은 샘플과 정답(label)을 저장
# DataLoader 는 Dataset 을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 감쌈
from torch.utils.data import Dataset
from torchvision import datasets

import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

### 파일에서 사용자 정의 데이터셋 만들기(class CustomImageDataset(Dataset))

### DataLoader로 학습용 데이터 준비하기(DataLoader)
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

### DataLoader를 통해 순회하기(next(iter))
# 이미지와 특징(feature), 정답(label)을 표시
train_features, train_labels = next(iter(train_dataloader))
img = train_features[0].squeeze()   # squeeze() 함수는 차원의 원소가 1인 차원을 제거
label = train_labels[0]

```

### Step 1-3. Transform

```python
# 변형(transform) 을 해서 데이터를 조작하고 학습에 적합하게 만듬
### Import
from torchvision.transforms import ToTensor, Lambda

import torch
from torchvision import datasets

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

### ToTensor()
# ToTensor 는 PIL Image나 NumPy ndarray 를 FloatTensor 로 변환하고
# 이미지의 픽셀의 크기(intensity) 값을 [0., 1.] 범위로 비례하여 조정(scale)

### Lambda 변형(Transform)
# 정수를 원-핫으로 부호화된 텐서로 바꾸는 함수를 정의
# 먼저 (데이터셋 정답의 개수인) 크기 10짜리 영 텐서(zero tensor)를 만들고 
# scatter_ 를 호출하여 주어진 정답 y 에 해당하는 인덱스에 value=1 을 할당
```

### Step 1-4. 신경망 모델 구성하기

```python
### Import
# torch.nn 네임스페이스는 신경망을 구성하는데 필요한 모든 구성 요소를 제공
# PyTorch의 모든 모듈은 nn.Module 의 하위 클래스(subclass)

from torch import nn
from torch.utils.data import DataLoader

import os
import torch
from torchvision import datasets, transforms

### 학습을 위한 장치 얻기
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

### 클래스 정의하기
# __init__ 에서 신경망 계층들을 초기화
# nn.Module 을 상속받은 모든 클래스는 forward 메소드에 입력 데이터에 대한 연산들을 구현
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)  # 인스턴스(instance)를 생성하고 이를 device 로 이동
X = torch.rand(1, 28, 28, device=device)
#  일부 백그라운드 연산들 과 함께 모델의 forward 를 실행. model.forward() 를 직접 호출 X
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)  # 예측 확률
y_pred = pred_probab.argmax(1)           # 예측 확률에서 가장 큰 값의 index 호출

# 모델 매개변수
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

### Step 1-5. TORCH.AUTOGRAD를 사용한 자동 미분

```python
# PyTorch에는 torch.autograd라고 불리는 자동 미분 엔진이 내장(역전파)
import torch

x = torch.ones(5)   # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

### 변화도(Gradient) 계산하기
loss.backward()
print(w.grad)
print(b.grad)

### 변화도 추적 멈추기
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
# or
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
```

### Step 1-6. TORCH.AUTOGRAD를 사용한 자동 미분

```python
### 모델 매개변수 최적화하기
...

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

### 하이퍼파라미터(Hyperparameter)
learning_rate = 1e-3
batch_size = 64
epochs = 5

### 손실 함수와 Optimizer
loss_fn = nn.CrossEntropyLoss() # 손실 함수를 초기화
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

### 최적화 단계(Optimization Loop)
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 예측(prediction)과 손실(loss) 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


```


```python
torch.gather(input, dim, index, out=None, sparse_grad=False)  
```