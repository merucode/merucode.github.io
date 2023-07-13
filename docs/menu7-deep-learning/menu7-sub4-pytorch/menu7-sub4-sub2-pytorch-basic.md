---
layout: default
title: Pytorch Basic Start
parent: Pytorch
grand_parent: Deep Learning
nav_order: 2
---
# Pytorch Basic
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

<!------------------------------------ STEP ------------------------------------>

## STEP 0. Reference Site

* [파이토치(PYTORCH) 기본 익히기](https://tutorials.pytorch.kr/beginner/basics/intro.html)

<!------------------------------------ STEP ------------------------------------>

<br>

## STEP 1. Tenser

```python
### Tenser: GPU나 다른 하드웨어 가속기에서 실행할 수 있다는 점만 제외하면 NumPy 의 ndarray와 유사

### GPU가 존재하면 텐서를 이동
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
```

<!------------------------------------ STEP ------------------------------------>

<br>

## STEP 2. DATASET과 DATALOADER

```python
### Dataset: 샘플과 정답(label)을 저장
### DataLoader: Dataset 을 샘플에 쉽게 접근할 수 있도록 순회 가능한 객체(iterable)로 만듬


### DataLoader로 학습용 데이터 준비
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

<!------------------------------------ STEP ------------------------------------>

<br>

## STEP 3. 변형(TRANSFORM)

```python
### TRANSFORM : 데이터를 조작하고 학습에 적합하게 만듭

### 모든 TorchVision 데이터셋들은 변형 로직을 갖는, 호출 가능한 객체(callable)를 받는 매개변수 두개를 가짐
### transform : 특징(feature)을 변경
### target_transform : 정답(label)을 변경

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)


### ToTensor() 
# 정규화 : PIL Image나 NumPy ndarray를 FloatTensor로 변환하고, 이미지의 픽셀의 크기(intensity) 값을 [0., 1.] 범위로 비례하여 조정(scale)

### Lambda 변형(Transform) 
# 원핫인코딩 : 이 함수는 먼저 (데이터셋 정답의 개수인) 크기 10짜리 영 텐서(zero tensor)를 만들고, 
#             scatter_ 를 호출하여 주어진 정답 y 에 해당하는 인덱스에 value=1 을 할당
target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```


<!------------------------------------ STEP ------------------------------------>


<br>


## STEP 4. 신경망 구성

```python

### 학습을 위한 장치 얻기
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


### 클래스 정의하기
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

# 인스턴스(instance)를 생성하고 이를 device 로 이동한 뒤, 구조(structure)를 출력
model = NeuralNetwork().to(device)
print(model)

# 모델을 사용하기 위해 입력 데이터를 전달
X = torch.rand(1, 28, 28, device=device)
logits = model(X)   # forward 실행(model.forward 직접 호출 안함)
# logits 결과는 [1 ,10] tenser
# 2차원 텐서의 dim=0은 각 분류(class)에 대한 원시(raw) 예측값 10개가, dim=1에는 각 출력의 개별 값들이 해당 
# dim=1(개별 값)들을 기준으로 하여 모두 변환하여 합산시 1로 만드는 softmax 변환 수행
pred_probab = nn.Softmax(dim=1)(logits) 
y_pred = pred_probab.argmax(1)  # 최대값을 가지는 요소의 인덱스 추출
print(f"Predicted class: {y_pred}")
```

<br>
