---
layout: default
title: Pytorch Basic Form
parent: Pytorch
grand_parent: Deep Learning
nav_order: 7
---

# Pytorch Basic Form
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

## STEP 1. Basic Coding Style

```python
### ① Module Class
class Net(nn.Module):
  def __init__(self):
    # 신경망 구성요소 정의

  def forward(self, input):
    # 신경망 동작 정의
    return output

### ② Data Set Class(Load Data)
class Dataset():
  def __init__(self):
    # Load Data

  def __len__(self):
    # return len of data
    return len(data)

  def __getitem__(self, i):
    # i 번째 입력 데이터와 정답 반환
    return data[i], label[i]

### ③ Learning
for data, label in DataLoader():
  # Prediction
  predction = model(data)
  # Loss
  loss = LossFunction(prediction, label)
  # Backward()
  loss.backward()
  # Update weight
  optimizer.step()
```

<!------------------------------------ STEP ------------------------------------>

## STEP 2. Data 


<br>

<!------------------------------------ STEP ------------------------------------>

## STEP 3. Module

### Step 3-. TL Output Control

```python
from efficientnet_pytorch import EfficientNet # EfficientNet 모듈
import torch.nn as nn

# 사전 훈련된 'efficientnet-b7' 모델 불러오기
model = EfficientNet.from_pretrained('efficientnet-b0')

# 사전 모델 마지막 계층 수정(출력값 갯수 수정)
model._fc = nn.Sequential(
    nn.Linear(model._fc.in_features, model._fc.out_features), # 2560 > 1000
    nn.ReLU(),          # 활성화 함수
    #nn.Dropout(p=0.5),  # 50% 드롭아웃
    nn.Linear(model._fc.out_features, 2) # 1000 > 2
)
```


### Step 3-. Info

* Model 파라미터 총 갯수 확인

```
### 단일 모델
print(f"모델 파라미터 갯수: {sum(param.numel() for param in model.parameters())}")

### 모델 리스트(앙상블)
for idx, model in enumerate(models_list):
  num_parmas = sum(param.numel() for param in model.parameters())
  print(f"모델{idx+1} 파라미터 갯수: {num_parmas}")

```

<br>

<!------------------------------------ STEP ------------------------------------>

## STEP 4. Learning

### Step 4-1. Setting

* 병렬 GPU 사용

```
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model = nn.DataParallel(model)       # 병렬 GPU 사용
```

### Step 4-2. 훈련/검증 동시 진행 포멧

* 에폭 진행 중 과대 적합 확인 가능

```
#훈련 및 성능 검증 포멧
for epoch in range(epochs):
  # == [훈련] ================================
  # 모델을 훈련 상태로 설정
  # 에폭별 손실값 초기화(훈련 데이터용)

  # 반복 횟수 만큼 반복
  for images, labels in tqdm(loader_train):
    # 기출기 초기화
    # 순전파
    # 손실값 계산(훈련 데이터용)
    # 역전파
    # 가중치 갱신

  # == [검증] ================================
  # 모델을 평가 상태로 설정
  with torch.no_grad(): # 기울기 계산 비활성화
    # 미니 배치 단위로 검증
    for images, labels in loader_valid:
      # 순전파
      # 손실값 계산(검증 데이터용)
      # 예측값 및 실제값 계산
  # 검증 데이터 손실값 및 ROC AUC 점수 출력
```

### Step 4-3. 훈련 함수 포멧(최적 가중치 저장)

```python
def train(model, loader_train, loader_valid, criterion, optimizer, 
          scheduler=None, epochs=10, save_file='model_state_dic.pth'):
  # 총 에폭만큼 반복
  for epoch in range(epochs):
    # == [훈련] ================================
    # 모델을 훈련 상태로 설정
    # 에폭별 손실값 초기화(훈련 데이터용)

    # 반복 횟수 만큼 반복
    for images, labels in tqdm(loader_train):
      # 기출기 초기화
      # 순전파
      # 손실값 계산(훈련 데이터용)
      # 역전파
      # 가중치 갱신

    # == [검증] ================================
    # 모델을 평가 상태로 설정
    with torch.no_grad(): # 기울기 계산 비활성화
      # 미니 배치 단위로 검증
      for images, labels in loader_valid:
        # 순전파
        # 손실값 계산(검증 데이터용)

    # == [최적 모델 가중치 찾기] ===============
    # 현 에폭에서의 검증 데이터 손실값이 지금까지 중 가장 작다면
            # 현 에폭의 모델 가중치(현재까지의 최적 모델 가중치) 저장

    return torch.load(save_file)  # 최적 모델 가중치 반환
```


<br>




<!------------------------------------ STEP ------------------------------------>

## STEP 5. Evaluation



<br>