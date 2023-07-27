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


<br>

<!------------------------------------ STEP ------------------------------------>

## STEP 4. Learning



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

<br>


<!------------------------------------ STEP ------------------------------------>

## STEP 5. Evaluation



<br>