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
* [빠른 시작(QUICKSTART)](https://tutorials.pytorch.kr/beginner/basics/quickstart_tutorial.html)

<!------------------------------------ STEP ------------------------------------>

## STEP 0. Check Cuda

* `bash`

```bash
!nvidia-smi
```



* `python`

```python
import torch

# GPU 사용 가능 -> True, GPU 사용 불가 -> False
print(torch.cuda.is_available())
```



<br>

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

# nn.Flatten : 28x28를 784 픽셀 값을 갖는 연속된 배열로 변(dim=0의 미니배치 차원은 유지) ex) 3x28x28 > 3x784
# nn.Sequential : 순서를 갖는 모듈의 컨테이너

# 인스턴스(instance)를 생성하고 이를 device 로 이동한 뒤, 구조(structure)를 출력
model = NeuralNetwork().to(device)
print(model)

# 모델을 사용하기 위해 입력 데이터를 전달
X = torch.rand(1, 28, 28, device=device)
logits = model(X)   # forward 실행(model.forward 직접 호출 안함)
# logits 결과는 [1 ,10] tenser
# 2차원 텐서의 dim=0은 각 분류(class)에 대한 원시(raw) 예측값 10개가, dim=1에는 각 출력의 개별 값들이 해당 
# dim 매개변수는 값의 합이 1이 되는 차원
pred_probab = nn.Softmax(dim=1)(logits) 
y_pred = pred_probab.argmax(1)  # 최대값을 가지는 요소의 인덱스 추출
print(f"Predicted class: {y_pred}")
```

<!------------------------------------ STEP ------------------------------------>

<br>

## STEP 5. 모델 매개변수 최적화

```python
### 하이퍼파라미터(Hyperparameter)
learning_rate = 1e-3    # 각 배치/에폭에서 모델의 매개변수를 조절하는 비율. 값이 작을수록 학습 속도가 느려지고, 값이 크면 학습 중 예측할 수 없는 동작이 발생
batch_size = 64         # 매개변수가 갱신되기 전 신경망을 통해 전파된 데이터 샘플의 수
epochs = 5              # 데이터셋을 반복하는 횟수

### 손실 함수(loss function)


### 옵티마이저(Optimizer)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 학습 단계(loop)에서 최적화는 세단계로 이루어짐
# 1. optimizer.zero_grad()를 호출하여 모델 매개변수의 변화도를 재설정
# 2. loss.backwards()를 호출하여 예측 손실(prediction loss)을 역전파(손실의 변화도를 저장)
# 3. 변화도를 계산한 뒤에는 optimizer.step()을 호출하여 역전파 단계에서 수집된 변화도로 매개변수를 조정


### 최적화 단계(Optimization Loop) : 
# 에폭 : 최적화 단계의 각 반복(iteration). 하나의 에폭은 다음 두 부분으로 구성
#       학습 단계(train loop) - 학습용 데이터셋을 반복(iterate)하고 최적의 매개변수로 수렴
#       검증/테스트 단계(validation/test loop) - 모델 성능이 개선되고 있는지를 확인하기 위해 테스트 데이터셋을 반복(iterate)


### 전체 구현
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


### 손실 함수와 옵티마이저를 초기화하고 train_loop와 test_loop에 전달
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

<!------------------------------------ STEP ------------------------------------>

<br>

## STEP 6. 모델 저장하고 불러오기

```python
import torch
import torchvision.models as models

### 모델 가중치 저장하고 불러오기
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

# 모델 가중치를 불러오기 위해서는 먼저 동일한 모델의 인스턴스(instance)를 생성
model = models.vgg16() # 여기서는 ``weights`` 를 지정하지 않았으므로, 학습되지 않은 모델을 생성합니다.
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()    # 추론(inference)을 하기 전에 model.eval() 메소드를 호출하여 드롭아웃(dropout)과 배치 정규화(batch normalization)를 평가 모드(evaluation mode)로 설정


### 모델의 형태를 포함하여 저장하고 불러오기
torch.save(model, 'model.pth')
model = torch.load('model.pth')
```

* 추가 학습 : [PYTORCH에서 일반적인 체크포인트(CHECKPOINT) 저장하기 & 불러오기](https://tutorials.pytorch.kr/recipes/recipes/saving_and_loading_a_general_checkpoint.html) 