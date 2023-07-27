---
layout: default
title: Image Performance Improvement
parent: Image
grand_parent: AI Competition
nav_order: 5
---

# Image Performance Improvement
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

## STEP 1. Dataset

### Step 1-1. 이미지 증강

```python
from torchvision import transforms as T   # 이미지 변환을 위한 모듈

transform_train = T.Compose([T.ToTensor(),
                       T.Pad(32, padding_mode='symmetric'),
                       T.RandomHorizontalFlip(),
                       T.RandomVerticalFlip(),
                       T.RandomRotation(10),
                       T.Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))
                       ])

transform_test = T.Compose([T.ToTensor(),
                       T.Pad(32, padding_mode='symmetric'),                      
                       T.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))
                       ])

dataset_train = ImageDataset(df=train, img_dir='train/', transform=transform_train)
dataset_valid = ImageDataset(df=valid, img_dir='train/', transform=transform_test)
```

<br>

## STEP 2. Model

### Step 2-1. 더 깊은 모델 사용

### Step 2-2. 활성화 함수 변경


<br>

## STEP 3. Learning

### Step 3-1. Optimizer 변경

* AdamW : adam에 가중치 감쇠 추가 적용, 일반화 성능 우수(과대적합 억제)

```
from torch.optim.adamw import AdamW
optim = AdamW(model.parameters(), lr=0.00006, weight_decay=0.0001)
```

* Adammax : 

```
from torch.optim.adamax import Adamax 
optim = Adamax(model.parameters(), lr=0.00006)
```

* 학습률 : 배치 크기가 줄어들수록 학습사이즈 작게, 배치 크기가 클수록 학습률 크게

### Step 3-2. 에폭 증가