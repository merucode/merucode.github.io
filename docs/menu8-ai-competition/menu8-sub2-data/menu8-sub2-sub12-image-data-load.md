---
layout: default
title: [DL-Image] Data Load
parent: Data
grand_parent: AI Competition
nav_order: 12
---

# [DL-Image] Data Load 
{: .no_toc }


## STEP 0. Unzip

```python
from zipfile import ZipFile

data_path = '/content/'

with ZipFile(data_path + 'chest-xray-pneumonia.zip') as zipper:
  zipper.extractall()
```

<br>

## STEP 1. Load Data

### Step 1-1. folder and file

```python
import pandas as pd

# 데이터 경로
data_path = '/content/chest_xray/'

# 훈련, 검증, 테스트 데이터 경로 설정
train_path = data_path + 'train/'
valid_path = data_path + 'val/'
test_path = data_path + 'test/'

# 이미지 파일명 리스트 생성
from glob import glob

all_normal_imgs = []    # 모든 정상 이미지를 담을 리스트 초기화
all_pneumonia_imgs = [] # 모든 폐렴 이미지를 담을 리스트 초기화

for cat in ['train/', 'val/', 'test/']:
  data_cat_path = data_path + cat
  # 정상, 폐렴 이미지 경로
  normal_imgs = glob(data_cat_path + 'NORMAL/*')
  pneumonia_imgs = glob(data_cat_path + 'PNEUMONIA/*')
  # 정상, 폐렴 이미지 경로를 리스트에 추가
  all_normal_imgs.extend(normal_imgs)
  all_pneumonia_imgs.extend(pneumonia_imgs)

print(f"정상 흉부 이미지 개수: {len(all_normal_imgs)}")
print(f"폐렴 흉부 이미지 개수: {len(all_pneumonia_imgs)}")
```

### Step 1-2. from csv file

```python
import pandas as pd

# 데이터 경로
data_path = '/content/chest_xray/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sample_submission.csv')
```

<br>

## STEP. Dataset

### Step. Csv DF exsist

```python
import cv2
from torch.utils.data import Dataset
import numpy as np

class ImageDataset(Dataset):
  # 초기화 메서드(생성자)
  def __init__(self, df, img_dir='./', transform=None, is_test=False):
    super().__init__() # 상송받은 Dataset 생성자 호출
    # 전달받은 인수들 저장
    self.df = df
    self.img_dir = img_dir
    self.transform = transform
    self.is_test = is_test

  # 데이터셋 크기 반환 메서드
  def __len__(self):
    return len(self.df)

  # idx 해당하는 데이터 반환 메서드
  def __getitem__(self, idx):
    img_id = self.df.iloc[idx, 0]               # 이미지 ID
    img_path = self.img_dir + img_id + '.jpg'   # 이미지 파일 경로
    image = cv2.imread(img_path)                # 이미지 파일 읽기
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 이미지 색상 보정

    if self.transform is not None:  # 변화기가 있다면 이미지 변환
      # image = self.transform(image=image)['image']    # albumentations 모듈 변환기
      image = self.transform(image)                 # torchvision transform 모듈 변환기

    # 테스트 데이터면 이미지 데이터만 반환, 그렇치 않으면 타깃값도 반환
    if self.is_test:
      return image          # 테스트용
    else:
      # 타깃값 4개 중 가장 큰 값의 인덱스
      label = np.argmax(self.df.iloc[idx, 1:5])
      return image, label   # 훈련/검증용
```

### Step. ImageFolder

* ImageFolder 별로 Normal, target 구분되어 있는 경우
* albumentations 변환기 사용 불가

<br>

## STEP. Dataloader

* [[blog] Pytorch의 Dataloader 함수의 num_workers 1 이상 이용하기](https://velog.io/@giseg2118/Pytorch%EC%9D%98-Dataloader-%ED%95%A8%EC%88%98%EC%9D%98-numworkers-0-%EC%9D%B4%EC%83%81-%EC%9D%B4%EC%9A%A9%ED%95%98%EA%B8%B0)

```
DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_gpu*4,pin_memory=True, drop_last=True)
```