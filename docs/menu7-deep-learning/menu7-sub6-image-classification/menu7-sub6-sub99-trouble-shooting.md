---
layout: default
title: Trouble Shooting
parent: Image Classification
grand_parent: Deep Learning
nav_order: 99
---

# Trouble Shooting
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

## STEP 1. 학습 시 `RuntimeError: stack expects each tensor to be equal size` 발생

### Step 1-1. Trouble

* `RuntimeError: stack expects each tensor to be equal size, but got [3, 128, 128] at entry 0 and [1, 128, 128] at entry 16`

### Step 1-2. Cause

* 학습 중 배치 파일에 채널 수가 맞지 않는 이미지 존재

### Step 1-3. Solution

* 채널 수가 맞지 않는 이미지 확인

```python
from torchvision.transforms import Compose
# trans = Compose([ToPILImage()])
not_match_img_idx = []

print('train')
for i in range(0,len(train_set)):
  if train_set[i][0].shape != torch.Size([3, 128, 128]):
    print(train_set[i][0].shape, i)
    not_match_img_idx.append(i)
    #plt.imshow(trans(train_set[i][0]))
    #plt.show()

print('test')
for i in range(0,len(test_set)):
  if test_set[i][0].shape != torch.Size([3, 128, 128]):
    print(test_set[i][0].shape, len(train_set) + i)
    not_match_img_idx.append(len(train_set) + i)
    # plt.imshow(trans(test_set[i][0]))
    # plt.show()

print(not_match_img_idx)
```

* 채널 수가 맞지 않는 이미지 삭제

```python
import os

images = sorted(glob.glob(path_to_image+"/*.jpg"))
annotations = sorted(glob.glob(path_to_annotation + "/*.png"))

for i in not_match_img_idx:
  print('Remove : ', images[i])
  print('Remove : ', annotations[i])

  os.remove(images[i])
  os.remove(annotations[i])

# plt.imshow(Image.open('./oxford-iiit-pet/images/staffordshire_bull_terrier_22.jpg'))
# plt.show()
```
