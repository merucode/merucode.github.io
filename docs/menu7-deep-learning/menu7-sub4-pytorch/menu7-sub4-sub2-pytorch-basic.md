---
layout: default
title: Pytorch Basic
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



<br>


<!------------------------------------ STEP ------------------------------------>

## STEP 5. Evaluation



<br>