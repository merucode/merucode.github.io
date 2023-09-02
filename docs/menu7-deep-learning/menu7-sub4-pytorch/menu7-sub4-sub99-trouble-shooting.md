---
layout: default
title: Pytorch Toruble Shooting
parent: Pytorch
grand_parent: Deep Learning
nav_order: 99
---
# Pytorch Toruble Shooting
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

## STEP. indices should be either on cpu or on the same device as the indexed tensor (cpu)

### Step 1-1. Trouble
    * 훈련/검증 에폭 동시 진행 중 검증 데이터 cpu 계산을 위한 코드에서 에러 발생 

### Step 1-2. Cause

### Step 1-3. Solution
    * 검증 데이터 계산 코드 이전 labels를 cpu로 전송
    
```
      ...
      labels = labels.to("cpu") # ERROR HANDLING: indices should be either on cpu or on the same device as the indexed tensor (cpu)
      true_onehot = torch.eye(4)[labels].cpu().numpy()     # 실제값 (원-핫 인코딩 형식)
      ...
```

<br>
<!------------------------------------ STEP ------------------------------------>

## STEP. Nan Error

### Step 1-1. Trouble
  * 훈련 과정 중 tensor 값이 nan으로 되면서 error 발생

### Step 1-2. Cause
  * log(0) 또는 value/0 등 과 같은 연산
  * lr의 영향에 의한 발산

### Step 1-3. Solution
  * log(a + 1e-6) 또는
  * a = torch.nan_to_num(a)
  * Reference Site
    * [[blog]](https://powerofsummary.tistory.com/165)


<br>
<!------------------------------------ STEP ------------------------------------>


## STEP. Checkpoint Load 중 Device ERROR

### Step 1-1. Trouble
  * Chekcpoint Load 후 훈련 중 optimizer.step() 시 device error발생

### Step 1-2. Cause

### Step 1-3. Solution

```python
model.to(device)

ckpt = torch.load(<model_path>, map_location=device)

model.load_state_dict(ckpt['state_dict'])
optimizer.load_state_dict(ckpt['optimizer'])
scheduler.load_state_dict(ckpt['scheduler'])

del ckpt
```

  * Reference Site
    * [[pytorch]](https://github.com/pytorch/pytorch/issues/2830#issuecomment-718816292)