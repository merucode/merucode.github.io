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