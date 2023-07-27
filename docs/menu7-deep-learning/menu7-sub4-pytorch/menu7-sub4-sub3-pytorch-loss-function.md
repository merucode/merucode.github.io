---
layout: default
title: Pytorch Loss Function
parent: Pytorch
grand_parent: Deep Learning
nav_order: 3
---
# Pytorch Loss function
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


* CrossEntropyLoss
    * 다중분류 사용
    * input : softmax 출력값
    * target : 정답 레이블    ex) [0, 0, 0, 1 ]의 타깃값 → 3

```
import torch.nn as nn
loss_function = nn.CrossEntropyLoss()
```