---
layout: default
title: Colab Basic
parent: Pytorch
grand_parent: Deep Learning
nav_order: 2
---

# Colab Basic
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

## STEP 1. Colab 

### Step 1-1. Pytorch Install

### Step 1-2. Version Cehck

```python
import sys
import torch
print(f"Python version:{sys.version}")                  # python
print("Torch version:{}".format(torch.__version__))     # torch
print("cuda version: {}".format(torch.version.cuda))    # cuda
print("cudnn version:{}".format(torch.backends.cudnn.version()))    # cudnn
```
