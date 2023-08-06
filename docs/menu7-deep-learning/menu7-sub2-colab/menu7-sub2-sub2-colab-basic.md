---
layout: default
title: Colab Basic
parent: Colab
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

### Step 1-3. Colab 런타임 유지

```javascript
function ClickConnect(){
    console.log("코랩 연결 끊김 방지"); 
    document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect, 60 * 1000)
```

### Step 1-4. GitHub Clone

```bash
!git clone -b 31-Colab-Study-Udemy-Custom_ENV_snake_game https://github.com/merucode/RL.git
!cd RL && mv * ../
```