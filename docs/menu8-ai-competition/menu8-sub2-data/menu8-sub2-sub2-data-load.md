---
layout: default
title: Data Load
parent: Data
grand_parent: AI Competition
nav_order: 2
---

# Data Load
{: .no_toc }


* [[blog] Pytorch의 Dataloader 함수의 num_workers 1 이상 이용하기](https://velog.io/@giseg2118/Pytorch%EC%9D%98-Dataloader-%ED%95%A8%EC%88%98%EC%9D%98-numworkers-0-%EC%9D%B4%EC%83%81-%EC%9D%B4%EC%9A%A9%ED%95%98%EA%B8%B0)

```
DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_gpu*4,pin_memory=True, drop_last=True)
```