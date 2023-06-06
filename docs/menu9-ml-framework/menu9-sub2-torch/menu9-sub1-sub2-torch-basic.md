---
layout: default
title: Basic
parent: Torch
grand_parent: ML Framework
nav_order: 2
---

# Torch Basic
{: .no_toc }


* gather 함수(https://pajamacoder.tistory.com/2)
	```python
	torch.gather(input, dim, index, out=None, sparse_grad=False)  
	→ TensorGathers values along an axis specified by dim.
	```
	* input: input으로 받는 텐서에요.
	* dim: 어떤 axis를 변동을 줄지 
	* index: output의 shape를 지정하고 , 어떻게 치환해줄지
	* ex)
		```python
		import torch 
		import numpy as np
		 
		t = torch.tensor([i for i in  range(4*2*3)]).reshape(4,2,3) 
		print(t)
		
		# 1,0,3 은 추출하고 싶은 원소의 타겟 dimension의 원소 index 이다. 
		ind_A = torch.tensor([1,0,3]) 
		
		# torch.gather()에서 index tensor의 차원수는 input tensor의 차수원수와 같아야 한다. 
		# 즉 이 예제에서 t.dim() == ind_A.dim() 이어야 torch.gather()를 사용 할 수 있다.  
		# 이를 위해 ind_A의 차원을 t와 맞춰 주면 
		ind_A = ind_A.unsqueeze(1).unsqueeze(2) 
		
		# 여기 까지는 차원의 수만 맞춘것이다. gather가 정상적으로 동작하기 위해서는 타겟으로 하는 dimension를 제외한  # t와 ind_A의 나머지 dimension의 값이 같아야 한다. 
		# 즉 내가 추출하고자 하는 원소가 dim 0의 원소라면 t.size(), ind_A.size() 에서 
		# t.size(1)==ind_A.size(1) and t.size(2)==ind_A.size(2)의 조건을 만족해야 한다. 
		ind_A = ind_A.expand(ind_A.size(0), t.size(1), t.size(2)) 
		
		# 여기 까지 코드를 실행 시킨면 ind_A.size() = [3,2,3] 이고 t.size()=[4,2,3] 이다.  
		# 앞서 설명했듯 target dimension 인 ind_A.size(0)!=t.size(0) 을 제외한 1,2 차원의 값이 2,3으로 같다.  # 최종적으로 위 그림 같이 dim=0에서 1,0,3 번째 원소를 추출하여 새로운 텐서를 구성하기 위해 아래 구문을 실면행하면된다. 
		res = res.gather(0,ind_A)
		```
	* 장황해 지만 다시 정리하면 torch.gather 메서드는 input tensor의 타겟 dimension으로 부터 원하는 원소를 추출해 새로운 텐서를 만들때 사용 하며 index tensor는 다음을 만족해야 한다.
		1. inputTensor.dim()==indexTensor.dim()
		2. inputTensor.size() == [x,y,z] 이고 indexTensor.size()==[x',y',z'] 일 때
			* 타겟 dimension=0 이면 y==y' and z==z' 이어야 한다.
			* 타겟 dimension=1 이면 x==x' and z==z' 이어야 한다.
			* 타겟 dimension=2 이면 x==x' and y==y' 이어야 한다.
			* 타겟 dimension 이란 torch.gather(dim=x, indexTensor) 에서 dim 파라미터에 할당되는 값을 의미한다.
