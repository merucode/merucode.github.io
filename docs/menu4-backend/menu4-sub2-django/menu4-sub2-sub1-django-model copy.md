---
layout: default
title: Django Summary
parent: Django
grand_parent: Backend
nav_order: 1
---

# Django Summary
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

## STEP 1. Django Model

### Step 1-1. Implement Model Relation

### Step 1-1. Implement Model Relation

*  `models.py`
	```python
	### 1:N
	class N_Model(models.Model): 
		... 
		field_name = models.ForeignKey(<1_model>, on_delete=<option>, ...)
	# on_delete option : CASCADE, PROTECT, SET_NULL(null=True 필요) 


	### 1:1
	class dependent_Model(models.Model):
		...
		field_name = models.OneToOneField(<main_model>, on_delete=<option>, ...)
	

	### M:N
	class  MyModel(models.Model):
		...
		field_name = models.ManyToManyField(<to_model>, ...)


	### self relation <to_model> as 'self'
	class User(AbstractUser): 
		... 
		# 팔로우(M:N)
		following = models.ManyToManyField('self', symmetrical=False)	
		# 친구(M:N)
		friends = models.ManyToManyField('self', symmetrical=True)
		
	# 댓글에 댓글(1:N)
	class Comment(models.Model): 
		... 
		parent_comment = models.ForeignKey('self') 
		# symmetrical은 다대다 관계에서만 사용합니다
	
	
	### Generic
	from django.contrib.contenttypes.models import ContentType 
	from django.contrib.contenttypes.fields import GenericForeignKey
	class MyModel(models.Model): 
		...
		content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE) 
		object_id = models.PositiveIntegerField() 
		content_object = GenericForeignKey('content_type', 'object_id')
	```