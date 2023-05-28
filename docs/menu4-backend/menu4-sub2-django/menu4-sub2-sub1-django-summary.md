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

  <br>

  <!------------------------------------ STEP ------------------------------------>

## STEP 2. Django CRUD

### Step 2-1. CRUD 
* `QuerySet` : 여러 오브젝트의 집합
* Example Data
	```python
	class Skill(models.Model): 
		title = models.CharField(max_length=50, blank=False) 
		summary = models.TextField(blank=False) 
		dt_created = models.DateTimeField(auto_now_add=True) 
		
		def  __str__(self): 
			return self.title
	```

*  `python manage.py shell`
	```python
	### .all()
	Skill.objects.all()
	
	### .filter()
	# filter은 '__' 사용하여 연산자 사용
	Skill.objects.filter(id=3)	
	# title에 'C++' 포함
	Skill.objects.filter(title__contains='C++')
	# title이 '기초'로 시작하고 'Python'을 포함
	Skill.objects.filter(title__startswith='기초').filter(title__contains='Python')

	### .filter()와 .all() 쿼리셋 반환
	### .get()은 오브젝트 자체 반환
	
	### .get()
	skill = Skill.objects.get(id=1)
	print(skill.id) # 1
	skill = Skill.objects.get(title__contains='C++')
	# 오브젝트가 여러 개이거나 없다면 오류

	### .order_by()
	Skill.objects.order_by('dt_created')  # 오름차순(오래된순)
	Skill.objects.order_by('-dt_created') # 내림차순(최신순)
	# summary에 '웹 개발'이 들어가는 필터 후 정렬
	Skill.objects.filter(summary__contains='웹 개발').order_by('dt_created')
	
	### .exists()
	Skill.objects.filter(title__startswith='프로그래밍 기초').exists() # True
	Skill.objects.filter(title__contains='C++').exists() # False
	
	### .count()
	Skill.objects.count() # 5
	Skill.objects.filter(title__startswith='프로그래밍 기초').count() # 2
	
	### Create(.create())
	Skill.objects.create(title='웹 개발', summary='HTML과 CSS')
	# id, dt_created는 자동 생성
	
	### Update
	skill = Skill.objects.get(id=6)	# 가져오기
	skill.title = 'Web 퍼블리싱'		# 수정
	skill.save()					# 저장

	### Delete(.delete())
	skill = Skill.objects.get(id=6) # 가져오기
	skill.delete() 					# 삭제
	```
### Step 2-2. Generic View와 CRUD

* Generic View 사용시 django가 알아서 CRUD 수행

	```python
	class SkillUpdateView(UpdateView): 
		model = Skill 
		form_class = SkillForm 
		template_name = 'myapp/skill_form.html' 
		pk_url_kwarg = 'skill_id'
	```
