---
layout: default
title: Site
parent: Web
grand_parent: Project
nav_order: 2
---

# Site

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

## STEP 0. Setting

* We start from [[Backend]-[Django]-[Deploy-with-Docker]](https://merucode.github.io/docs/menu4-backend/menu4-sub2-django/menu4-sub2-sub8-deploy-with-docker.html#step-7-final-file-structure)



## STEP 1. Create App and Load Data

### Step 1-1. [Dev] Creat App

* **`bash`**

  ```bash
  $ docker compose up -d --build
  $ docker exec backend python manage.py startapp charts
  # check to create folder of charts
  $ docker compose donw -v
  ```

* **`django/mysite/settings.py`**

  ```python
  INSTALLED_APPS = [
  	...
      'django.contrib.staticfiles',
      'charts',	### Update
  ]
  ```

* **`django/mysite/urls.py`**

  ```python
  from django.contrib import admin
  from django.urls import path, include ## Update
  ...
  urlpatterns = [
      path('admin/', admin.site.urls),
      path('', include('charts.urls')), ## Update
  ]
  ...
  ```

* **`django/charts/urls.py`(create)**

  ```python
  from django.urls import path
  from . import views
  
  app_name = 'charts'
  
  urlpatterns = [
      path('', views.index, name='index'),
  ]
  ```

* **`django/charts/views.py`**

  ```python
  from django.shortcuts import render
  
  ### Add
  def index(request):
      return render(request, 'charts/index.html')
  ```

* **`django/templates/base.html`(create)**

  ```html
  <!-- % load static % -->
  <!doctype html>
  <html lang="ko">
  <head>
      <!-- Required meta tags -->
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
      <title>mysite</title>
  </head>
  <body>
  <!-- 기본 템플릿 안에 삽입될 내용 Start -->
  
  <!-- % block content % -->
  <!-- % endblock content % -->
      
  <!-- 기본 템플릿 안에 삽입될 내용 End -->
  </body>
  </html>
  ```

  


* **`django/templates/charts/index.html`(create)**

  ```html
  <!-- % load static % -->
  <!-- % block content % -->
  
  <h1>Hello!</h1>
  
  <!-- % endblock content % -->
  ```

* **`django/mysite/settings.py`**

  ```python
  ...
  TEMPLATES = [
      {
          'BACKEND': 'django.template.backends.django.DjangoTemplates',
          'DIRS': [BASE_DIR / 'templates'],	## Update
          ...
  ```

  

### Step 1-2. [Dev] Check

* **`bash`**

  ```bash
  $ docker compose up -d --build
  # connect to 'http://localhost:8000/'
  
  $ docker compose down -v
  ```



###  Step 1-3. Load Data

* Make `test table(test_1)` in AWS DB

  ![image-20230502093744048](./../../../images/menu11-sub1-sub2-site/image-20230502093744048-1682997331877-1.png)

* **`django/charts/models.py`**

  ```python
  from django.db import models
  
  class test_1_model(models.Model):
      date = models.DateField(primary_key=True, db_column='날짜')
      open = models.IntegerField(db_column='시가')
      high = models.IntegerField(db_column='고가')
      low = models.IntegerField(db_column='저가')
      close = models.IntegerField(db_column='종가')
      volume = models.IntegerField(db_column='거래량')
      volume_price = models.IntegerField(db_column='거래대금')
      rate_return = models.IntegerField(db_column='등락률')
      stock_name = models.CharField(max_length=20, db_column='stock_name')
      
      class Meta:
          managed = False
          db_table = "test_1"
  ```


* **`django/charts/urls.py*`**

  ```python
  ...
  urlpatterns = [
      path('', views.index, name='index'),
      path('chart/', views.chart, name='chart'),	## Add
  ]
  ```

* **`django/charts/views.py`**

  ```python
  from django.shortcuts import render
  from .models import test_1_model
  
  def index(request):
      return render(request, 'charts/index.html')
  
  def chart(request):
      search_stock_name = request.GET.get('stock_name')
      test_list = test_1_model.objects.filter(stock_name__exact=search_stock_name)
      context = {'test_list':test_list}
  
      return render(request, 'charts/chart.html', context)
  ```
  
* **`django/teplates/charts/index.html`**

  ```html
  <!-- % extends 'base.html' % -->
  <!-- % block content % -->
  
  <h1>index!!</h1>
  
  <form action = "{% url 'charts:chart' %}" method = "get">
      <label for="stock_name">STOCK NAME : </label>
      <input id="stock_name" type="text" name="stock_name">
      <input type="submit" value="OK">
  </form>
  
  <!-- % endblock content % -->
  ```

* **`django/teplates/charts/chart.html`**

  ```html
  <!-- % extends 'base.html' % -->
  <!-- % block content % -->
  
  <h1>chart!</h1>
  
  <form action = "" method = "get">
      <label for="stock_name">STOCK NAME : </label>
      <input id="stock_name" type="text" name="stock_name">
      <input type="submit" value="OK">
  </form>
  
  {% if test_list %}
      <ul>
      {% for question in test_list %}
          <li>{{question.date}} {{question.close}} {{question.stock_name}}</a></li>
      {% endfor %}
      </ul>
  {% else %}
      <p>해당 종목 데이터가 없습니다.</p>
  {% endif %}
  
  <!-- endblock content % -->
  ```
  

* **`bash`**

  ```bash
  $ docker compose docker-compose.dev.yml up -d --build
  
  $ docker exec backend python manage.py makemigrations charts
  $ docker exec backend python manage.py migrate charts
  # connect to 'http://localhost:8000/'
  
  $ docker compose docker-compose.dev.yml down
  ```



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP



<br>



<!------------------------------------ STEP ------------------------------------>

## STEP

