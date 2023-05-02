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

* We start from [[Backend]-[Django]-[Deploy-with-Docker]]([Deploy with Docker | Just the Docs Template (merucode.github.io)](https://merucode.github.io/docs/menu4-backend/menu4-sub2-django/menu4-sub2-sub8-deploy-with-docker.html#step-7-final-file-structure))



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
