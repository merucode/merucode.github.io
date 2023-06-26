---
layout: default
title: fastapi-react-postgresql-nginx
parent: Docker Format
grand_parent: Docker
nav_order: 11
---

# fastapi-react-postgresql-nginx
{: .no_toc .d-inline-block }
ing
{: .label .label-green }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>
<!------------------------------------ STEP ------------------------------------>

## STEP 0. Reference Site

* [Github]

<br>

## Step 1. Basic Library Install

### Step 1-1. File Structure

* **File structure**

  ```bash
  .
  ├── 📁backend
  │   ├── 📄Dockerfile
  │   ├── 📄main.py
  │   └── 📄requirements.txt
  ├── 📁database
  │   ├── 📄Dockerfile
  │   └── 📁postgresql
  ├── 📄docker-compose.yml
  ├── 📁frontend
  │   ├── 📄Dockerfile
  │   ├── 📁node_modules
  │   ├── 📄package-lock.json
  │   ├── 📄package.json
  │   ├── 📁public
  │   └── 📁src
  └── 📁nginx
      ├── 📄Dockerfile
      └── 📄nginx.conf
  ```

### Step 1-2. Install Basic Docker format
* [react](https://merucode.github.io/docs/menu2-docker/menu2-sub8-docker-format/menu2-sub8-sub3-react.html)
* [fastapi](https://merucode.github.io/docs/menu2-docker/menu2-sub8-docker-format/menu2-sub8-sub4-fastapi.html)
* [postgresql](https://merucode.github.io/docs/menu2-docker/menu2-sub8-docker-format/menu2-sub8-sub7-postgresql.html)


<br>

### Step 1-2. Install Basic Docker format
* [react](https://merucode.github.io/docs/menu2-docker/menu2-sub8-docker-format/menu2-sub8-sub3-react.html)
* [fastapi](https://merucode.github.io/docs/menu2-docker/menu2-sub8-docker-format/menu2-sub8-sub4-fastapi.html)
* [postgresql](https://merucode.github.io/docs/menu2-docker/menu2-sub8-docker-format/menu2-sub8-sub7-postgresql.html)

<br>

## STEP 2. Connect using Nginx

### Step 2-1. Nginx Docker Code

* `docker-compose.yml`
	```dockerfile
	...
	  nginx:
	    container_name: nginx
	    build:
	      context: ./nginx/
	      dockerfile: Dockerfile
	    ports:
	      - 80:80
	    depends_on:
	      - backend
	      - frontend
	```

* `nginx/Dockerfile`

	```dockerfile
	FROM nginx:1.25-alpine

	# Delete conf and copy
	RUN rm /etc/nginx/conf.d/default.conf
	COPY nginx.conf /etc/nginx/conf.d/
	```

* `nginx/nginx.conf`

	```nginx
	# frontend : 3000 port
	upstream frontend {
	    server frontend:3000;
	}

	# backend : 8000 port
	upstream backend {
	    server backend:8000;
	}

	server {
	    listen 80;

	    # When request in "/", connect to "http://frontend" 
	    location / {
	        proxy_pass http://frontend;
	        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
	        proxy_set_header Host $host;
	        proxy_redirect off;
	    }

	    # When request in "/api", connect to "http://backend" 
	    # Using rewrite and fastapi root_path, handling 'api/' as '/' in backend(fastapi)
	    location /api {
	    	rewrite ^/api/(.*)$ /$1 break;
	        proxy_pass http://backend/api;
	        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
	        proxy_set_header Host $host;
	        proxy_redirect off;
	    }
	}
	```

### Step 2-2. Connect Nginx

* `docker-compose.yml`

	```dockerfile
	version: '3.8'

	services:
	  frontend:
	    container_name: frontend
	    build:
	      context: ./frontend/
	      dockerfile: Dockerfile
	    volumes:
	      - ${PWD}/frontend/:/usr/src/app/
	    expose:
	      - 3000
	    command: sh -c "npm install && npm run start"

	  backend:
	    container_name: backend
	    build:
	      context: ./backend/
	      dockerfile: Dockerfile
	    volumes:
	      - ${PWD}/backend/:/usr/src/app
	    expose:
	      - 8000
	    command: ["uvicorn", "main:app","--root-path", "/api", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--reload"]

	  database:
	    container_name: database
	    build:
	      context: ./database/
	      dockerfile: Dockerfile
	    volumes:
	      - ${PWD}/database/postgresql:/var/lib/postgresql/data/
	    expose:
	      - 5432
	    env_file:
	      - .database.env

	  nginx:
	  ...
	```

	* ports → expose 변경(nginx에서 분배)
	* backend command 변경
		* `—proxy-header` : nginx 에서 프록시 패스로 해당 어플리케이션을 연결하고 싶을 때 추가해주는 옵션
		* `—root-path` : api uri의 prefix 를 설정함
 * connect to `http://[EC2 Public IP]`, `http://[EC2 Public IP]/api/`

<br>

## STEP 3. Connect Backend and Frontend


### Step 3-1. Frontend

* **File Structure**

	```bash
	📁frontend
	├── 📄Dockerfile
	├── 📁node_modules
	├── 📄package-lock.json
	├── 📄package.json
	├── 📁public
	└── 📁src
		├── 📄Main.js
		├── 📄index.js
		├── 📁components
		│   ├── 📄App.js
		│   └── 📄Header.jsx
		└── 📁pages
		    └── 📄HomePage
	```

* `bash`
	*	Install `axios`, `react-router-dom@6`
	```bash
	$ docker compose up -d --build
	$ docker exec -it frontend /bin/sh
	# npm install axios react-router-dom@6 --save
	```
* Make as [React-router-basic-form]()

* `pages/HomePage/HomePage.jsx`

	```jsx
	import axios from "axios";
	import { useEffect, useState } from 'react';
	
	function HomePage() {
		const [update,setUpdate] = useState('');
		let message;

		useEffect(() => {
	    axios.get('https://temanet.co.kr/api/hello')
	      .then((res) => {
	        message = res.data.message
	        setUpdate(message)
	      })
		  }, [update]);

	    return (
	    <div>
		    <h1>HomePage</h1>
		    <div>From Backend Data : </div>
		    <div>{update}</div>
		</div>
	    );
	}

	export default HomePage;
	```

### Step 3-2. Backend

