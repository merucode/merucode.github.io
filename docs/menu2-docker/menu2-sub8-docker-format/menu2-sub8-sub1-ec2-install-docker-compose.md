---
layout: default
title: EC2 Docker Install
parent: Docker Format
grand_parent: Docker
nav_order: 1
---

# EC2 Docker Install
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

### Step 1. 인스턴스 docker/docker compose install

* docker-desktop은 docker-engine과 windows 혹은 mac을 연결해주는 프로그램

  * 서버는 ubuntu만 이용하므로 docker-engine으로 충분

* **`instance`-`bash`**(**docker/docker compose 설치**)

  ```bash
  ### Set up the repository
  #1. Update the apt package index and install packages to allow apt to use a repository over HTTPS:
  sudo apt-get update
  sudo apt-get install ca-certificates curl gnupg
  
  #2. Add Docker’s official GPG key:
  sudo mkdir -m 0755 -p /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  
  #3. Use the following command to set up the repository:
  echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" |  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  
  ### Install Docker Engine
  #1. Update the apt package index:
  sudo apt-get update
  
  #2. Install Docker Engine, containerd, and Docker Compose.
  sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  
  #3. docker install check :  --rm명령어를 주어 컨테이너 실행 후 바로 삭제되도록 한다.
  sudo docker run --rm hello-world
  
  ### Install Docker-compose
  #1. Update the package index, and install the latest version of Docker Compose:
  sudo apt-get update
  sudo apt-get install docker-compose-plugin
  
  #2. Verify that Docker Compose is installed correctly by checking the version.
  docker compose version
  ```

  - [docker install(linux)](https://docs.docker.com/desktop/install/ubuntu/)
  - [docker compose install(linux)](https://docs.docker.com/compose/install/linux/#install-using-the-repository)


  * **`instance`-`bash`(sudo 없이 사용 가능하게 docker 권한 부여)** 

  ```bash
  $ sudo usermod -aG docker $USER
  # OR
  $ sudo usermod -aG docker $(whoami)
  
  # console 종료 후 재연결
  ```