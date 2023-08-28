---
layout: default
title: Object Oriented
parent: Python Basic
grand_parent: Python
nav_order: 81
---

# Object Oriented
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
## Step 1. 객체란?
### Step 1-1. 객체
* **객체**(object) : **속성**과 **행동**으로 이루어진 존재
	* *ex> 자동차 객체*
		* *속성 : 높이, 의자갯수, 색깔*
		* *행동 : 시동, 전진, 후진*
	 * *ex> 인스타그램 유저 객체*
		* *속성 : 이메일, 비밀번호, 친구목록*
		* *행동 : 팔로우*

* **객체지향 프로그래밍** :  프로그램을 여러 개의 독립된 **객체들과 그 객체들 간의 상호작용**으로 파악
	* *ex> 총게임*
		* *케릭터 객체*
			* *속성 : ID, 사용중인 총, 체력, 목숨*
			* *행동 :  총 발사, 달리기, 죽음(체력 0)*
		* *총 객체*
			*  *속성 : 모델명, 무게, 장전된 총알 갯수*
			*  *행동 : 총알 발사*
		* *총알 개체*
			* *속성 : 공격력*
			* *행동 : 총알 맞은 캐릭터 체력을 공격력만큼 감소시킴*
		* *객체 소통*
			* *케릭터 객체 총 발사 신호 → 총 객체* 
			* *총 객체 총알 발사 신호 → 총알 객체*
			* *총알 객체 케릭터 체력 감소 신호 → 케릭터 객체*
			
<br>
			
### Step 1-2. 객체지향 프로그래밍으로 프로그램 만들기
* 프로그램에 **어떤 객체들이 필요**한지 정한다
* **객체들의 속성과 행동**을 정한다
* **객체들이 서로 어떻게 소통**할지 정한다

<br><br>

## Step 2. 객체 만드는 법

### Step 2-1. 클래스와 인스턴스
* **객체 = 인스턴스**
* 클래스(틀)로 인스턴스(붕어빵)를 만든다
	```python
	class User:
		pass
	user1 = User() # 서로 다른 존재
	user2 = User() # 서로 다른 존재
  ```
* **객체 속성 → 인스턴스 변수**
* **객체 행동 → 인스턴스 메소드(함수)**
<br>

### Step 2-2. 인스턴스 변수(객체 속성)
* **인스턴스 변수 정의하기**
* **`인스턴스이름.속성이름 = 속성에 넣을 값`**
	```python
	user1.name = "김대위"
	user1.email = "captain@naver.com"
	```
* **인스턴스 변수 사용하기**
*	**`인스턴스이름.속성이름`**
	```python
	print(user1.name)
	```
<br>

### Step 2-3. 인스턴스 메소드(객체 행동)
* **[참고] 메소드의 종류** : ①인스턴스 메소드 ②클래스 메소드 ③정적 메소드

* **인스턴스 메소드** : 인스턴스 변수를 사용하거나, 인스턴스 변수에 값을 설정하는 메소드

* **인스턴스 메소드 정의하기**
	```python
	class User:
		def say_hello(some_user):
			print(f"안녕하세요. 저는{some_user.name}입니다")
	```
* **인스턴스 메소드 사용하기**
*	**`클래스이름.메소드이름(인스턴스)`**
*	**`인스턴스이름.메소드이름()`**
	```python
	User.say_hello(user1)	
	user1.say_hello()
	```
	※ **`인스턴스이름.메소드이름()`** 사용 시 **해당 인스턴스가 첫번째 파라미터로 자동 전달** 
	→ 파이썬 인스턴스 메소드 첫 번째 파라미터 이름을 **self**로 사용하도록 권장	
	```python
	class User:
		def say_hello(self):
			print(f"안녕하세요. 저는{self.name}입니다")
	```

* **인스턴스 변스와 같은 이름을 갖는 파라미터** : 문제 없음(일반적)
	```python
	class User:
		def check_name(self, name):
			return self.name == name
	```
	
<br><br>

### Step 2-4. 특수 메소드
* **특수 메소드** : 특정 상황에서 자동으로 호출되는 메소드

* **`__init__`** : 인스턴스 생성 시 자동으로 호출(변수 초기값 설정)

* **`__str__`** :  `print` 함수 호출 시 자동 호출(`__str__` return값 출력)
	```python
	class User:
		def __init__(self, name, eamil):
			self.name = name
			self.email = email
			
		def __str__(self):
			return f"사용자:{self.name}, 이메일:{self.email}"
	
	user1 = User("Young", "younh@naver.com")
	# 1) user1 인스턴스 생성 
    # 2) __init__ 메소드 자동 호출 
    print(user1) 
    # __str__ 메소드 return값 출력
    ```
<br>

### Step 2-5. 클래스 변수
* **인스턴스 자신만의 속성 → 인스턴스 변수**
* **여러 인턴스 공유 속성 → 클래스 변수**
	* *ex> 클래스의 인스턴스 호출 횟수*

* **클래스 변수 정의하기**
	```python
	class User:
		count = 0
		
	User.count = 1
	```
* **클래스 변수 사용하기**
*	**`클래스이름.속성이름`**
	```python
	class  User:
		count =  0
		
		def __init__(self):
			User.count += 1
			
	print(User.count)
	```
* **같은 이름의 클래스/인스턴스 변수** : 인스턴스 변수 우선 → **클래스 변수 설정 시 클래스 이름으로만!**
	```python
	class User:
		count =  0
	
	user1 = User()
	user2 = User()
	user1.count = 5
	User.count = 3
		
	print(user1.count)	# 5 출력(인스턴스 변수)
	print(user2.count) 	# 3 출력(클래스 변수)
	```
<br>

### Step 2-6. 클래스 메소드
* **데코레이터**(decorator)
	``` python
	def print_hello():
		print("안녕하세요!")
	
	def add_print_to(original):
		def wrapper():
			print("start")		# 부가기능 original(print_hello)를 꾸며줌
			original()
			print("end")		# 부가기능
		return wrapper
	
	add_print_to(print_hello)	# 해당 명령어만으로는 함수만 리턴하기 때문에 출력 안됨
	print_hello = add_print_to(print_hello)
	print_hello()
	```
	**or use "@"**
	```python
	def add_print_to(original):
		def wrapper():
			print("start")		# 부가기능 					
			original()
			print("end")		# 부가기능
		return wrapper
	
	@add_print_to
	def print_hello():
		print("안녕하세요!")

	print_hello()
	```
	
* **클레스 메소드 → 클래스 변수 값을 읽거나 설정** (cf. 인스턴스 메소드 → 인스턴스 변수 값을 읽거나 설정)

	 ```python
	 class User:
		 count = 0
		 
		 def __init__(self):
			 User.count += 1
		 
		 @classmethod
		 def number_of_users(cls): # self 대신 cls 사용
			 print(f"총 유저 수 : {cls.count}")	

	user1 = User()
	user2 = User()

	User.number_of_users()
	user1.number_of_users()
	```
  
* **클래스 메소드 사용**
	```python
	User.number_of_users()
	user1.number_of_users()
	# 두 경우 모두 첫번째 파라미터(cls) 자동 전달
	# 클래스 메소드 데코레이터 사용 덕분

	### 인스턴스 메소드 사용 비교
	User.say_hello(user1)
	user1.say_hello()	
	# 인스턴스에서 메소드 사용 시에만 
	# 첫번째 파라미터(self) 자동 전달
	```	
* **클래스 변수**와 **인스턴수 변수** 모두 사용 →	**인스턴스 메소드 사용**
  *	**클래스 메소드는 인스턴스 변수 사용 불가**
  
	```python
	class User:
		count = 0
		
		def number_of_users(self):
			print(f"총 유저 수: {User.count}")
			# cls 변수 없이 클래스 변수 사용법
	```

<br>

### Step 2-7. 정적 메소드 
* **정적 메소드** : 인스턴스 변수(self), 클래스 변수(cls)를 전혀 다루지 않는 메소드
	```python
	class  User: 
		@staticmethod  
		def  is_valid_email(email_address):
			return  "@"  in email_address
	
	user1 = User()
	```
* **정적메소드 사용**
	```python
	print(User.is_valid_email("taehosung@codeit.kr"))
	print(user1.is_valid_email("taehosung"))
	```
* **어떤 속성을 다루지 않고, 단지 기능(행동)적인 역할**만 하는 메소드를 정의할 때 정적 메소드로 정의

<br><br>

## Step 3. 미리 알아야 할 것 들
### Step 3-1. python
* **python** : 순수 객체 지향 언어 → **모든 것이 객체**
* python의 int, str, list ,dict 등 python의 기본 class에서 가져다 사용하는 인스턴스(객체)

<br>

### Step 3-2. 가변 vs 불변 Type
* 가변 객체 : 한번 생성한 인스턴스 속성 변경 가능
	* *ex> list*
* 불변 객체 : 한번 생성한 인스턴스 속성 변경 불가
	* *ex> tuple*
	```python
	object1 = [1, 2,3]
	object2 = (1, 2, 3)
	object1[0] = 4 	# 가능
	object2[0] = 4 	# 에러 발생
	object2 = (4, 2, 3) # 가능(새로운 인스턴스 지정)
	```
<br><br>

## Step 4. 객체 프로그래밍 예제
*  속성
	 * 시계는 현재 시간을 속성으로 가집니다. `Counter` 클래스를 사용해서 시, 분, 초를 나타낼 수 있습니다.
		-   초: 1부터 59까지 셀 줄 아는  `Counter`  클래스의 인스턴스
		-   분: 1부터 59까지 셀 줄 아는  `Counter`  클래스의 인스턴스
		-   시: 1부터 23까지 셀 줄 아는  `Counter`  클래스의 인스턴스
* 행동
	-   1초 증가시키기
	    -   시간을 1초씩 증가시킵니다.
	    -   이때 주의할 점은 시간을 증가시킬 때 59초가 60초가 되면 초를 다시 00초로 바꾼 후에 분을 1분 증가시키고, 59분이 60분이 되면 분을 다시 00분으로 바꾼 후에 시를 1시간 증가시키는 것입니다. 이것은 당연한 시간의 원리이니 따로 설명하지 않겠습니다. 이 부분을 구현할 때  `Counter`  클래스의  `tick`  메소드의 리턴값(`True`  또는  `False`)이 어떻게 활용될지 생각해 보세요.
	-   값 변경하기: 이미  `Counter`  클래스에는 값을 설정하는 메소드가 있습니다. 시계 클래스에서 시간을 설정할 때 시, 분, 초를 각각 따로 설정하는 건 귀찮겠죠? 시, 분, 초의 값을 한번에 설정하는 메소드를 만듭시다.
* 이러한 속성과 행동을 가지는 `Clock` 클래스를 정의해 보세요!

<br>

* **모범 답안**

	```python
	class Counter:
    """시계 클래스의 시,분,초를 각각 나타내는데 사용될 카운터 클래스"""

	    def __init__(self, limit):
	        """인스턴스 변수 limit(최댓값), value(현재까지 카운트한 값)을 설정한다.
	        인스턴스를 생성할 때 인스턴스 변수 limit만 파라미터로 받고, value는 초깃값 0으로 설정한다."""    
	        self.limit = limit
	        self.value = 0

	    def set(self, new_value):
	        """파라미터가 0 이상, 최댓값 미만이면 value에 설정한다. 아닐 경우 value에 0을 설정한다."""
	        self.value = new_value if 0 <= new_value < self.limit else 0

	    def tick(self):
	        """value를 1 증가시킨다. 카운터의 값 value가 limit에 도달하면 value를 0으로 바꾼 뒤 
	        True를 리턴한다. value가 limit보다 작은 경우 False를 리턴한다."""
	        self.value += 1

	        if self.value == self.limit:
	            self.value = 0
	            return True
	        return False

	class  Clock: 
		""" 시계 클래스 """ 
		HOURS = 24  # 시 최댓값 
		MINUTES = 60  # 분 최댓값 
		SECONDS = 60  # 초 최댓값  
		
		def  __init__(self, hour, minute, second): 
		""" 각각 시, 분, 초를 나타내는 카운터 인스턴스 3개(hour, minute, second)를 정의한다.
		현재 시간을 파라미터 hour시, minute분, second초로 지정한다. """ 
			self.hour = Counter(Clock.HOURS) 
			self.minute = Counter(Clock.MINUTES) 
			self.second = Counter(Clock.SECONDS) 
			self.set(hour, minute, second) 
		
		def  set(self, hour, minute, second): 
		"""현재 시간을 파라미터 hour시, minute분, second초로 설정한다.""" 
			self.hour.set(hour) 
			self.minute.set(minute) 
			self.second.set(second) 
			
		def  tick(self): 
		""" 초 카운터의 값을 1만큼 증가시킨다. 초 카운터를 증가시킴으로써, 
		분 또는 시가 바뀌어야하는 경우를 처리한다. """ 
			if self.second.tick(): 
				if self.minute.tick(): 
					self.hour.tick() 
					
		def  __str__(self): 
		""" 현재 시간을 시:분:초 형식으로 리턴한다. 시, 분, 초는 두 자리 형식이다. 
		예시: "03:11:02" """  
			return  "{}:{}:{}".format(self.hour, self.minute, self.second)
```
