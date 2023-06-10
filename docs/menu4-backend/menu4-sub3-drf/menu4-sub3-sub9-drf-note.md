---
layout: default
title: DRF Note
parent: DRF
grand_parent: Backend
nav_order: 9
---

# DRF Note
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

## STEP 1. Serializer Field

### Step 1-1. Type of Field

| 항목                    | 구조                                                         | 생성방법                                                     |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `CharField`             | `CharField(max_length=None, min_length=None, allow_blank=False)` | `name = serializers.CharField()`                             |
| `IntegerField`          | `IntegerField(max_value=None, min_value=None)`               | `running_time = serializers.IntegerField()`                  |
| `DateField`             | `DateField(format=api_settings.DATE_FORMAT)`<br>`format="%Y/%m/%d"` 형식으로 적용(기본 포멧 `%Y-%m-%d`) | `opening_date = serializers.DateField()`                     |
| `DateTimeField`         | `DateTimeField(format=api_settings.DATETIME_FORMAT)`<br>더 자세한 날짜·시간 포맷이 궁금하시면 [링크](https://www.w3schools.com/python/python_datetime.asp) | `created = serializers.DateTimeField()`                      |
| `FileField`             | `FileField(max_length=None, allow_empty_file=False, use_url=True)` | `file = serializers.FileField()`                             |
| `ImageField`            | `ImageField(max_length=None, allow_empty_file=False, use_url=True)` | `image = serializers.ImageField(max_length=None, allow_empty_file=False, use_url=True)` |
| `SerializerMethodField` | `SerializerMethodField(method_name=None)`                    | `age = serializer.SerializerMethodField()`                   |

* `SerializerMethodField` : 사용자가 정의한 함수를 통해 직렬화 과정에서 새로운 값을 생성할 수 있는 필드입니다. 데이터 생성 시에는 사용할 수 없는 `read_only` 필드

  * 함수의 이름을 설정하는 `method_name` 옵션이 존재하는데요. 설정하지 않으면 자동으로 함수 이름이 `get_변수명`으로 지정

    ```python
    # 나이는 데이터베이스에 존재하지 않지만, 생일 데이터를 통해 새로운 값을 생성해 준다.
    age = serializer.SerializerMethodField()
    
    def get_age(self, obj):
        return datetime.now().year - obj.birthday.year + 1
    ```



### Step 1-2. Serializer 옵션의 종류

| 옵션         | 내용                                                         | 사용법                                                |
| ------------ | ------------------------------------------------------------ | ----------------------------------------------------- |
| `read_only`  | 데이터를 직렬화할 때 해당 필드를 사용하고, <br>역직렬화할 때는 사용하고 싶지 않을 때 설정 | `id = serializers.IntegerField(read_only=True)`       |
| `write_only` | 데이터를 생성할 때에는 입력해야 하지만, <br>데이터를 조회할 때는 보여주면 안 되는 필드에 사용 | `manager = serializers.CharField(write_only=True)`    |
| `required`   | 필드를 필수적으로 입력해야 하는지 정의해 주는 옵션으로, <br>기본값은 `True`<br>데이터를 입력하지 않는다면 `is_valid()` 실행시 에러 발생 | `created = serializers.DateTimeField(required=False)` |
| `source`     | 어떤 값을 참조할지 정의하는 옵션<br>`Serializer` 기본 정의 필드명을 기준으로 값을 참조<br> 사용하는 필드명과 모델 필드명이 다를 경우 `source` 옵션 사용 |                                                       |

* `required`

  * 자동으로 데이터 생성일이 만들어지는 경우 `required` 옵션을 `False`로 설정하여 데이터 입력 없이도 에러가 발생하지 않도록 처리

    ```python
    # created 필드는 데이터가 생성될 경우 자동으로 추가된다.
    created = models.DateTimeField(auto_now_add=True)
    
    created = serializers.DateTimeField(required=False)
    ```

  * `Serializer`에 `partial` 옵션을 사용하면 `required`가 `True`인 필드를 입력하지 않아도 됩니다. 그래서, 데이터 수정 요청을 처리할 때 모든 값을 넣지 않아도 됐던 것입니다.
  * `read_only=True`는 데이터 생성·수정 요청 시 필드에 입력된 값을 무시하고 역직렬화 미수행
  * `required=False`는 값을 입력하지 않아도 되지만, 만약에 입력한다면 해당 값도 역직렬화 수행

* `source` 

  *  영화 관련 데이터를 전달할 때 이름을 `name` 대신 `movie_name`으로 사용법

    ```python
    # 모델에서 name 이란 필드를 통해 영화의 이름을 받아옴.
    class Movie(models.Model):
        name = models.CharField(max_length=30)
    
    # Serializer에서는 영화의 name 필드를 movie_name 이란 이름으로 사용함.
    class MovieSerializer(serializer.Serializer):
        movie_name = serializers.CharField(source='name')
    
    # Serializer를 통해 생성된 JSON 데이터
    {
      "movie_name": "영화 이름",
      # ...
    }
    ```




<br>



## STEP 2. Model Serializer Meta Option

* **`model`**

  * `model`은 `ModelSerializer` 클래스가 어떤 모델로 시리얼라이저를 생성할지 지정해 주는 필수 옵션입니다. 아래 코드와 같이 `Meta` 클래스 내부에 모델 이름을 작성해 주는 식으로 사용합니다.

    ```python
    class MovieSerializer(serializers.ModelSerializer):
        class Meta:
            model = Movie
    ```

* **`fields`**

  * `ModelSerializer`에서 어떤 필드를 사용할지 선언하는 옵션입니다. 아래와 같이 필요한 필드들의 이름을  `fields`에 작성하면 됩니다. 필드의 타입은 모델 필드의 타입을 보고 자동으로 유추합니다.

    ```python
    # Movie 모델
    class Movie(models.Model):
        name = models.CharField(max_length=30)
        opening_date = models.DateField()
        running_time = models.IntegerField()
        overview = models.TextField()
        
    class MovieSerializer(serializers.ModelSerializer):
        class Meta:
            model = Movie
            fields = ['id', 'name', 'opening_date', 'running_time', 'overview']
    ```

  * `fields`를 `__all__`로 해주면 모델에 존재하는 모든 필드를 사용할 수 있습니다.

    ```python
    class MovieSerializer(serializers.ModelSerializer):
        class Meta:
            model = Movie
            fields = '__all__'
    ```

  * `fields`는 조건부 필수 옵션으로, `exclude`를 사용하지 않으면 필수로 입력해야 합니다.

* **`exclude`**

  * `fields`와 정반대인 옵션입니다. 모델을 기준으로 어떤 필드를 제외할지 나타냅니다. 모델에 총 5개의 필드가 있는데 시리얼라이저에서는 4개의 필드만 사용하고 싶다면 `exclude`로 하나의 필드를 제외할 수 있습니다.

  * 아래의 두 `MovieSerializer`는 동일한 기능을 합니다.

    ```python
    # id, name, opening_date, running_time 총 4개의 필드를 fields 옵션을 사용해 직렬화 시키는 방법
    class MovieSerializer(serializers.ModelSerializer):
        class Meta:
            model = Movie
            fields = ['id', 'name', 'opening_date', 'running_time']
    
    # id, name, opening_date, running_time 총 4개의 필드를 exclude 옵션을 사용해 직렬화 시키는 방법
    class MovieSerializer(serializers.ModelSerializer):
        class Meta:
            model = Movie
            exclude = ['overview']
    ```

  * `exclude`는 조건부 필수 옵션으로, 위에서 설명한 `fields`를 사용하지 않으면 필수로 입력해야 합니다.

* **`read_only_fields`**

  * 이전 레슨에서 모델 시리얼라이저는 `id` 필드와 같이 데이터베이스에서 기본으로 생성되는 필드에 `read_only` 옵션을 자동으로 추가해 준다고 했었죠? 하지만, 만약 그 외 필드(자동 생성되지 않는 필드)에 선택적으로 `read_only`를 추가하려면 `read_only_fields` 옵션을 사용하면 됩니다.

    ```python
    # read_only_fields를 사용한다면 필드명 작성을 통해 read_only 옵션을 추가할 수 있음.
    class MovieSerializer(serializers.ModelSerializer):
        class Meta:
            model = Movie
            fields = ['id', 'name', 'opening_date', 'running_time', 'overview']
            read_only_fields = ['name']
    ```

* **`extra_kwargs`**

  * 다양한 필드에 여러 옵션을 추가해야 할 경우 `extra_kwargs`를 사용합니다. `extra_kwargs`를 사용하면 필드를 직접 정의하지 않아도 된다는 `ModelSerializer`의 장점을 극대화할 수 있습니다.

  * 만약 영화 모델의 줄거리(`overview`)를 생성 시에만 사용하고 싶다면, 아래 코드와 같이 `extra_kwargs`를 사용하여 `write_only` 옵션을 추가할 수 있습니다.

    ```python
    # extra_kwargs를 사용한다면 간단하게 특정한 필드에 옵션을 추가할 수 있음.
    class MovieSerializer(serializers.ModelSerializer):
        class Meta:
            model = Movie
            fields = ['id', 'name', 'opening_date', 'running_time', 'overview']
            extra_kwargs = {
                'overview': {'write_only': True},
            }
    ```

  * 참고로 꼭 `read_only_fields`나 `extra_kwargs` 같은 옵션을 사용하지 않고 필드를 직접 정의해도 됩니다.

    ```python
    class MovieSerializer(serializers.ModelSerializer):
        name = serializers.CharField(read_only=True)
        overview = serializers.CharField(write_only=True)
    
        class Meta:
            model = Movie
            fields = ['id', 'name', 'opening_date', 'running_time', 'overview']
    ```



<br>



## STEP 3.

