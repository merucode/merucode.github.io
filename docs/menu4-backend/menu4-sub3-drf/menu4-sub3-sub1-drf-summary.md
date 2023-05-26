---
layout: default
title: DRF Summary
parent: DRF
grand_parent: Backend
nav_order: 9
---

# DRF Summary
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

## STEP 1. DRF Basic

* **Model Serializer**

  * `movies/serializers.py`

    ```python
    class MovieSerializer(serializers.ModelSerializer):
        class Meta:
            model = Movie
            fields = ['id', 'name', 'opening_date', 'running_time', 'overview']
    ```

    



