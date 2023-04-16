---
layout: default
title: Just-the-docs
nav_order: 10
---
# Just-the-docs
{: .no_toc }
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>

<!---------------------------------- STEP 1 ---------------------------------->
## STEP 1. Label

* `.label`, `.label .label-green`, `.label .label-red`, `.label .label-yellow`

```
Default label
{: .label }
```

Default label
{: .label }


```
Label
{: .d-inline-block }

New
{: .label .label-green } # Can use .label-yellow, .label-red
```

Label
{: .d-inline-block }

New
{: .label .label-green }

<br>

<!---------------------------------- STEP 2 ---------------------------------->
## STEP 2. Callouts
* `.highlight`, `.important`, `.new`, `.note`, `.warning`

```
{: .highlight } # Can use 
A paragraph
```

{: .highlight }
A paragraph


```
{: .note-title }
> My note title
>
> A paragraph with a custom title callout
```

{: .note-title }
> My note title
>
> A paragraph with a custom title callout

<br>

<!---------------------------------- STEP 3 ---------------------------------->
## STEP 3. Page TOC

```html
<!-- write below code at top of page.md -->
# Title
{: .no_toc }
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>
```

<br>

<!---------------------------------- STEP 3 ---------------------------------->
## STEP 4. [Mermaid]

```
```mermaid
flowchart TB
    c1-->a2
    subgraph one
    a1[VM]--text-->a2[VM2]
    end
    subgraph two
    b1-->b2
    end
    subgraph three
    c1-->c2
    end
    one --ex--> two
    three --> two
    two --> c2
```mermaid
```

```mermaid
flowchart TB
    c1-->a2
    subgraph one
    a1[VM]--text-->a2[VM2]
    end
    subgraph two
    b1-->b2
    end
    subgraph three
    c1-->c2
    end
    one --ex--> two
    three --> two
    two --> c2
```

<br>

<!---------------------------------- STEP  ------------------------------------>
## STEP 5. Related Site

### Step 5-1. Markdown

* [Easiest way of writing mathematical equation in R Markdown]
* [Markdown 수식작성법]
* [Markdown 행렬]

---
[Mermaid]: https://mermaid.js.org/syntax/flowchart.html
[Easiest way of writing mathematical equation in R Markdown]: https://www.youtube.com/watch?v=4I3PCDME5U8
[Markdown 수식작성법]: https://velog.io/@d2h10s/LaTex-Markdown-%EC%88%98%EC%8B%9D-%EC%9E%91%EC%84%B1%EB%B2%95
[Markdown 행렬]: https://cheris8.github.io/etc/MD-LaTex/