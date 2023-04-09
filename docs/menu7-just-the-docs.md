---
layout: default
title: just-the-docs
nav_order: 7
---
# just-the-docs
{: .no_toc }
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>


[Link button](https://just-the-docs.github.io/just-the-docs/docs/ui-components){: .btn .btn-green }

## STEP 1. Label

```
Default label
{: .label }
```

Default label
{: .label }

<br>

```
New
{: .label .label-green } # Can use .label-yellow, .label-red
```

New
{: .label .label-green }

<br>
<br>

## STEP 2. Callout

```
{: .highlight } # Can use .ighlight, .important, .new, .note, and .warning.
A paragraph
```

{: .highlight }
A paragraph

<br>

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
<br>

## STEP 3. Page TOC

```
### write below code at top of page
# Title
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>
```