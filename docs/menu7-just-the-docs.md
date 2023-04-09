---
layout: default
title: Just-the-docs
nav_order: 7
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

## STEP 1. Label

```
Default label
{: .label }
```

Default label
{: .label }



```
Callouts
{: .d-inline-block }

New
{: .label .label-green } # Can use .label-yellow, .label-red
```

Callouts
{: .d-inline-block }

New
{: .label .label-green }

<br>
<br>

## STEP 2. Callouts

```
{: .highlight } # Can use .ighlight, .important, .new, .note, and .warning.
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
<br>

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