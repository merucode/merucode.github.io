---
layout: default
title: Summary
parent: NLP
grand_parent: Data Science
nav_order: 1
---


```python

### Order by frequency 
vocab = Counter(tokens)
vocab = vocab.most_common() # return tuple in list [(word1,count1), (word2, count2)...]

for (word, frequency) in vocab: # use result at for
    i = i + 1
    word_to_idx[word] = i

```