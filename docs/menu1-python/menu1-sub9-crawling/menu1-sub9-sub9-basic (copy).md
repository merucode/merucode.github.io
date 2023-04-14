---
layout: default
title: Scraping Example
parent: Crawling
grand_parent: Python
nav_order: 1
---

# Scraping Example
{: .no_toc }



### Step. Naver 종목토론실 게시글 Scraping

```python
post_urls = ['https://finance.naver.com' + link['href'] for link in table.select('.title > a')]
post_contents = []
for post_url in post_urls:
    post_html = requests.get(post_url)
    post_soup = BeautifulSoup(post_html.content, 'html.parser')
    post_content = post_soup.find('div', {'class': 'view_se'}).get_text().strip()
    post_contents.append(post_content)
df['content'] = post_contents
```

