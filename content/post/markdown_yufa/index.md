---
title: Markdown 基础语法
summary: just basic
date: '2025-03-11'
authors:
  - william
tags:
 - basic knowledge and skills
---



---
    这是分割线
    # 标题
    ## 副标题
    *斜体*
    **粗体**
    - 无序标题
      - 次
    1. 第一
    2. 第二

---
这是分割线
# 标题
## 副标题
*斜体*
**粗体**
- 无序标题
  - 次
1. 第一
2. 第二

---

    `shift`+`option`+`~` or `~` in english board means code or 强调

`shift`+`option`+`~` or `~` in english board means code or 强调

    [william's blog的链接](https://xiaoaojianghu6.github.io)

[william's blog的链接](https://xiaoaojianghu6.github.io)

    [标题](#标题 "Goto 标题")
    这是文内索引

[标题](#标题 "Goto 标题")
这是文内索引

---

    \#
    让一个符号变成普通符号
    > 引用

\#
让一个符号变成普通符号
> 引用

    ---
    - [ ] An uncompleted task
    - [x] A completed task

---
- [ ] An uncompleted task
- [x] A completed task
      

---
    这是表格
    First Header  | Second Header
    ------------- | -------------
    Content Cell  | Content Cell
    Content Cell  | Content Cell

---
First Header  | Second Header
------------- | -------------
Content Cell  | Content Cell
Content Cell  | Content Cell

---

`control`+`command`+` `    for emoji

like this:🥮  🎥  🎙️...

---

图片插入

✅ 方法 1：拖拽上传（适用于 VS Code、Typora、Notion）

    ![描述文本](./图片文件名.png)

✅ 方法 2：Markdown 语法 + 本地路径

    ![描述文本](./images/example.png)

✅ 方法 3：Jupyter Notebook 本地图片

    用 HTML：
    <img src="your_image.png" width="300">

    或者：
    ![描述文本](attachment:your_image.png)

但`attachment:`语法需要手动将图片拖入Jupyter，并确保文件路径正确

---
