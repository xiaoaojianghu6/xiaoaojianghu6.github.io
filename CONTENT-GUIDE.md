# 网站内容修改指南

> 改完任何内容文件后，运行一次 `python3 build.py`，再把改动提交（git add -A && git commit && git push）即可发布。
> 所有内容都在 `content/` 和 `projects/` 里，**不需要碰生成的 HTML**。

## 一、速查表：我想改 X，去哪个文件？

| 我想改… | 打开这个文件 |
|---|---|
| 首页大标题、英文简介、视差画廊图片 | `content/home.yaml` |
| 首页行者区古文引用 / 背水诗 / 赤子区五张卡片 | `content/home.yaml` |
| 首页作品卡片的顺序、要不要显示 | `projects/<项目名>/project.yaml` 里的 `order` / `home` |
| 行者区（地点、文字、照片、增删地点） | `content/wanderer.yaml` |
| 诗人区（诗卡、诗的全文、长诗） | `content/poems.yaml` |
| 赤子区（电影/音乐/书籍/美食/运动） | `content/enthusiast.yaml` |
| 述己（简介、轮播图、时间线、微信/邮箱） | `content/about.yaml` |
| 导航、全屏菜单、页脚跑马灯、联系方式 | `content/site.yaml` |
| 创客区项目（增删改，见下文第三节） | `projects/<项目名>/project.yaml` |
| 菜单/导航的文字 | `content/site.yaml` |

## 二、各文件怎么改

所有文件都是 YAML 格式，规则：

- `#` 开头是注释，不影响内容
- 列表项用 `- ` 开头，一层缩进 2 个空格（**不要用 Tab**）
- 文字里有冒号 `:` 时，给整句加英文引号：`title: "我的标题: 副标题"`
- 换行效果：行者区引用/长诗里写 `<br>` 即可

### 加一张照片 / 换一张照片
1. 把图片放进对应目录（行者区照片放 `top/wanderer/`，赤子区海报放 `top/enthusiast/`，项目图放 `projects/<项目名>/media/`）
2. 建议文件名用英文小写+短横线（如 `wutai-mountain.jpg`），避免中文文件名
3. 在 yaml 里把路径改成 `/top/wanderer/xxx.jpg`（以 `/` 开头）
4. `python3 build.py`，完成

> 照片很大的话先压一下（预览图 2000px 宽足够）。

## 三、创客区：添加一个新项目（重点）

一个项目 = 一个文件夹，全部内容自包含：

```
projects/
  my-new-project/          ← 复制 projects/_template/ 改成这个英文名
    project.yaml           ← 项目所有文字数据
    media/                 ← 这个项目的所有图片/视频/PDF
      cover.jpg
      demo.mp4
```

步骤：

1. **复制模板**：`cp -r projects/_template projects/my-new-project`
2. **丢素材**：把封面图、演示视频、截图等放进 `media/`
3. **改 `project.yaml`**：
   - `slug`：和文件夹名一致
   - `order`：排序位置（1 = 最前面，列表/首页/详情页"下一项目"全部自动按它接线，**不用再手动改下一项目链接**）
   - `title / subtitle / role / date / year`：标题和元信息
   - `categories`：分类（用于列表页筛选，如 `INTERNSHIP`）
   - `tags`：标签（首页卡片和列表页显示）
   - `hero`：封面图路径（详情页头图），`list_image`：列表页卡片图（一般和 hero 相同）
   - `overview`：一段话简介（数组，每项一段）
   - `home: true`：是否显示在首页（false 可只出现在 /builder 列表）
   - `sections`：详情页正文，从上到下依次渲染，支持 5 种类型：

```yaml
sections:
  - type: text              # 文字区块，tag 是小标题
    tag: 核心优势
    paragraphs:
      - 第一段…
      - 第二段…
  - type: image             # 单张整宽图片
    src: /projects/my-new-project/media/图1.jpg
    alt: 图片说明
  - type: video             # 视频（自动播放循环，窗口样式与其他项目统一）
    src: /projects/my-new-project/media/demo.mp4
    caption: 视频说明
  - type: image_panel       # 带标签的图组（可单张或多张对比）
    images:
      - { src: /projects/my-new-project/media/a.jpg, alt: A图, label: A, caption: 说明 }
  - type: pdf               # PDF 阅读器窗口
    src: /projects/my-new-project/media/paper.pdf
    title: 论文标题
```

4. **生成并发布**：`python3 build.py` → 首页、/builder 列表、详情页、"下一项目"链接、sitemap 全部自动更新，一处都不用手动同步

## 四、本地预览

```
python3 build.py
python3 -m http.server 8000
# 浏览器打开 http://localhost:8000
```

## 五、目录结构说明

```
content/          所有页面内容（改这里）
projects/         创客区项目（一项目一文件夹，_template 是模板）
templates/        页头/菜单/页脚骨架 + 各页自定义样式（一般不用动）
build.py          生成器：content+projects → HTML
  ├─ index.html 等生成的页面（自动产出，别手改）
_astro/           编译好的 CSS/JS（全站共用，别动）
top/  about/  common/  fonts/   图片、字体等静态资源
tools/            一次性迁移脚本（历史参考，已不再使用）
```

## 六、一些说明

- 生成出来的 `*.html` 是产物，手改会在下次 build 时被覆盖
- build.py 会在生成前做校验：yaml 里有引用不存在的图片、项目缺必填字段都会报错并停止
- `新项目/`、`微信.JPG`、`_archive/` 在 .gitignore 里，不会发布
- 如果以后想零本地操作（只改文件就自动发布），可以在仓库 Settings → Pages 里把来源切到 GitHub Actions，再配一个跑 `python3 build.py` 的 workflow——目前未启用，保持"push 即部署"
