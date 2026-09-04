# xiaoaojianghu6.github.io

William Liu (HAO-Z) 的个人网站，纯静态 HTML，部署于 GitHub Pages（push 即发布）。

## 改内容 / 加项目

**不要直接改 HTML。** 所有内容都在数据文件里，改完运行 `python3 build.py` 重新生成即可：

- `content/*.yaml` —— 各页面文字与图片
- `projects/<项目名>/` —— 创客区项目（一个项目一个文件夹，`_template` 是新项目模板）

详细说明见 **[CONTENT-GUIDE.md](CONTENT-GUIDE.md)**（改哪里、怎么加项目、怎么预览）。

## 目录结构

```
content/     页面内容数据（YAML，带中文注释）
projects/    创客区项目数据 + 各项目媒体文件
templates/   页头/菜单/页脚骨架与各页自定义样式
build.py     生成器（Python 3 + PyYAML，无其他依赖）
_astro/      编译好的全站 CSS/JS
top/ about/ common/ fonts/   静态资源
```
