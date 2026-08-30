# haozi.dev 全面审计报告

**审计日期:** 2026-07-27
**审计范围:** 架构、视觉、响应式、构建流程
**参照标准:** shuheng.cloud (PurpleInk)

---

## 一、项目概览

William Liu (HAO-Z) 个人作品集网站。双语（中/英），6 个主区块 + 9 个项目详情页。GitHub Pages 部署。

**技术栈:** Vite + React 19 + Tailwind CSS v4 + GSAP 3.15 + Lenis 1.1.6
**构建输出:** 混血架构 — Astro 编译产物 + 手写 HTML + Python 脚本生成

---

## 二、架构问题（11 项）

### CRITICAL

| # | 问题 | 详情 |
|---|------|------|
| 1 | Astro 源码丢失 | `_astro/` 目录仅有编译产物，无任何 `.astro` 源文件。about/index.html、builder/index.html、index.html 的组件无法重新生成或修改 |
| 2 | build.py 模板文件名不匹配 | 脚本读取 `_tpl_before.html` / `_tpl_after.html`，但实际文件名为 `_template_before.html` / `_template_after.html`。运行脚本立即 FileNotFoundError |
| 3 | 字体预加载引用不存在的文件 | 5/6 页面 `<link rel="preload" href="/fonts/zhi-mang-xing-400.ttf">` — 实际文件是 `ZhiMangXing-subset.woff2` |

### HIGH

| # | 问题 | 详情 |
|---|------|------|
| 4 | public/ 目录不存在 | vite.config.js 引用 `public/` 做子页面路由，但该目录不存在。dev 模式下 /about/、/poet/、/wanderer/、/enthusiast/ 无法访问 |
| 5 | 两套架构并存 | Astro 编译页 (about/builder/index.html) 有 header/menu/footer/JS 交互；手写 HTML (enthusiast/poet/wanderer) 无 header/no menu/no footer。页面间体验断裂 |
| 6 | 项目数据 4 处重复 | projects.py (Python dict) → builder.html (JSON data-props) → builder/index.html (JSON data-props) → index.html (inline HTML)。排序不一致 |

### MEDIUM

| # | 问题 | 详情 |
|---|------|------|
| 7 | builder.html 和 builder/index.html 是两个不同实现 | 前者是 Vite dev entry (仅 React)，后者是 Astro 编译输出 (完整 shell)。开发/生产不一致 |
| 8 | 无构建编排 | `python3 builder/build.py` 和 `npm run build` 是独立手动步骤，无脚本串联 |
| 9 | 死脚本 | `"dev:old"` 引用的 `legacy.html` 不存在 |

### LOW

| # | 问题 | 详情 |
|---|------|------|
| 10 | 未使用的依赖 | framer-motion, zustand, js-yaml, react-icons 在 src/ 中无任何导入 |
| 11 | Cinzel 字体格式 | 6 个变体均为 TTF (~277KB)，可转 WOFF2 减少约 40% 体积 |

---

## 三、视觉分析

### 现有优势（必须保留）

- **书法字体品牌识别:** Zhi Mang Xing（标题/导航）+ Cinzel（英文/装饰）+ LXGW WenKai（正文）+ FZXingKai（诗歌）
- **PNG 纹理背景系统:** texture-for-black.png（深色区）+ texture-for-gray.png（浅色区），每 section 叠加不同底色营造微色调差异
- **Splash 屏幕动画:** 4 层时序（进度条 4s → 文字 blur-out 3s → 半屏分割 0.9s → DOM 移除），base64 内联字体确保首帧渲染
- **GSAP ScrollTrigger 系统:** 月亮 133 帧滚动序列、parallax 画廊 pin+scrub、hero 内容平移、footer 视差拉入、WorksSlide 水平滚动
- **Lenis 平滑滚动:** v1.1.6，lerp 0.1，触摸设备特殊处理
- **Stalker 自定义光标:** React island + GSAP quickSetter + ticker 弹簧物理
- **玻璃拟态卡片:** enthusiast-gallery 使用 `backdrop-filter: blur(12px)` + 半透明背景 + 各卡片独立悬停色
- **数据属性驱动主题:** `data-[color=dark/bright]` 切换全站明暗，`data-[active=true]` 触发动画

### 现有劣势（与 shuheng.cloud 差距）

| 差距点 | haozi.dev 现状 | shuheng.cloud 做法 |
|--------|---------------|-------------------|
| mix-blend 图层混合系统 | 2 次 ad-hoc 使用 | color/difference/multiply 三层系统化叠加 |
| backdrop-filter + mask-image 联动 | 仅 enthusiast 卡片有基础 backdrop-filter | header blur(12px) + mask-image 6 阶渐变消失 |
| mask-image 渐变裁剪 | 完全未使用 | header/footer/section 过渡全部使用 |
| filter: blur() 作为动画工具 | scroll-reveal 用了 blur(4px)→0 但参数单一 | GSAP from blur(10px/5px)→0，三列不同参数创造空间深度 |
| SVG feGaussianBlur 有机背景 | 无 | Hero 超大半径模糊椭圆 (stdDeviation=125) |
| Canvas 鼠标跟随光效 | 无 | 模板卡片 onMouseMove + radialGradient + lerp 平滑 |

### 现有资产清单

```
fonts/ (9 files, ~1.7MB)
  ZhiMangXing-subset.woff2  551KB
  LXGWWenKai-subset.woff2   407KB
  FZXingKai-subset.woff2    438KB
  cinzel-400~900.ttf x6     ~277KB

common/ (7 files)
  favicon.ico, icon.svg, apple-touch-icon.png
  ogp.png, menu.mp4 (96s)
  texture-for-black.png, texture-for-gray.png

top/ (61 images)
  hero/shadow-movie.mp4 + shadow-image.jpg
  about/ 8 images (视差画廊)
  enthusiast/ 32 images (电影/音乐/书籍/运动/美食卡片)
  poet/ 5 images, wanderer/ 8 images

detail/ (52 files)
  9 项目 × (mv-pc.jpg + mv-sp.jpg + mv-vertical.jpg + 项目图片/PDF/视频)
```

### Astro 组件映射（7 个已编译组件）

| data-astro-cid | 推断组件名 | HTML 作用域 |
|---|---|---|
| nj3qennt | Layout/Background.astro | `<html>`, `<body>`, `<div id="bg">` |
| z6iz25dn | Header.astro | `<header>`, 所有 nav 项 |
| xcczry7d | Menu.astro | 全屏叠加菜单 |
| 5saot5ic | Hero.astro | Hero section + 视频 + 月亮动画 |
| obodp3z4 | ParallaxGallery.astro | 8 张 parallax 图片 |
| cwk4bb2u | ImageLink.astro | Works 画廊项目项 |
| iravouwq | Footer.astro | GSAP 视差 footer + 跑马灯 |

---

## 四、响应式问题（10 项）

### 根因
全站 **零 `sm:` 断点**（Tailwind 640px），仅依赖 `md:`（768px）。小屏手机（320-639px）和平板（640-767px）共享同一套"移动端"样式，且该样式实际是桌面优先的退化效果。

### 问题清单

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | Wanderer 文本容器 `max-width:860px` + `padding:4rem 1rem 0 clamp(1rem,18vw,20rem)` | index.html:85 `#wanderer-gallery` | 320px 屏幕下最小宽度 981px，强制横向溢出 |
| 2 | `w-screen` 在 footer divider | index.html:95 `class="w-screen"` | 100vw 包含滚动条宽度，可溢出视口 |
| 3 | 0 个 `sm:` 断点 | 全站 CSS | 无小屏专用适配 |
| 4 | 仅 1 个 `@media` 查询 | index.html | 所有响应式依赖 Tailwind md: |
| 5 | 菜单关闭按钮 40px×40px | 所有页面 `.ts-menu-close` | WCAG 要求最小 44px |
| 6 | 社交图标 22px×22px 无 padding | about/index.html | 远低于 44px 触控标准 |
| 7 | 菜单字体 12px/13px | 所有页面菜单序数词和英文标签 | 低于推荐最小 16px |
| 8 | overflow-x:hidden 掩盖根因 | body + main | 隐藏溢出但不修复，内容可能被裁剪 |
| 9 | 教育时间线强制水平滑动 | about/index.html | 移动端无垂直布局回退 |
| 10 | 5 个硬编码 px 宽度 | index.html | 600/760/767/860/1120px，无一使用 clamp() |

### 已有的好做法
- viewport meta 正确设置
- 主要标题使用 `clamp()` 流体排版
- 视频/图片使用 `object-cover` + `w-full`

---

## 五、推荐重构方案（不立即执行）

### 技术决策: 全量迁移到 Astro
**原因:** 
- Astro 源码丢失，无法增量修改
- Python HTML 字符串拼接不可维护
- 两套架构并存，统一成本高于重建
- Astro Islands 完美匹配现有的 React 组件穿插模式

### 迁移原则
- **保留:** 所有字体/图片/视频/纹理、GSAP/Lenis 动画逻辑、Splash 时序、Stalker 光标、WorksSlide 组件
- **删除:** `_astro/`、`builder/__pycache__/`、`assets/`、`build.py`、`projects.py`、`微信.JPG`、`新项目/`、`_archive/old-site/`
- **重建:** 7 个 Astro 组件 + 6 个页面 + Content Collections 替代 projects.py
- **新增:** 6 项现代 CSS 技术 + 10 项响应式修复
- **不执行:** 图片色调统一（个人作品集不需要品牌色统一）

### 目录结构（目标）
```
src/
├── layouts/Base.astro
├── components/
│   ├── Header.astro
│   ├── Menu.astro
│   ├── SplashScreen.astro
│   ├── Footer.astro
│   ├── TextureBackground.astro
│   ├── ScrollReveal.astro
│   ├── WorksSlide.jsx          # 迁移自原 src/components/
│   └── Stalker.jsx             # 迁移自原 Astro island
├── pages/
│   ├── index.astro
│   ├── about.astro
│   ├── builder.astro
│   ├── builder/[slug].astro
│   ├── enthusiast.astro
│   ├── poet.astro
│   └── wanderer.astro
├── content/
│   ├── config.ts               # Zod schema
│   └── projects/*.md           # 替代 projects.py
└── styles/
    └── design-tokens.css
```

### 视觉升级清单

| # | 技术 | 应用位置 |
|---|------|---------|
| 1 | mix-blend-mode: difference | Header 竖排文字 — 滚动自动反色 |
| 2 | backdrop-filter + mask-image | Header 毛玻璃渐变消失 |
| 3 | mask-image 渐变裁剪 | Section 之间过渡 |
| 4 | filter: blur() 差异化参数 | ScrollReveal — 三列不同起始 blur 创造深度 |
| 5 | Canvas onMouseMove 光效 | Builder 项目卡片 |
| 6 | SVG feGaussianBlur 有机背景 | Hero 替代纯色背景 |

### 响应式修复清单

| # | 修复 |
|---|------|
| 1 | 全站引入 sm:(640px) 断点 |
| 2 | 硬编码 px → clamp() 流体值 |
| 3 | w-screen → max-w-full overflow-hidden |
| 4 | 触控目标 ≥ 44px |
| 5 | 菜单字体 ≥ 16px |
| 6 | 社交图标包裹 44px 触控区 |
| 7 | 引入 container queries |
| 8 | 修复根因后移除 overflow-x:hidden |
| 9 | 时间线移动端垂直布局 |
| 10 | 竖排文字 clamp() 适配 |

---

## 六、当前 Skill 栈（已更新）

| Skill | 用途 | 状态 |
|-------|------|------|
| impeccable | 前端设计/构建 | 默认主 skill |
| huashu-design | 设计方向 + 品牌资产 + 反 AI slop | 已修复路由 |
| emil-kowalski | Framer Motion/motion.dev | 已修复路由 |
| modern-css-layers | mix-blend/backdrop/mask/blur/SVG filter | 新建 |
| tailwindcss-mobile-first | 响应式 clamp/container queries/触控目标 | 新装 |
| antislopui | GSAP/Lenis/Motion/ZProximity + 8 agent | 新装 |
| motion-dev-animations | Motion.dev 120fps GPU 动画 | 新装 |
| frontend-design | 已删除（零现代 CSS 覆盖） | 已废黜 |
| theme-factory | 已删除（纯调色板，零设计贡献） | 已废黜 |

---

## 七、参照标准

shuheng.cloud (PurpleInk) — Next.js + Framer Motion + GSAP ScrollTrigger + Tailwind v4
核心技术栈：mix-blend 三模式图层系统 + backdrop-filter/mask-image 联动 + Canvas 鼠标光效 + blur() 动画 + SVG feGaussianBlur + mask-image 渐变裁剪

分析日期: 2026-07-27
