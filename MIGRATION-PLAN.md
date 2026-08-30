# haozi.dev 重构执行计划

**基于:** SITE-AUDIT-2026-07-27.md
**执行前必读:** 完整审计报告
**警告:** 此文件是执行指南，实际重构由高级模型完成。不动现有代码。

---

## Phase 1: 项目初始化

```bash
# 在原目录下初始化 Astro
npm create astro@latest . -- --template minimal --install --skip-houston
npm install @astrojs/react @astrojs/tailwindcss
npm install react react-dom gsap framer-motion
npm install @tailwindcss/vite tailwindcss
```

### astro.config.mjs
```js
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwind from '@astrojs/tailwindcss';

export default defineConfig({
  integrations: [react(), tailwind()],
  site: 'https://haozi.dev',
  output: 'static',
});
```

### 删除清单
```
rm -rf _astro/
rm -rf builder/__pycache__/
rm -rf assets/
rm -f builder/build.py builder/projects.py
rm -f builder/_template_before.html builder/_template_after.html
rm -f 微信.JPG
rm -rf 新项目/
rm -rf _archive/old-site/
rm -f builder.html           # 被 pages/builder.astro 替代
rm -f package-lock.json      # 重新生成
rm -f "dev:old" script
```

### 清理 package.json
```json
{
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview"
  },
  "dependencies": {
    "@astrojs/react": "latest",
    "@astrojs/tailwindcss": "latest",
    "astro": "latest",
    "react": "^19.2.6",
    "react-dom": "^19.2.6",
    "gsap": "^3.15.0",
    "framer-motion": "^12.40.0",
    "clsx": "^2.1.1",
    "tailwind-merge": "^3.6.0",
    "tailwindcss": "^4.3.0"
  }
}
```

---

## Phase 2: 组件依赖关系

```
Base.astro (全局 layout)
├── Header.astro            # 需要: data-[color] 主题切换, mix-blend-difference 文字
├── Menu.astro              # 需要: menu.mp4 背景, 7 导航项, mailto
├── SplashScreen.astro      # 需要: base64 字体, 4.6s 完整时序
├── Footer.astro            # 需要: GSAP ScrollTrigger 视差, 跑马灯 marquee
├── TextureBackground.astro # 需要: texture-for-black/gray.png, bg-color 覆盖
├── ScrollReveal.astro      # 需要: IntersectionObserver, filter:blur()→0
└── Stalker.jsx             # React island, client:only="react"

index.astro (首页)
├── Hero section            # 需要: shadow-movie.mp4, 月亮 133 帧动画, Lenis
├── Parallax gallery        # 需要: 8 张 about 图片, GSAP pin+scrub, skewY
├── Works gallery           # 需要: 5 个项目, hover 预览, ImageLink 系统
├── Wanderer gallery        # 需要: 8 段古文, wanderer-figure.jpg 浮动
├── Poet gallery            # 需要: 背水 8 行诗, openingFlash 动画
└── Enthusiast gallery      # 需要: 5 张玻璃拟态卡片, 各卡片独立悬停色

builder/[slug].astro (项目详情)
├── Content Collection 读取 # 替代 projects.py
├── Hero 图片              # mv-pc.jpg
├── Overview 段落
├── Sections (text/image/video/pdf/image_panel)
├── Back 链接
└── Next 项目链接           # mv-vertical.jpg 桌面 / mv-sp.jpg 移动
```

---

## Phase 3: Content Collection Schema

### src/content/config.ts
```typescript
import { defineCollection, z } from 'astro:content';

const projects = defineCollection({
  schema: z.object({
    title: z.string(),
    subtitle: z.string(),
    role: z.string(),
    date: z.string(),
    hero: z.string(),
    hero_alt: z.string(),
    overview: z.array(z.string()),
    tags: z.array(z.string()),
    categories: z.array(z.string()),
    year: z.string(),
    image: z.string(),
    next_slug: z.string().optional(),
    next_title: z.string().optional(),
    sections: z.array(z.object({
      type: z.enum(['text', 'image', 'video', 'pdf', 'image_panel']),
      tag: z.string().optional(),
      paragraphs: z.array(z.string()).optional(),
      src: z.string().optional(),
      alt: z.string().optional(),
      caption: z.string().optional(),
      title: z.string().optional(),
      images: z.array(z.object({
        src: z.string(),
        alt: z.string(),
        label: z.string(),
        caption: z.string().optional(),
      })).optional(),
    })),
  }),
});

export const collections = { projects };
```

---

## Phase 4: 关键代码模式

### mix-blend-difference 文字（Header 竖排导航）
```astro
<a class="text-pampas mix-blend-difference vertical-rl">
  {zh_title}
</a>
```

### backdrop-filter + mask-image 联动（Header 毛玻璃）
```astro
<div class="pointer-events-none fixed top-0 left-0 z-40 h-32 w-full"
  style="backdrop-filter: blur(12px);
    mask-image: linear-gradient(to bottom, black 0%, black 20%,
      rgba(0,0,0,0.4) 60%, transparent 100%);">
</div>
```

### mask-image section 过渡
```css
.section-fade-bottom {
  mask-image: linear-gradient(to bottom, black 0%, black 80%, transparent 100%);
}
```

### Canvas 鼠标光效（Builder 卡片）
参考 `modern-css-layers/references/canvas-effects.md`

### 响应式 clamp() 替换硬编码 px
```css
/* 前: max-width:860px */
/* 后: */
width: clamp(320px, 90vw, 860px);

/* 前: font-size:12px */
/* 后: */
font-size: clamp(0.875rem, 2vw, 1rem);

/* 触控目标 */
.touch-target { min-width: 44px; min-height: 44px; }
```

---

## Phase 5: 验证清单

- [ ] `npm run dev` 首页 6 section 全部渲染
- [ ] 移动端 375px Chrome DevTools 无横向溢出
- [ ] 所有 sm: 断点生效（640px 以下专用布局）
- [ ] Header 滚动时 backdrop-filter + mask-image 渐变消失
- [ ] 所有子页面导航/布局/字体一致
- [ ] Builder 页面 project cards 正常渲染和筛选
- [ ] `npm run build` 成功，dist/ 输出完整
- [ ] `npx serve dist` 所有路由正常
- [ ] 无 console 错误（特别是字体 404）
- [ ] canonical URL 为 https://haozi.dev
- [ ] OG 图片 URL 正确
- [ ] 所有触控目标 ≥ 44px
- [ ] 所有正文字体 ≥ 16px
