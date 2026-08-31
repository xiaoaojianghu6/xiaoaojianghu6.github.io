#!/usr/bin/env python3
"""一次性：从现有页面抽取高风险逐字片段（base64 字体、内联 JS、SVG 图标等）到 templates/snippets/。"""
import os, re

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('templates/snippets', exist_ok=True)

def grab(src, start_marker, end_marker, include_end=True, name=''):
    i = src.find(start_marker)
    assert i >= 0, f'{name}: start not found: {start_marker[:60]}'
    j = src.find(end_marker, i)
    assert j >= 0, f'{name}: end not found: {end_marker[:60]}'
    return src[i:j + (len(end_marker) if include_end else 0)]

def save(name, content):
    open(f'templates/snippets/{name}', 'w', encoding='utf-8').write(content)
    print(f'{name}: {len(content)} chars')

home = open('index.html', encoding='utf-8').read()
about = open('about/index.html', encoding='utf-8').read()
poet = open('poet/index.html', encoding='utf-8').read()
enth = open('enthusiast/index.html', encoding='utf-8').read()
detail = open('builder/multimodal-rag/index.html', encoding='utf-8').read()

# 1. 首页 head 预载/预取/gtag 段（css 链接之前）
save('home_head_preloads.html', grab(home,
    '<link rel="preload" as="image" href="/top/hero/shadow-image.jpg"',
    '<link rel="stylesheet" href="/_astro/about.Cl2ZlrCQ.css">', include_end=False, name='home_head_preloads'))

# 2. 首页 head 尾段（splash 字体+页面 CSS + hoisted/page 脚本 + partytown loader）
save('home_head_tail.html', grab(home,
    '<style>/* Inline subset fonts for splash',
    '</script></head>', include_end=False, name='home_head_tail') + '</script>')

# 3. 首页 body 开头：开屏动画整块（style+markup+scripts，到 bg div 之前）
save('home_splash.html', grab(home,
    '<style>\n.splash-overlay',
    '<div id="bg" class="bg fixed', include_end=False, name='home_splash'))

# 4. 首页尾部脚本（scroll-reveal + 行者人像 + works hover + 视频自动播放）
i = home.find('<script>\n(function(){\n  var revealObserver')
j = home.find('})();</script>', home.find('document.addEventListener(\'visibilitychange\''))
assert i > 0 and j > i
save('home_scripts.html', home[i:j + len('})();</script>')])

# 5. astro-island 运行时 + Stalker 光标（home/about/builder/detail 共用）
save('astro_island.html', grab(home,
    '<style>astro-island,astro-slot',
    '</astro-island>', name='astro_island'))

# 6. about 社交图标 4 项
i = about.find('<ul class="mt-9 flex gap-8 md:mt-14"')
j = about.find('</ul>', i) + len('</ul>')
save('about_social_icons.html', about[i:j])

# 7. about 微信二维码弹层（触发脚本在 build.py 里手写）
i = about.find('<div id="qr-modal"')
j = about.find('</p></div></div>', i)
assert i > 0 and j > i
save('about_wechat_modal.html', about[i:j + len('</p></div></div>')])

# 8. enthusiast 美食 slider GSAP 脚本
i = enth.find('<script type="module">\nimport {g as r}')
j = enth.find('})();\n</script>', i)
save('enthusiast_food_script.html', enth[i:j + len('})();\n</script>')])

# 9. poet 内联 JS（poems 对象之后的部分；poems 对象由 build.py 从 yaml 重新生成）
i = poet.find('// ===== Modal 交互 =====')
j = poet.find('</script>', poet.find("document.querySelectorAll('.long-poem-stanza')"))
assert i > 0 and j > i
save('poem_logic.js.html', poet[i:j])

# 10. 详情页 SCROLL 指示器 + 横滚容器开头
i = detail.find('<div id="scroll" class="fixed right-0')
j = detail.find('md:whitespace-nowrap">', i) + len('md:whitespace-nowrap">')
save('detail_scroll_open.html', detail[i:j])

# 11. detail 尾部（astro-island；详情页现无视频自动播放脚本与页脚）
save('detail_tail.html', grab(detail, '<style>astro-island,astro-slot', '</astro-island>', name='detail_tail'))

print('snippets done')
