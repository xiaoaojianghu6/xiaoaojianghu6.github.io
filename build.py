#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""============================================================
站点生成器 —— 把 content/ + projects/ 里的数据渲染成静态 HTML
============================================================
用法：
    python3 build.py            # 生成全部页面
    python3 build.py --check    # 只做数据校验，不写文件

数据在哪里改：
    content/*.yaml              # 各页面文字/图片内容
    projects/<slug>/project.yaml# 创客区项目（一个项目一个文件夹）
    content/site.yaml           # 导航/菜单/页脚/SEO
详细说明见 CONTENT-GUIDE.md
"""
import os
import sys
import glob
import re
import json

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

VERBOSE = True

# ---------------------------------------------------------- 基础工具

def load_yaml(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)

def read_file(path):
    with open(path, encoding='utf-8') as f:
        return f.read()

def partial(name, **subs):
    """读 templates/partials/<name> 并替换 {{TOKEN}}"""
    html = read_file(f'templates/partials/{name}')
    for key, val in subs.items():
        html = html.replace('{{%s}}' % key, val)
    leftover = re.findall(r'\{\{[A-Z_]+\}\}', html)
    if leftover:
        raise SystemExit(f'模板 {name} 有未替换的占位符: {leftover}')
    return html

def snippet(name):
    return read_file(f'templates/snippets/{name}')

def page_style(name):
    return read_file(f'templates/pages/{name}')

def write(path, content):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    log(f'  写出 {path} ({len(content)//1024}KB)')

def log(msg):
    if VERBOSE:
        print(msg)

# ---------------------------------------------------------- 数据加载

SITE = load_yaml('content/site.yaml')
HOME = load_yaml('content/home.yaml')
WANDERER = load_yaml('content/wanderer.yaml')
POEMS = load_yaml('content/poems.yaml')
ENTHUSIAST = load_yaml('content/enthusiast.yaml')
ABOUT = load_yaml('content/about.yaml')

def load_projects():
    """读取 projects/*/project.yaml，按 order 排序，自动接“下一项目”"""
    projects = []
    for yml in sorted(glob.glob('projects/*/project.yaml')):
        slug = yml.split('/')[1]
        if slug == '_template':
            continue
        p = load_yaml(yml)
        p.setdefault('slug', slug)
        p.setdefault('order', 99)
        p.setdefault('home', True)
        projects.append(p)
    projects.sort(key=lambda x: x.get('order', 99))
    for i, p in enumerate(projects):
        nxt = projects[(i + 1) % len(projects)]
        p['_next'] = {'slug': nxt['slug'], 'title': nxt['title']}
    return projects

PROJECTS = load_projects()

# ---------------------------------------------------------- 校验

MEDIA_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.gif', '.mp4', '.mov', '.pdf',
              '.woff2', '.ttf', '.svg', '.ico')

def check_paths(data, errors):
    """校验 yaml 中引用的本地媒体文件是否存在"""
    import urllib.parse
    if isinstance(data, dict):
        for v in data.values():
            check_paths(v, errors)
    elif isinstance(data, list):
        for v in data:
            check_paths(v, errors)
    elif isinstance(data, str):
        if data.startswith('/') and not data.startswith('//'):
            path = urllib.parse.unquote(data.split('#')[0])
            if os.path.splitext(path)[1].lower() in MEDIA_EXTS and not os.path.exists('.' + path):
                errors.append('文件不存在: ' + path)

def validate():
    errors = []
    for name, data in [('home', HOME), ('wanderer', WANDERER), ('poems', POEMS),
                       ('enthusiast', ENTHUSIAST), ('about', ABOUT), ('site', SITE)]:
        check_paths(data, errors)
    for p in PROJECTS:
        for key in ('title', 'subtitle', 'role', 'date', 'hero', 'overview'):
            if key not in p:
                errors.append(f"项目 {p['slug']} 缺少必填字段: {key}")
        check_paths(p, errors)
    if errors:
        print('!! 数据校验失败：')
        for e in errors:
            print('   -', e)
        sys.exit(1)
    log(f'数据校验通过：{len(PROJECTS)} 个项目，5 个内容文件')

# ---------------------------------------------------------- 共用片段生成

CN_NUM = '一二三四五六七八九十'

def cn_num(i):
    return CN_NUM[i] if 0 <= i < len(CN_NUM) else str(i + 1)

def render_header_nav():
    items = []
    for nav in SITE['header_nav']:
        spans = ('<span style="display:block;height:1px;width:14px;background:currentColor;opacity:0.25"></span> ' * 3)
        items.append(
            f'<li class="relative" data-astro-cid-z6iz25dn style="display:flex;flex-direction:column;align-items:center;gap:3px"> '
            f'{spans}'
            f'<a href="{nav["href"]}" class="ts-text-link text-3xl vertical-rl md:text-2xl" data-vertical="rl" '
            f'style="font-family:\'Zhi Mang Xing\',cursive" data-astro-cid-z6iz25dn>{nav["label"]}</a> '
            f'{spans}</li>')
    return ''.join(items)

def render_menu_items():
    items = []
    for m in SITE['menu_items']:
        items.append(
            f'<li style="list-style:none" data-astro-cid-xcczry7d> '
            f'<a href="{m["href"]}" class="group" style="text-decoration:none;transition:opacity 0.6s" '
            f'data-stalker-color="bright" data-astro-cid-xcczry7d> '
            f'<span style="display:flex;flex-direction:row;align-items:center;gap:clamp(1rem,2.5vw,2rem);width:100%"> '
            f'<span style="font-size:12px;color:var(--color-taupe-gray,#B4AC97)">({m["num"]})</span> '
            f'<span style="font-family:\'Zhi Mang Xing\',cursive;font-size:clamp(1.8rem,5vw,2.8rem);'
            f'color:var(--color-pampas,#F9F9F6);line-height:1.2">{m["zh"]}</span> '
            f'<span style="flex:1;height:1px;min-width:30px;background:var(--color-emperor,#4F4F4F)"></span> '
            f'<span style="font-family:Cinzel,serif;font-size:13px;color:var(--color-silver-chalice,#A9A9A9);'
            f'letter-spacing:0.15em">{m["en"]}</span> </span> </a> </li>')
    return ''.join(items)

def render_chrome(color, blend, style_attr='', menu_max_width='600px'):
    return partial('chrome.html',
                   HEADER_COLOR=color, HEADER_BLEND=blend, HEADER_STYLE=style_attr,
                   HEADER_NAV=render_header_nav(),
                   MENU_ITEMS=render_menu_items(),
                   MENU_MAX_WIDTH=menu_max_width)

def render_marquee_row():
    spans = []
    words = SITE['marquee_words']
    for i, w in enumerate(words):
        weight = '600' if i == 0 else '400'
        color = 'var(--color-emperor,#4F4F4F)' if i == 1 else 'var(--color-taupe-gray,#B4AC97)'
        spans.append(f'<span style="font-family:Cinzel,serif;font-size:clamp(7rem,14vw,12rem);'
                     f'font-weight:{weight};color:{color};letter-spacing:0.03em;padding:0 5rem">{w}</span> ')
    return ('<span style="display:flex;flex-shrink:0;white-space:nowrap;align-items:center"> '
            + ''.join(spans) + '</span> ')

def render_footer():
    row = render_marquee_row()
    return partial('footer.html', MARQUEE_ROW=row + row)

def render_footer_scripts(prefetch_slugs=None):
    pages = ', '.join('"%s"' % u for u in SITE['prefetch_pages'])
    s = (f'<script>(function(){{var p=[{pages}];'
         f'p.forEach(function(u){{fetch(u,{{priority:"low"}}).catch(function(){{}})}});}})();</script>')
    if prefetch_slugs:
        slugs = ', '.join('"%s"' % s for s in prefetch_slugs)
        s += (f'<script>(function(){{var s=[{slugs}];'
              f's.forEach(function(u){{fetch("/builder/"+u+"/",{{priority:"low"}}).catch(function(){{}})}});}})();</script>')
    return s

VIDEO_AUTOPLAY_SCRIPT = '''<script>(function(){
var vids=document.querySelectorAll('video[autoplay]');
if(!vids.length)return;
var played=new WeakSet();
function tp(v){if(played.has(v))return;var p=v.play();if(p&&p.then){p.then(function(){played.add(v)}).catch(function(){})}}
function tpa(){vids.forEach(tp)}
tpa();
function og(){tpa();document.removeEventListener('click',og);document.removeEventListener('touchstart',og)}
document.addEventListener('click',og);document.addEventListener('touchstart',og);
document.addEventListener('visibilitychange',function(){if(!document.hidden)tpa()});
if('IntersectionObserver' in window){
var ob=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting)tp(e.target)})},{threshold:0.1});
setTimeout(function(){vids.forEach(function(v){if(v.paused&&!v.ended)ob.observe(v)})},2000)}
})();</script>'''

WF_READY_SCRIPT = ('<script>document.fonts.ready.then(function(){'
                   'document.documentElement.classList.add(\'wf-ready\')});'
                   'setTimeout(function(){document.documentElement.classList.add(\'wf-ready\')},800)</script>')

CSS_LINKS = ('<link rel="stylesheet" href="/_astro/about.Cl2ZlrCQ.css">\n'
             '<link rel="stylesheet" href="/_astro/about.j05OPDV1.css">\n'
             '<link rel="stylesheet" href="/_astro/font-override.css">')

FONT_PRELOADS_ALL = ('<link rel="preload" href="/fonts/ZhiMangXing-subset.woff2" as="font" type="font/woff2" crossorigin>'
                     '<link rel="preload" href="/fonts/LXGWWenKai-subset.woff2" as="font" type="font/woff2" crossorigin>'
                     '<link rel="preload" href="/fonts/FZXingKai-subset.woff2" as="font" type="font/woff2" crossorigin>')
FONT_PRELOAD_ZMX = ('<link rel="preload" href="/fonts/ZhiMangXing-subset.woff2" as="font" type="font/woff2" crossorigin>')

GTAG_BLOCK = ("""<!-- Google tag (gtag.js) --><script async src="https://www.googletagmanager.com/gtag/js?id="""
              + SITE['ga_id'] + '"></script><script>'
              'window.dataLayer = window.dataLayer || [];'
              'function gtag(){dataLayer.push(arguments);}'
              "gtag('js', new Date());"
              "gtag('config', '" + SITE['ga_id'] + "');</script>")

def render_head(title, desc, canonical, extra_head='', html_attr=''):
    t = title.replace('"', '&quot;')
    d = desc.replace('"', '&quot;')
    return f'''<!DOCTYPE html><html lang="zh-CN"{html_attr}> <head><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="/common/favicon.ico" sizes="32x32"><link rel="icon" href="/common/icon.svg" type="image/svg+xml"><link rel="apple-touch-icon" href="/common/apple-touch-icon.png"><link rel="manifest" href="/manifest.webmanifest"><link rel="sitemap" href="/sitemap.xml"><title>{t}</title><meta charset="UTF-8"><link rel="canonical" href="{canonical}"><meta name="description" content="{d}"><meta name="robots" content="index, follow"><meta property="og:title" content="{t}"><meta property="og:type" content="website"><meta property="og:image" content="/common/ogp.png"><meta property="og:url" content="{canonical}"><meta property="og:description" content="{d}"><meta property="og:locale" content="zh_CN"><meta property="og:site_name" content="{SITE['site_name']}"><meta property="og:image:url" content="/common/ogp.png"><meta property="og:image:alt" content="{t}"><meta name="twitter:card" content="summary"><meta name="twitter:site" content="{SITE['twitter_site']}"><meta name="twitter:title" content="{t}"><meta name="twitter:image" content="/common/ogp.png"><meta name="twitter:image:alt" content="{t}"><meta name="twitter:description" content="{d}"><meta name="twitter:creator" content="{SITE['twitter_site']}">{extra_head}{CSS_LINKS}</head> '''

# ---------------------------------------------------------- MORE 按钮 / 区块标题（首页各区块共用）

def more_button(href, dark=True):
    if dark:
        return (f'<div class="ts-focus-in mt-16 data-[active=\'true\']:animate-text-focus-in md:mt-[5.5rem]"> '
                f'<a href="{href}" class="ts-more group relative inline-block overflow-hidden rounded-[50%] border '
                f'px-10 py-5 font-serif-en leading-none transition-colors duration-[0.6s] '
                f"data-[color='bright']:border-silver-chalice data-[color='dark']:border-gray "
                f"data-[color='bright']:text-pampas data-[color='dark']:text-black "
                f"data-[color='bright']:hover:border-black data-[color='dark']:hover:border-pampas "
                f"data-[color='bright']:hover:text-black data-[color='dark']:hover:text-pampas "
                f'md:px-12 md:py-6 md:text-lg" data-color="bright"> '
                f'<span class="absolute left-1/2 top-1/2 z-0 h-0 w-0 -translate-x-1/2 -translate-y-1/2 rounded-[50%] '
                f"transition-all duration-[0.6s] group-hover:h-full group-hover:w-full data-[color='bright']:bg-pampas "
                f"data-[color='dark']:bg-black\" data-color=\"bright\"></span> "
                f'<span class="relative z-0">MORE</span> </a> </div>')
    return (f'<div class="ts-focus-in mt-16 data-[active=\'true\']:animate-text-focus-in md:mt-[5.5rem]"> '
            f'<a href="{href}" class="ts-more group relative inline-block overflow-hidden rounded-[50%] border '
            f'px-10 py-5 font-serif-en leading-none transition-colors duration-[0.6s] md:px-12 md:py-6 md:text-lg" '
            f'style="border-color:#1a1a1a;color:#1a1a1a"> '
            f'<span class="absolute left-1/2 top-1/2 z-0 h-0 w-0 -translate-x-1/2 -translate-y-1/2 rounded-[50%] '
            f'transition-all duration-[0.6s] group-hover:h-full group-hover:w-full" style="background:#1a1a1a"></span> '
            f'<span class="relative z-0" style="color:#1a1a1a">MORE</span> </a> </div>')

def section_title_block(href, en, zh, style=''):
    color = 'color:inherit'
    return (f'<div class="text-taupe-gray"> <h2 class="ts-focus-in data-[active=&quot;true&quot;]:animate-text-focus-in '
            f'text-current flex"> <a href="{href}" style="display:flex;flex-direction:column;align-items:center;'
            f'gap:0.3rem;text-decoration:none;{color}"> '
            f'<span class="md:text-lg leading-none font-medium" style="font-family:Cinzel,serif;letter-spacing:0.15em">{en}</span> '
            f'<span class="text-[2rem] md:text-[2.625rem] leading-none font-medium" '
            f"style=\"font-family:'Zhi Mang Xing',cursive\">{zh}</span> </a> </h2> </div>")

# ---------------------------------------------------------- 首页

def render_work_card(p, idx):
    n = cn_num(idx)
    img = p.get('list_image') or p['hero']
    cat = (p.get('categories') or [''])[0]
    tags = ''.join(f'<li class="text-black text-base md:text-lg" data-astro-cid-cwk4bb2u>{t}</li>'
                   for t in p.get('tags', []))
    return (
        '<li class="ts-image-link image-link group md:contents-right md:[&:not(:first-child)>a]:pt-14 '
        '[&:not(:last-child)>a]:pb-9 md:[&:not(:last-child)>a]:pb-0" data-astro-cid-cwk4bb2u data-image-only="true">\n'
        f' <a href="/builder/{p["slug"]}/" class="flex flex-col gap-4 border-t border-gray-300 md:flex-row '
        'md:gap-16 md:border-none md:pl-[10%]" data-astro-cid-cwk4bb2u> '
        '<div class="flex md:hidden justify-between pt-3 transition-colors duration-300 '
        'group-[.is-opacity]:opacity-50" data-astro-cid-cwk4bb2u> '
        f'<span class="text-black" data-astro-cid-cwk4bb2u>({n})</span> '
        f'<span class="font-serif-en text-black" data-astro-cid-cwk4bb2u>{p.get("year","")}</span> </div> '
        '<div class="flex-1 flex flex-col md:pt-3 md:border-t border-gray-300 transition-colors duration-300 '
        'group-[.is-opacity]:opacity-50" data-astro-cid-cwk4bb2u> '
        '<div class="hidden justify-between md:flex" data-astro-cid-cwk4bb2u> '
        f'<span class="text-black text-lg" data-astro-cid-cwk4bb2u>({n})</span> '
        f'<span class="font-serif-en text-black text-lg" data-astro-cid-cwk4bb2u>{p.get("year","")}</span> </div> '
        f'<h3 class="font-medium text-[1.6rem] md:text-[2rem] md:mt-6 text-black" data-astro-cid-cwk4bb2u>{p["title"]}</h3> '
        '<ul class="mt-4 font-serif-en text-black text-base md:text-lg" data-astro-cid-cwk4bb2u> '
        f'<li data-astro-cid-cwk4bb2u>{cat}</li> </ul> '
        '<ul class="flex md:flex-col flex-wrap gap-y-2 gap-x-3 leading-none mt-7 md:mt-auto" data-astro-cid-cwk4bb2u> '
        f'{tags} </ul> </div> '
        '<picture class="order-1 md:order-2 cursor-pointer" style="width:100%;max-width:50vw" '
        'data-astro-cid-cwk4bb2u data-image-trigger="true"> '
        f'<img src="{img}" alt="{p["title"]}" class="w-full object-contain" data-astro-cid-cwk4bb2u> </picture> </a> '
        '<picture class="image-link__bg pointer-events-none invisible fixed left-0 top-0 z-0 order-3 hidden '
        'h-screen w-screen opacity-0 transition-all duration-300 md:block" aria-hidden="true" data-astro-cid-cwk4bb2u> '
        f'<img src="{img}" alt="{p["title"]}" class="h-full w-full object-contain" '
        'style="object-position:right center" data-astro-cid-cwk4bb2u> </picture> </li>')

def render_home():
    h = HOME['hero']
    title_join = '<br>'.join(h['title_lines'])
    # --- 视差画廊 ---
    left = []
    right = []
    for item in HOME['parallax']:
        li = (f'<li class="ts-parallax-gallery-image {item["class"]}" data-astro-cid-obodp3z4> '
              '<picture class="block w-full mix-blend-multiply" data-astro-cid-obodp3z4> '
              f'<img src="{item["src"]}" alt="{item["alt"]}" class="w-full" data-astro-cid-obodp3z4> </picture> '
              '<div class="bg-layer absolute -bottom-1/2 left-0 -z-10 h-[200%] w-full bg-taupe-gray" '
              'data-astro-cid-obodp3z4></div> </li>')
        (left if item['side'] == 'left' else right).append(li)
    parallax = (
        '<section id="parallax-gallery"> <div class="relative h-screen overflow-hidden py-[4.6rem] text-pampas '
        'contents-full md:py-44"> <div class="absolute left-0 top-0 h-full w-full bg-mine-shaft-texture" '
        'style="background-color:#4A3D35"></div> <div class="flex justify-between"> '
        '<ul class="flex flex-1 flex-col items-start">' + ''.join(left) + '</ul> '
        '<ul class="flex flex-1 flex-col items-end">' + ''.join(right) + '</ul> </div> '
        '<div class="absolute left-1/2 top-1/2 z-20 flex h-screen w-screen -translate-x-1/2 -translate-y-1/2 '
        'flex-col items-center justify-center"> <div class="text-taupe-gray"> '
        '<h2 class="ts-focus-in data-[active=&#34;true&#34;]:animate-text-focus-in text-current flex"> '
        '<a href="/about" class="md:vertical-rl grid gap-2 md:gap-4" '
        'style="text-decoration:none;color:var(--color-taupe-gray,#B4AC97)"> '
        '<span style="font-family:Cinzel,serif;letter-spacing:0.15em;font-size:clamp(0.8rem,1.5vw,1rem)">ABOUT</span> '
        "<span style=\"font-family:'Zhi Mang Xing',cursive;font-size:clamp(2rem,4vw,2.625rem);line-height:1.3\">述己</span> "
        '</a> </h2> </div> '
        + more_button('/about') + ' </div> </div> </section>')

    # --- 创客区作品卡 ---
    home_count = HOME.get('project_count', 4)
    home_projects = [p for p in PROJECTS if p.get('home', True)][:home_count]
    cards = ''.join(render_work_card(p, i) for i, p in enumerate(home_projects))
    works = (
        '<section id="works-gallery" style="background-color:#C2B3A4;'
        'background-image:url(/common/texture-for-gray.png);background-repeat:repeat" '
        'class="relative py-24 text-current contents-full md:py-[17.5rem]"> '
        '<div class="flex md:justify-end md:pr-[20%]"> '
        '<h2 class="ts-focus-in data-[active=&#34;true&#34;]:animate-text-focus-in text-current flex"> '
        '<a href="/builder" class="md:vertical-rl grid gap-2 md:gap-4"> '
        '<span class="md:text-xl leading-none font-medium" style="font-family:Cinzel,serif;letter-spacing:0.15em;'
        'color:#1a1a1a">BUILDER</span> '
        "<span class=\"leading-none font-medium\" style=\"font-family:'Zhi Mang Xing',cursive;color:#1a1a1a;"
        'font-size:clamp(2.5rem,6vw,5rem)">作品集</span> </a> </h2> </div> '
        '<ul class="mt-9 md:mt-24">\n' + cards + '\n</ul> '
        + section_title_block('/builder', 'BUILDER', '创 客')
        + more_button('/builder', dark=False) + ' </section>')

    # --- 行者区 ---
    quotes = []
    for q in HOME['wanderer_quotes']:
        mb = '10rem' if q.get('last') else '3.5rem'
        quotes.append(
            '<div class="scroll-reveal" style="font-family:\'FZXingKai\',\'LXGW WenKai\',serif;'
            'font-size:clamp(1.3rem,4.5vw,2.8rem);line-height:2.2;opacity:0;transform:translateY(50px);'
            'filter:blur(2px);transition:opacity 0.9s ease-out,transform 0.9s ease-out,filter 0.9s ease-out;'
            f'margin-bottom:{mb};text-align:center"><p>{q["text"]}</p></div>')
    fig = HOME['wanderer_figure']
    wanderer = (
        '<section id="wanderer-gallery" class="relative py-[4.6rem] md:py-44 text-pampas contents-full"> '
        '<div class="absolute left-0 top-0 h-full w-full bg-mine-shaft-texture" style="background-color:#3E4337"></div> '
        '<style>@keyframes wandererFloat{0%,100%{transform:translateY(-50%) translateX(0)}'
        '25%{transform:translateY(-51.2%) translateX(2px)}50%{transform:translateY(-48.8%) translateX(-1px)}'
        '75%{transform:translateY(-50.4%) translateX(1px)}} '
        '@media(max-width:767px){.wanderer-quotes-wrap{padding:3.5rem 1.25rem 0}}'
        '#wanderer-figure img{object-fit:contain}'
        '@media(max-width:767px){#wanderer-figure{left:auto;right:0;width:clamp(220px,58vw,420px);'
        'bottom:8vh;top:auto;transform:translateY(0)}}</style> '
        '<div id="wanderer-figure" style="position:fixed;left:5vw;top:50%;'
        'transform:translateY(-50%);z-index:5;width:clamp(420px,42vw,750px);opacity:0;'
        'transition:opacity 1.5s ease-out;pointer-events:none;animation:wandererFloat 7s ease-in-out infinite"> '
        '<div style="position:relative;width:100%;padding-bottom:140%"> '
        f'<img src="{fig}" alt="徒步中的行者" style="position:absolute;inset:0;z-index:1;width:100%;height:100%;"></div> </div> '
        '<div class="relative z-10 flex flex-col items-center justify-center min-h-screen"> '
        '<div class="wanderer-quotes-wrap" style="max-width:860px;width:100%;padding:4rem 1rem 0 clamp(1rem,18vw,20rem)"> '
        + ''.join(quotes) + ' </div> '
        + section_title_block('/wanderer/', 'WANDERER', '行 者')
        + more_button('/wanderer/') + ' </div> </section>')

    # --- 诗人区 ---
    pg = HOME['poet_gallery']
    plines = []
    for ln in pg['lines']:
        plines.append(
            '<div class="scroll-reveal" style="font-family:\'FZXingKai\',\'LXGW WenKai\',serif;'
            'font-size:clamp(2.4rem,5vw,3.5rem);line-height:2.4;opacity:0;transform:translateY(50px);'
            'filter:blur(2px);transition:opacity 0.9s ease-out,transform 0.9s ease-out,filter 0.9s ease-out;'
            f'margin-bottom:2.5rem;text-align:center"><p>{ln}</p></div>')
    poet = (
        '<section id="poet-gallery" class="relative py-[4.6rem] md:py-44 text-pampas contents-full"> '
        '<div class="absolute left-0 top-0 h-full w-full bg-mine-shaft-texture" style="background-color:#4A3E35"></div> '
        '<style>@keyframes openingFlash{0%{opacity:0;transform:translateY(20px);filter:blur(6px)}'
        '15%{opacity:0.45;transform:translateY(0);filter:blur(0)}70%{opacity:0.45}'
        '100%{opacity:0;transform:translateY(-12px);filter:blur(2px)}}'
        '.opening-flash.is-visible{animation:openingFlash 2.8s ease-in-out forwards} '
        '@media(max-width:767px){.poet-gallery-head{padding:2.5rem 1.25rem 0}} </style> '
        '<div class="poet-gallery-head relative z-10 flex md:justify-end md:pr-[20%]"> '
        '<h2 class="ts-focus-in data-[active=true]:animate-text-focus-in text-current flex"> '
        '<a href="/poet/" class="md:vertical-rl grid gap-2 md:gap-4" '
        'style="text-decoration:none;color:var(--color-taupe-gray,#B4AC97)"> '
        '<span class="md:text-xl leading-none font-medium" style="font-family:Cinzel,serif;letter-spacing:0.15em">POET</span> '
        "<span class=\"leading-none font-medium\" style=\"font-family:'Zhi Mang Xing',cursive;"
        'font-size:clamp(2.5rem,6vw,5rem)">诗集</span> </a> </h2> </div> '
        '<div class="relative z-10 flex flex-col items-center justify-center min-h-screen"> '
        '<div style="max-width:760px;width:100%;padding:2rem 2rem 0"> '
        f'<p class="opening-flash" style="font-family:\'Zhi Mang Xing\',cursive;font-size:clamp(2.4rem,5vw,3.5rem);'
        f'opacity:0;text-align:center;letter-spacing:0.3em;color:var(--color-taupe-gray,#B4AC97)">{pg["label"]}</p> '
        f'<p style="font-family:\'Zhi Mang Xing\',cursive;font-size:clamp(2.4rem,5vw,3.5rem);text-align:center;'
        f'margin-top:4rem;margin-bottom:0.4rem;color:var(--color-pampas,#F9F9F6)">{pg["title"]}</p> '
        '<p style="font-family:\'LXGW WenKai\',serif;font-size:0.8rem;letter-spacing:0.2em;opacity:0.35;'
        f'margin-bottom:8rem;text-align:center">{pg["date"]}</p> '
        + ''.join(plines) + ' </div> '
        + section_title_block('/poet/', 'POET', '诗人')
        + more_button('/poet/') + ' </div> </section>')

    # --- 赤子区 ---
    cards_html = []
    for c in HOME['enthusiast_cards']:
        cards_html.append(
            f'<a href="/enthusiast/#{c["anchor"]}" class="enthusiast-card card-{c["color"]}"> '
            '<span style="font-size:1.8rem;opacity:0.5">' + c['emoji'] + '</span> '
            '<span style="font-family:Cinzel,serif;font-size:0.75rem;font-weight:600;letter-spacing:0.18em;'
            f'opacity:0.45" class="card-en">{c["en"]}</span> '
            "<span style=\"font-family:'Zhi Mang Xing',cursive;font-size:clamp(1.4rem,2.5vw,1.9rem);opacity:0.8\">"
            f'{c["zh"]}</span> '
            '<span style="font-family:\'LXGW WenKai\',serif;font-size:0.8rem;opacity:0.5;text-align:center;'
            f'line-height:1.5">{c["sub"]}</span> </a>')
    enthusiast = (
        '<section id="enthusiast-gallery"> <div class="relative md:h-screen md:overflow-hidden py-[4.6rem] '
        'text-pampas contents-full md:py-44"> '
        '<div class="absolute left-0 top-0 h-full w-full bg-mine-shaft-texture" style="background-color:#4E3E2C"></div> '
        '<style> .enthusiast-card{background:rgba(249,249,246,0.04);backdrop-filter:blur(12px);'
        '-webkit-backdrop-filter:blur(12px);border:1px solid rgba(249,249,246,0.08);border-radius:8px;'
        'padding:1.5rem 1rem 1.6rem;transition:all 0.4s ease;cursor:pointer;display:flex;flex-direction:column;'
        'align-items:center;gap:0.6rem;text-decoration:none;color:inherit;'
        'box-shadow:inset 0 0 0 1px rgba(249,249,246,0.03)} '
        '.enthusiast-card:hover{background:rgba(249,249,246,0.08);transform:translateY(-2px)} '
        '.card-en{transition:color 0.4s ease} '
        '.card-cinema:hover .card-en{color:rgba(200,160,80,0.7)} '
        '.card-music:hover .card-en{color:rgba(130,160,210,0.7)} '
        '.card-books:hover .card-en{color:rgba(180,150,110,0.7)} '
        '.card-sport:hover .card-en{color:rgba(110,180,140,0.7)} '
        '.card-food:hover .card-en{color:rgba(220,140,80,0.7)} '
        '.enthusiast-cards-grid{display:flex;flex-wrap:wrap;justify-content:center;gap:2.4rem;max-width:1120px;'
        'margin:0 auto} .enthusiast-card{flex:0 0 auto;width:clamp(155px,18vw,190px)} '
        '@media(max-width:767px){.enthusiast-cards-grid{gap:1rem}.enthusiast-card{width:clamp(110px,30vw,155px);'
        'padding:1.2rem 0.8rem 1.4rem}} '
        '.enthusiast-center{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);z-index:20;'
        'display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;width:100vw} '
        '.enthusiast-center-inner{display:flex;flex-direction:column;align-items:center;gap:7rem} '
        '@media(max-width:767px){.enthusiast-center{position:static;transform:none;height:auto;width:100%;'
        'padding:3rem 0}.enthusiast-center-inner{gap:3.5rem;padding:0 1rem}} </style> '
        '<div class="enthusiast-center"> <div class="enthusiast-center-inner"> '
        '<div class="enthusiast-cards-grid">' + ''.join(cards_html) + '</div> '
        + section_title_block('/enthusiast/', 'ENTHUSIAST', '赤 子')
        + more_button('/enthusiast/') + ' </div> </div> </div> </section>')

    # --- hero ---
    hero = (
        '<section id="hero" class="relative" data-astro-cid-5saot5ic> '
        '<div id="hero-container" data-astro-cid-5saot5ic> '
        '<div class="flex h-lvh flex-col justify-center overflow-y-hidden py-16 text-silver-chalice '
        'bg-mine-shaft-texture contents-full" data-astro-cid-5saot5ic> '
        '<div class="flex flex-col justify-center lg:grid lg:grid-cols-3 lg:items-end" data-astro-cid-5saot5ic>  '
        '<p class="hidden font-serif-en md:block" style="position:absolute;right:clamp(1rem,5vw,4rem);top:50%;'
        'transform:translateY(-50%);max-width:26rem;z-index:5;font-family:Cinzel,\'LXGW WenKai\',serif;'
        f'font-size:0.85rem;line-height:1.8" data-astro-cid-5saot5ic>\n{h["en_paragraph"]}\n</p> </div> '
        '<div id="hero-video" class="is-loading absolute left-1/2 top-0 z-0 h-full w-screen -translate-x-1/2 '
        'overflow-hidden transition-all duration-700 [&.is-loading]:z-50" data-astro-cid-5saot5ic> '
        '<video poster="/top/hero/shadow-image.jpg" webkit-playsinline preload="auto" playsinline muted autoplay loop '
        'class="h-full w-full object-cover" data-astro-cid-5saot5ic> '
        '<source src="/top/hero/shadow-movie.mp4" type="video/mp4" data-astro-cid-5saot5ic> </video> </div> '
        '<p class="mt-auto font-serif-en md:text-xl" style="position:relative;z-index:1;'
        f'margin-left:clamp(1rem,5vw,5rem)" data-astro-cid-5saot5ic>\n{h["tags_line"]}\n</p> '
        '<h1 class="mt-3 flex items-end justify-between gap-4 md:mt-0 md:justify-normal md:gap-10" '
        'data-astro-cid-5saot5ic> '
        '<h2 class="font-splash text-white leading-tight" style="position:relative;z-index:1;'
        "font-family:'Zhi Mang Xing',cursive;font-size:clamp(3.2rem,11vw,9rem);letter-spacing:0.12em;"
        'text-shadow:0 0 60px rgba(0,0,0,.5),0 2px 8px rgba(0,0,0,.3);line-height:1.3;margin-top:0.6em;'
        f'margin-left:clamp(0.5rem,3vw,3rem)">{title_join}</h2> '
        f'<span style="writing-mode:vertical-rl;color:rgba(255,255,255,0.8);font-family:Cinzel,serif;'
        f'font-size:clamp(0.7rem,1.5vw,0.95rem);letter-spacing:0.1em" data-astro-cid-5saot5ic>{h["vertical_label"]}</span> '
        '</h1> </div> '
        '<div class="scroll-nav absolute bottom-[5px] right-0 z-0 hidden h-[17.5rem] w-[1px] bg-emperor md:block" '
        'data-astro-cid-5saot5ic> <div class="relative h-full w-full" data-astro-cid-5saot5ic> '
        '<div class="scroll-nav__inside absolute left-0 top-0 h-10 w-full bg-silver-chalice" '
        'data-astro-cid-5saot5ic></div> </div> </div> </div> </section> ')

    title_join = '<br>'.join(h['title_lines'])

    content = ('<main id="main" class="container px-[1.4rem] md:px-20 overflow-x-hidden false">      '
               + hero + '   ' + parallax + '  ' + works + '  '
               + snippet('astro_island.html') + ' '
               + wanderer + poet + enthusiast + '   ')

    extra_head = (snippet('home_head_preloads.html').replace('/sitemap-index.xml', '/sitemap.xml')
                  + '\n' + GTAG_BLOCK
                  + '\n' + FONT_PRELOADS_ALL
                  + '\n<link rel="preload" href="/_astro/about.j05OPDV1.css" as="style">'
                    '<link rel="preload" href="/_astro/about.Cl2ZlrCQ.css" as="style">'
                    '<link rel="preload" href="/_astro/font-override.css" as="style">'
                    '<link rel="prefetch" href="/about/"><link rel="prefetch" href="/poet/">'
                    '<link rel="prefetch" href="/wanderer/"><link rel="prefetch" href="/enthusiast/">'
                    '<link rel="prefetch" href="/builder/">\n'
                  + snippet('home_head_tail.html'))

    body = (' <body id="body" class="overflow-x-hidden font-serif text-taupe-gray transition-colors '
            'data-[bg-color=bright]:text-black" data-astro-cid-nj3qennt> '
            + snippet('home_splash.html')
            + ' <div id="bg" class="bg fixed left-0 top-0 -z-10 h-full w-screen" data-color="dark" '
            'data-astro-cid-nj3qennt> <div class="relative h-full w-full" data-astro-cid-nj3qennt> '
            '<div class="bg__dark absolute left-0 top-0 h-full w-full bg-mine-shaft-texture" '
            'data-astro-cid-nj3qennt></div> <div class="bg__bright absolute left-0 top-0 h-full w-full '
            'bg-gray-texture" data-astro-cid-nj3qennt></div> </div> </div> ')

    page = render_head('William Liu / HAO-Z', SITE.get('seo_description', 'William Liu -- Biomedical engineering student, AI researcher, and creative technologist based in Wuhan.'), '/',
                       extra_head=extra_head, html_attr=' data-astro-cid-nj3qennt') + body \
        + render_chrome('dark', 'true') + ' ' + content + '</main> ' + render_footer() \
        + '\n' + snippet('home_scripts.html') + '\n' + VIDEO_AUTOPLAY_SCRIPT \
        + '</body></html>'
    write('index.html', page)

# ---------------------------------------------------------- 行者区

def render_wanderer():
    w = WANDERER
    blocks = []
    for pl in w['places']:
        photos = ''.join(f'<img src="{ph["src"]}" alt="{ph["alt"]}">' for ph in pl['photos'])
        blocks.append(
            '    <div class="place-block">\n'
            f'      <p class="section-title">{pl["en"]}</p>\n'
            f'      <div class="place-title">{pl["title"]}</div>\n'
            f'      <div class="place-sub">{pl["sub"]}</div>\n'
            f'      <p class="place-text">{pl["text"]}</p>\n'
            f'      <div class="place-photos {pl.get("layout", "single")}">\n        {photos}\n      </div>\n'
            '    </div>\n')
    hl = '<br>'.join(w['hero']['title_lines'])
    content = f'''<main>
  <section class="wanderer-hero">
    <img fetchpriority="high" src="{w['hero']['cover']}" alt="行者封面" class="wanderer-hero-img">
    <div class="wanderer-hero-content">
      <p style="font-family:'Zhi Mang Xing',cursive;font-size:clamp(3.2rem,11vw,9rem);line-height:1.3;letter-spacing:0.12em;text-shadow:0 0 60px rgba(0,0,0,.5),0 2px 8px rgba(0,0,0,.3)">{hl}</p>
    </div>
  </section>

  <div style="max-width:900px;margin:0 auto">

{''.join(blocks)}
  </div>

  <div style="text-align:center;padding:4rem 2.5rem;border-top:1px solid rgba(249,249,246,0.08)">
    <a href="/" style="font-family:Cinzel,serif;font-size:0.8rem;letter-spacing:0.2em;opacity:0.35;text-decoration:none;color:inherit;transition:opacity 0.3s" onmouseover="this.style.opacity=0.7" onmouseout="this.style.opacity=0.35">← BACK TO HOME</a>
  </div>
</main>'''
    style = page_style('wanderer_style.css')
    extra_head = (f'<link rel="preload" as="image" href="{w["hero"]["cover"]}" fetchpriority="high">'
                  + FONT_PRELOAD_ZMX
                  + WF_READY_SCRIPT
                  + f'<style>\n{style}</style>')
    body_attr = ' style="background:#3E4337;color:#f9f9f6;overflow-x:hidden"'
    pre = ('<div class="page-bg" style="background-color:#3E4337;'
           'background-image:url(/common/texture-for-black.png);background-repeat:repeat"></div>')
    tail = ('\n' + '<script type="module" src="/_astro/hoisted.BAJWDJOX.js"></script>'
            '<script type="module" src="/_astro/page.LS5KDvwX.js"></script>'
            + render_footer_scripts() + '\n' + VIDEO_AUTOPLAY_SCRIPT)
    page = (render_head('行者 | William Liu', 'William Liu 的旅途记录。行于山野，止于心间。', '/wanderer/',
                        extra_head=extra_head)
            + f'<body{body_attr}>\n' + pre + '\n\n'
            + render_chrome('bright', 'false', ' style="color:#3D3D3D"') + '\n\n'
            + content + ' '
            + render_footer() + tail + '</body></html>')
    write('wanderer/index.html', page)

# ---------------------------------------------------------- 诗人区

def render_poem_card(pid, featured=False):
    poem = POEMS['poems'][pid]
    date_html = f'<span class="poem-card__date">{poem["date"]}</span>' if poem.get('date') else ''
    feat = ' poem-card--featured' if featured else ''
    excerpt = poem.get('excerpt') or poem['lines'].split('\\n')[0]
    return (f'''      <div class="poem-card{feat}" data-poem-id="{pid}" onclick="openPoem(this)">
        <div class="poem-card__header"><span class="poem-card__title">{poem['title']}</span>{date_html}</div>
        <p class="poem-card__excerpt">{excerpt}</p>
        <div class="poem-card__hint">展开阅读 · Click to read</div>
      </div>''')

def render_poet():
    pc = POEMS
    # 开篇诗
    opening_lines = '\n'.join(f'    <p class="opening-line">{ln}</p>' for ln in pc['opening']['lines'])
    opening = f'''  <section class="opening-section">
    <div class="opening-label">{pc['opening']['label']}</div>
    <p class="opening-title">{pc['opening']['title']}</p>
    <p class="opening-date">{pc['opening']['date']}</p>
{opening_lines}
  </section>'''
    # 照片+诗行
    rows = []
    for row in pc['photo_poem_rows']:
        side_cls = ' photo-poem-row--img-right' if row['image_side'] == 'right' else ''
        stack = '\n'.join(render_poem_card(pid) for pid in row['poems'])
        rows.append(f'''  <div class="photo-poem-row{side_cls}">
    <div class="photo-breather">
      <img src="{row['image']}" alt="{row['alt']}">
      <span class="photo-breather__caption">{row['caption']}</span>
    </div>
    <div class="poem-stack">
{stack}
    </div>
  </div>''')
    # 诗选
    zone_stack = '\n'.join(render_poem_card(pid) for pid in pc['poetry_zone']['poems'])
    zone = f'''  <div class="poetry-zone">
    <div class="poem-grid">
{zone_stack}
    </div>
  </div>'''
    # 长诗
    stanzas = '\n'.join(
        f'    <div class="long-poem-stanza">\n      <p>{s}</p>\n    </div>'
        for s in pc['long_poem']['stanzas'])
    long_poem = f'''  <section class="long-poem-section">
    <div class="long-poem-divider"></div>
    <div class="long-poem-label">Long Poem</div>
    <h2 class="long-poem-title">{pc['long_poem']['title']}</h2>

{stanzas}
  </section>'''
    # poems JS 对象（yaml 里的字面 \n 还原成真换行，再交给 json 转义）
    poems_obj = {}
    for pid, v in pc['poems'].items():
        poems_obj[pid] = {'title': v['title'], 'date': v.get('date', ''),
                          'lines': v['lines'].replace('\\n', '\n')}
    poems_js = json.dumps(poems_obj, ensure_ascii=False, indent=2)
    hl = '<br>'.join(pc['hero']['title_lines'])
    rows_before = chr(10).join(rows[:3])
    rows_after = rows[3]
    content = f'''<main>
  <section class="poet-hero">
    <img fetchpriority="high" src="{pc['hero']['cover']}" alt="诗人封面" class="poet-hero-img">
    <div class="poet-hero-content">
      <p style="font-family:'Zhi Mang Xing',cursive;font-size:clamp(2.5rem,10vw,9rem);line-height:1.3;letter-spacing:0.15em;text-shadow:0 0 40px rgba(0,0,0,.6),0 2px 6px rgba(0,0,0,.4)">{hl}</p>
    </div>
  </section>

<!-- ===== 开篇诗：{pc['opening']['title']} ===== -->
{opening}

{rows_before}

{zone}

{rows_after}

<!-- ===== 长诗：{pc['long_poem']['title']} ===== -->
{long_poem}

  <!-- Back to home -->
  <div class="back-link">
    <a href="/">← BACK TO HOME</a>
  </div>

  <!-- ===== 诗歌 Modal Overlay ===== -->
  <div class="poem-overlay" id="poemOverlay" onclick="closePoem()">
    <div class="poem-overlay__bg"></div>
    <div class="poem-overlay__modal" onclick="event.stopPropagation()">
      <button class="poem-overlay__close" onclick="closePoem()">&times;</button>
      <div class="poem-overlay__title" id="modalTitle"></div>
      <div class="poem-overlay__date" id="modalDate"></div>
      <div class="poem-overlay__text" id="modalText"></div>
      <div class="poem-overlay__footer" onclick="closePoem()">▲ 关闭 · Close</div>
    </div>
  </div>
</main>'''
    style = page_style('poet_style.css')
    extra_head = (f'<link rel="preload" as="image" href="{pc["hero"]["cover"]}" fetchpriority="high">'
                  + FONT_PRELOAD_ZMX + WF_READY_SCRIPT + f'<style>\n{style}</style>')
    tail = ('\n<script>\nvar poems = ' + poems_js + ';\n\n' + snippet('poem_logic.js.html')
            + '\n</script>\n'
            + '<script type="module" src="/_astro/hoisted.BAJWDJOX.js"></script>'
            '<script type="module" src="/_astro/page.LS5KDvwX.js"></script>'
            + render_footer_scripts() + '\n' + VIDEO_AUTOPLAY_SCRIPT)
    page = (render_head('诗人 | William Liu', 'William Liu 的诗集。文字是他拿来对抗时间的方式。', '/poet/',
                        extra_head=extra_head)
            + '<body style="background:#4A3E35;color:#f9f9f6;overflow-x:hidden">\n'
            + '<div class="page-bg bg-mine-shaft-texture" style="background-color:#4A3E35"></div>\n\n'
            + render_chrome('dark', 'false', ' style="color:#f9f9f6"') + '\n\n'
            + content + ' '
            + render_footer() + tail + '</body></html>')
    write('poet/index.html', page)

# ---------------------------------------------------------- 赤子区

def render_two_col(items, start=1):
    out = []
    for i, it in enumerate(items, start):
        note = f'<span class="two-col-note">{it["note"]}</span>' if it.get('note') else ''
        out.append(f'        <div class="two-col-item"><span class="two-col-idx">{i:02d}</span>'
                   f'<span class="two-col-name">{it["name"]}</span>{note}</div>')
    return '\n'.join(out)

def render_enthusiast():
    e = ENTHUSIAST
    cin = e['cinema']
    films = '\n'.join(
        f'      <div><img src="{f["src"]}" alt="{f["alt"]}"><span class="film-caption">{f["caption"]}</span></div>'
        for f in cin['films'])
    movies_l = render_two_col(cin['movies'], 1)
    tv = '\n'.join(f'        <div class="two-col-item"><span class="two-col-name">{t}</span></div>'
                   for t in cin['tv_series'])
    tri = '\n'.join(f'        <div class="two-col-item"><span class="two-col-name">{t}</span></div>'
                    for t in cin['trilogies'])
    dirs = '\n'.join(f'        <div class="two-col-item"><span class="two-col-name">{t}</span></div>'
                     for t in cin['directors'])
    m_half = (len(e['music']['items']) + 1) // 2
    music_l = render_two_col(e['music']['items'][:m_half], 1)
    music_r = render_two_col(e['music']['items'][m_half:], m_half + 1)
    b_half = (len(e['books']['items']) + 1) // 2
    books_l = render_two_col(e['books']['items'][:b_half], 1)
    books_r = render_two_col(e['books']['items'][b_half:], b_half + 1)
    food_imgs = '\n'.join(
        f'        <div class="ts-slider-food-item slider__item"><img src="{f["src"]}" alt="{f["alt"]}" '
        'class="h-full w-full object-cover"></div>' for f in e['food']['images'])
    sport_cards = '\n'.join(
        f'      <div class="sport-card{" dashed" if c["dashed"] else ""}"><h4>{c["title"]}</h4>'
        f'<p>{c["text"]}</p></div>' for c in e['sport']['cards'])
    hl = '<br>'.join(e['hero']['title_lines'])
    content = f'''<main>
  <section class="hero">
    <img fetchpriority="high" src="{e['hero']['cover']}" alt="赤子封面" class="hero-img">
    <div class="hero-content">
      <p style="font-family:'Zhi Mang Xing',cursive;font-size:clamp(3.2rem,11vw,9rem);line-height:1.3;letter-spacing:0.12em;text-shadow:0 0 60px rgba(0,0,0,.5),0 2px 8px rgba(0,0,0,.3)">{hl}</p>
    </div>
  </section>

  <!-- CINEMA -->
  <div class="chapter-outer" id="cinema" style="background:rgba(62,67,55,0.3)"><div class="chapter-inner">
    <span class="section-label">Cinema &amp; Series</span>
    <div class="chapter-title">CINEMA</div>
    <span class="chapter-zh">电影·剧集</span>
    <p class="body-text">{cin['text']}</p>

    <p style="font-family:Cinzel,serif;font-size:0.7rem;letter-spacing:0.2em;opacity:0.25;margin:2.5rem 0 0.8rem">FILMS &amp; DIRECTORS</p>
    <div class="film-strip">
{films}
    </div>

    <div class="two-col" style="margin-top:2.5rem;padding-top:2rem;border-top:1px solid rgba(249,249,246,0.06)">
      <div>
        <div class="two-col-item" style="opacity:0.35;font-family:Cinzel;font-size:0.7rem;letter-spacing:0.1em;border-bottom:none;padding-bottom:0.3rem"><span class="two-col-idx"></span><span class="two-col-name">MOVIES</span></div>
{movies_l}
      </div>
      <div>
        <div class="two-col-item" style="opacity:0.35;font-family:Cinzel;font-size:0.7rem;letter-spacing:0.1em;border-bottom:none;padding-bottom:0.3rem"><span class="two-col-idx"></span><span class="two-col-name">TV SERIES</span></div>
{tv}
        <div class="two-col-item" style="opacity:0.35;font-family:Cinzel;font-size:0.7rem;letter-spacing:0.1em;border-bottom:none;padding:0.8rem 0 0.3rem"><span class="two-col-idx"></span><span class="two-col-name">TRILOGIES</span></div>
{tri}
        <div class="two-col-item" style="opacity:0.35;font-family:Cinzel;font-size:0.7rem;letter-spacing:0.1em;border-bottom:none;padding:0.8rem 0 0.3rem"><span class="two-col-idx"></span><span class="two-col-name">DIRECTORS</span></div>
{dirs}
      </div>
    </div>
  </div></div>

  <!-- MUSIC -->
  <div class="chapter-outer" id="music"><div class="chapter-inner">
    <span class="section-label">Music</span>
    <div class="chapter-title">MUSIC</div>
    <span class="chapter-zh">音乐</span>
    <p class="body-text">{e['music']['text']}</p>
    <div class="two-col">
      <div>
{music_l}
      </div>
      <div>
{music_r}
      </div>
    </div>
  </div></div>

  <!-- BOOKS -->
  <div class="chapter-outer" id="books" style="background:rgba(62,67,55,0.3)"><div class="chapter-inner">
    <span class="section-label">Books</span>
    <div class="chapter-title">BOOKS</div>
    <span class="chapter-zh">书籍</span>
    <p class="body-text">{e['books']['text']}</p>
    <div class="two-col">
      <div>
{books_l}
      </div>
      <div>
{books_r}
      </div>
    </div>
  </div></div>

  <!-- FOOD -->
  <div class="chapter-outer" id="food" style="padding-bottom:2rem"><div class="chapter-inner">
    <span class="section-label">Food</span>
    <div class="chapter-title">FOOD</div>
    <span class="chapter-zh">美食</span>
    <p class="body-text">{e['food']['text']}</p>
  </div></div>

  <div id="food-slider-wrapper" style="max-width:1000px;margin:0 auto;position:relative;height:110vh">
    <div id="food-slider" style="position:absolute;top:0;left:0;width:66%">
      <div class="ts-slider-food slider" style="aspect-ratio:1200/1014;width:100%">
{food_imgs}
      </div>
    </div>
  </div>
  {snippet('enthusiast_food_script.html')}

  <!-- SPORT -->
  <div class="chapter-outer" id="sport" style="background:rgba(62,67,55,0.3)"><div class="chapter-inner">
    <span class="section-label">Sport</span>
    <div class="chapter-title">SPORT</div>
    <span class="chapter-zh">运动</span>
    <p class="body-text">{e['sport']['text']}</p>
    <div class="sport-grid" style="margin-top:2rem">
{sport_cards}
    </div>
  </div></div>

  <div style="text-align:center;padding:4rem 2.5rem;border-top:1px solid rgba(249,249,246,0.08)">
    <a href="/" style="font-family:Cinzel,serif;font-size:0.8rem;letter-spacing:0.2em;opacity:0.35;text-decoration:none;color:inherit;transition:opacity 0.3s" onmouseover="this.style.opacity=0.7" onmouseout="this.style.opacity=0.35">← BACK TO HOME</a>
  </div>
</main>'''
    style = page_style('enthusiast_style.css')
    extra_head = (f'<link rel="preload" as="image" href="{e["hero"]["cover"]}" fetchpriority="high">'
                  + FONT_PRELOAD_ZMX + WF_READY_SCRIPT + f'<style>\n{style}</style>')
    tail = ('\n' + '<script type="module" src="/_astro/hoisted.BAJWDJOX.js"></script>'
            '<script type="module" src="/_astro/page.LS5KDvwX.js"></script>'
            + render_footer_scripts() + '\n' + VIDEO_AUTOPLAY_SCRIPT)
    page = (render_head('赤子 | William Liu', 'William Liu 的热爱。运动、美食、电影、音乐、书籍。', '/enthusiast/',
                        extra_head=extra_head)
            + '<body style="background:#4E3E2C;color:#f9f9f6;overflow-x:hidden">\n'
            + '<div class="page-bg" style="background-color:#4E3E2C;background-image:url(/common/texture-for-black.png);background-repeat:repeat"></div>\n\n'
            + render_chrome('dark', 'false', ' style="color:#f9f9f6"') + '\n\n'
            + content + ' '
            + render_footer() + tail + '</body></html>')
    write('enthusiast/index.html', page)

# ---------------------------------------------------------- 述己

def render_about():
    a = ABOUT
    slider_items = []
    for i, s in enumerate(a['slider']):
        prio = ' fetchpriority="high"' if i == 0 else ''
        slider_items.append(f'''<div class="ts-slider-item slider__item" data-astro-cid-pf7ldbt4> <picture class="block h-full w-full object-cover" data-astro-cid-pf7ldbt4> <source srcset="{s['src']}" media="(min-width: 768px)" class="h-full w-full object-cover" data-astro-cid-pf7ldbt4> <img{prio} src="{s['src']}" alt="{s['alt']}" class="h-full w-full object-cover" data-astro-cid-pf7ldbt4> </picture> </div>''')
    timeline_items = []
    for t in a['timeline']:
        timeline_items.append(f'''<li class="relative flex flex-col gap-6 border-t border-emperor pr-24 pt-6"> <span class="absolute -top-[calc(0.75rem/2)] left-0 block h-3 w-3 rounded-full bg-taupe-gray"></span> <time datetime="{t['time']}" class="font-serif-en text-taupe-gray" style="font-size:clamp(0.85rem,1.4vw,1.05rem)">{t['time']}</time> <span class="font-medium" style="font-family:'Zhi Mang Xing',cursive;color:var(--color-pampas,#F9F9F6);font-size:clamp(1.3rem,2.5vw,2rem)">{t['city']} <span style="font-size:0.6em;color:var(--color-taupe-gray,#B4AC97)">{t['city_sub']}</span></span> <p class="w-[50vw] whitespace-normal text-silver-chalice md:w-72" style="font-size:clamp(0.85rem,1.3vw,1rem)">{t['desc']}</p> </li>''')
    prof = a['profile']
    profile_lines = '<br data-astro-cid-kh7btl4r>\n'.join(prof['lines'])
    content = f'''   <div class="pt-40 text-taupe-gray" data-astro-cid-kh7btl4r> <h1 class="px-[1.4rem] pt-16 md:px-0 md:pt-32"><span class="-mt-5 align-top text-4xl font-medium leading-none vertical-rl md:text-5xl" style="font-family:'Zhi Mang Xing',cursive"><span>述己</span></span><span class="-mt-5 ml-2 align-top font-serif-en vertical-rl md:ml-[0.8rem] md:text-xl md:leading-none">(<!-- -->ABOUT<!-- -->)</span></h1> </div> <div id="about-text" class="ml-auto mt-16 w-9/12 md:mt-36 md:text-xl" data-astro-cid-kh7btl4r> <p data-astro-cid-kh7btl4r>
{a['bio_zh']}
</p> <p class="mt-7 text-silver-chalice md:mt-12" data-astro-cid-kh7btl4r style="font-family:'LXGW WenKai',serif;font-size:clamp(0.9rem,1.5vw,1.1rem)">
{a['bio_en']}
</p> </div> <div id="about-slider-wrapper" class="relative h-screen w-full pt-16 md:pt-[7.5rem]" data-astro-cid-kh7btl4r> <div id="about-slider" class="about-slider absolute right-0 top-16 h-40 md:top-[7.5rem] md:h-auto" data-astro-cid-kh7btl4r> <div class="ts-slider slider h-full w-full md:aspect-video" data-astro-cid-pf7ldbt4> {''.join(slider_items)} </div>   </div> </div> <section class="mt-32 md:mt-60"> <h2 class="mb-9 font-serif-en text-lg text-taupe-gray md:mb-14 md:text-xl">
(PROFILE)
</h2>  <div class="flex flex-col items-center justify-between gap-9 md:flex-row md:gap-60" data-astro-cid-kh7btl4r> <div class="ts-profile-image relative max-w-80 overflow-hidden md:max-w-[32.5rem]" data-astro-cid-kh7btl4r> <picture class="block w-full mix-blend-multiply" data-astro-cid-kh7btl4r> <img src="{prof['image']}" alt="{prof['name']}的个人资料图片" class="w-full" data-astro-cid-kh7btl4r> </picture> <div class="bg-layer absolute -bottom-1/2 left-0 -z-10 h-[200%] w-full bg-taupe-gray" data-astro-cid-kh7btl4r></div> </div> <div class="flex-1" data-astro-cid-kh7btl4r> <p class="font-serif-en text-sm text-taupe-gray md:text-base" data-astro-cid-kh7btl4r>
{prof['name_en']}
</p> <p class="mt-2 font-medium" style="font-size:clamp(1.5rem,4vw,2.8rem)" data-astro-cid-kh7btl4r>{prof['name']}</p> <p class="mt-7 text-silver-chalice md:mt-10" style="font-size:clamp(0.95rem,1.6vw,1.15rem)" data-astro-cid-kh7btl4r>
{profile_lines}
</p> {snippet('about_social_icons.html')}<div id="qr-modal" style="display:none;position:fixed;inset:0;z-index:100;background:rgba(0,0,0,0.7);backdrop-filter:blur(4px);align-items:center;justify-content:center" onclick="this.style.display='none'"><div style="background:#3a3028;border-radius:16px;padding:24px;box-shadow:0 20px 60px rgba(0,0,0,0.5)" onclick="event.stopPropagation()"><img src="{prof['wechat_qr']}" alt="微信二维码" style="width:200px;height:200px;object-fit:contain;border-radius:12px"><p style="text-align:center;margin-top:12px;font-family:'LXGW WenKai',serif;color:#B4AC97;font-size:14px">{prof['wechat_id']}</p></div></div> </div> </div>  </section> <section class="mt-32 md:mt-60"> <h2 class="mb-9 font-serif-en text-lg text-taupe-gray md:mb-14 md:text-xl">
(EDUCATION)
</h2>  <div class="ts-drag-scroll relative w-full select-none"> <ul class="ts-drag-scroll-container flex flex-nowrap overflow-x-scroll whitespace-nowrap pt-[calc(0.75rem/2)] contents-right md:overflow-x-hidden">
{chr(10).join(timeline_items)}
</ul> </div>   </section> '''
    content = ('<main id="main" class="container px-[1.4rem] md:px-20 md:max-w-[calc(85rem+10rem)]">   ' + content
               + snippet('astro_island.html') + '   ')
    style = page_style('about_style.css')
    extra_head = ('<link rel="preload" as="image" href="' + ABOUT['slider'][0]['src'] + '" fetchpriority="high">'
                  + f'<style>{style}</style>'
                  + WF_READY_SCRIPT
                  + '\n<script type="module" src="/_astro/hoisted.BAJWDJOX.js"></script>'
                    '<script type="module" src="/_astro/page.LS5KDvwX.js"></script>'
                  + '\n' + GTAG_BLOCK)
    tail = ('\n<script>document.getElementById(\'wechat-link\').addEventListener(\'click\',function(e){{e.preventDefault();document.getElementById(\'qr-modal\').style.display=\'flex\';}});</script>'.replace('{{', '{').replace('}}', '}')
            + render_footer_scripts() + '\n' + VIDEO_AUTOPLAY_SCRIPT)
    page = (render_head('ABOUT | William Liu', 'HAO-Z -- Biomedical engineering student, AI researcher, and creative technologist based in Wuhan.', '/about/',
                        extra_head=extra_head)
            + '<body class="font-serif text-pampas bg-mine-shaft-texture overflow-x-hidden" data-bg-color="dark" style="background-color:#4A3D35">    '
            + render_chrome('dark', 'false') + '    '
            + content + '</main> ' + render_footer() + tail + '</body></html>')
    write('about/index.html', page)

# ---------------------------------------------------------- 创客区列表页

def render_builder_list():
    lis = []
    for p in PROJECTS:
        cats = ''.join(f'<li>{c}</li>' for c in p.get('categories', []))
        tags = ''.join(f'<li>{t}</li>' for t in p.get('tags', []))
        lis.append(
            '            <li\n'
            '              class="aspect-works-slide h-full max-w-full animate-text-focus-in whitespace-nowrap '
            'border-l border-silver p-[1.4rem] transition-all duration-[1500ms] last-of-type:border-r '
            'peer-last:border-r data-[hidden]:max-w-0 data-[hidden]:animate-text-blur-out data-[hidden]:px-0 '
            'md:px-12 md:pb-6 md:pt-5"\n'
            f'              data-tags="{" ".join(p.get("tags", []))}"\n'
            '            >\n'
            f'                  <a\n                    href="/builder/{p["slug"]}/"\n'
            '                    class="ts-image-link flex h-full flex-col"\n                  >\n'
            '                    <div class="flex">\n'
            '                      <picture class="flex-1">\n'
            f'                        <img\n                          src="{p.get("list_image") or p["hero"]}"\n'
            f'                          alt="{p["title"]}"\n'
            '                          class="h-full w-full object-cover"\n                        />\n'
            '                      </picture>\n'
            '                      <div class="ml-3 flex font-serif-en vertical-rl">\n'
            f'                        <ul class="flex flex-wrap gap-4">{cats}</ul>\n'
            f'                        <span class="mt-auto inline-block">{p.get("year","")}</span>\n'
            '                      </div>\n'
            '                    </div>\n'
            '                    <div class="mt-4 flex-1 pr-8">\n'
            '                      <h2\n'
            '                        class="overflow-hidden text-ellipsis whitespace-normal text-xl font-medium '
            'md:text-[1.75rem]"\n'
            '                        style="display: -webkit-box; -webkit-line-clamp: 2; '
            '-webkit-box-orient: vertical"\n'
            f'                      >\n                        {p["title"]}\n                      </h2>\n'
            '                      <ul class="mt-6 flex min-h-[2.2rem] flex-wrap gap-x-4 gap-y-[0.2rem] '
            f'font-serif-en text-gray">\n                        {tags}\n                      </ul>\n'
            '                    </div>\n                  </a>\n            </li>')
    filters = SITE.get('builder_filters', ['ALL', 'INTERNSHIP', 'PERSONAL PROJECT', 'COMPETITION', 'COURSE PROJECT'])
    filter_lis = []
    for f in filters:
        checked = ' checked' if f == 'ALL' else ''
        dc = ' data-checked="true"' if f == 'ALL' else ''
        filter_lis.append(
            f'''                  <li
                    data-checked{dc}
                    class="invisible h-0 w-0 animate-text-blur-out opacity-0 transition-all data-[checked]:visible data-[checked]:h-auto data-[checked]:w-auto data-[checked]:animate-text-focus-in data-[checked]:opacity-100 group-data-[open]:visible group-data-[open]:h-auto group-data-[open]:w-auto group-data-[open]:animate-text-focus-in group-data-[open]:opacity-100 md:visible md:h-auto md:w-auto md:animate-none md:opacity-100 md:data-[checked]:animate-none"
                  >
                    <label class="flex items-center font-serif-en">
                      <input
                        class="relative m-0 box-content block h-4 w-4 appearance-none rounded-full border border-black p-0 before:absolute before:left-1/2 before:top-1/2 before:h-3 before:w-3 before:-translate-x-1/2 before:-translate-y-1/2 before:rounded-full checked:before:bg-black md:h-5 md:w-5 md:before:h-[0.875rem] md:before:w-[0.875rem]"
                        type="radio"
                        name="works-filter"
                        value="{f}"{checked}
                      />
                      <span class="ml-2 flex-1 md:text-[1.125rem]">{f}</span>
                    </label>
                  </li>''')
    slugs = [p['slug'] for p in PROJECTS]
    content = f'''   <section>
        <div id="works-wrapper" class="pt-[var(--header-height)]">
          <ul
            id="works-container"
            class="flex h-[calc(100vh-calc(var(--header-height)+3.25rem))] flex-nowrap justify-start whitespace-nowrap pb-20 md:h-[calc(100vh-calc(var(--header-height)+var(--filter-height)))]"
          >
            <li class="mr-20 md:mr-60 flex flex-col items-center">
              <h1 class="px-[1.4rem] pt-16 md:px-0 md:pt-32"><span class="-mt-5 align-top text-4xl font-medium leading-none vertical-rl md:text-5xl"><span>创客</span></span><span class="-mt-5 ml-2 align-top font-serif-en vertical-rl md:ml-[0.8rem] md:text-xl md:leading-none">(BUILDER)</span></h1>
              <p
                class="mt-16 font-splash text-center"
                style="font-family: 'Zhi Mang Xing', cursive; font-size: clamp(3rem, 8vw, 5.5rem); line-height: 1.3; letter-spacing: 0.1em; opacity: 0.55"
              >
                格物致知
                <br />
                工巧造化
              </p>
            </li>

{chr(10).join(lis)}

            <li class="flex items-start justify-center pl-24 pr-10 md:pl-72">
              <a
                href="/about"
                class="group inline-flex flex-col items-center text-2xl font-medium md:text-[2rem]"
                style="padding-top: 2rem"
              >
                <span class="vertical-rl"><span>述己</span></span>
                <span
                  class="mx-auto mt-6 flex h-[1.5rem] w-[1.5rem] items-center justify-center rounded-full border border-black text-black transition-colors duration-700 group-hover:bg-black group-hover:text-white md:mt-9 md:h-[2rem] md:w-[2rem]"
                  aria-hidden="true"
                >
                  <svg width="12" height="8" viewBox="0 0 12 8" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path
                      d="M5.92843 0.296874C5.92843 0.296874 6.85821 1.97234 7.07889 3.46277L0.628906 3.80681L0.628906 4.29834L7.07265 4.64238C6.82893 6.11193 5.81863 7.70375 5.81863 7.70375C5.81863 7.70375 8.88732 4.4667 11.6966 4.04838C8.8956 3.62585 5.92843 0.296754 5.92843 0.296754L5.92843 0.296874Z"
                      class="fill-current"
                    />
                  </svg>
                </span>
              </a>
            </li>
          </ul>
        </div>

        <div
          id="works-filter-group"
          class="group fixed bottom-0 left-0 w-screen data-[open]:z-30"
        >
          <div class="h-[1px] w-full bg-silver">
            <div class="h-full w-0 bg-black" id="works-progress"></div>
          </div>

          <div
            id="works-filter-overlay"
            class="invisible fixed left-0 top-0 h-screen w-screen bg-[rgba(0,0,0,0.5)] opacity-0 transition-all duration-500 group-data-[open]:visible group-data-[open]:opacity-100 md:hidden"
          ></div>

          <form class="container relative flex items-start overflow-hidden px-5 py-[0.875rem] bg-gray-texture group-data-[open]:z-40 md:px-20 md:py-7">
            <button
              type="button"
              id="works-filter-toggle"
              class="flex items-center font-serif-en text-[1.375rem] leading-none md:pointer-events-none"
            >
              FILTER
              <svg
                width="12"
                height="8"
                viewBox="0 0 12 8"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                class="ml-3 fill-current transition-transform duration-500 group-data-[open]:-rotate-[540deg] md:hidden"
              >
                <path d="M5.90872 7.43347L-0.183594 1.78197L0.907946 0.769409L5.90872 5.43189L10.9095 0.792959L12.001 1.80552L5.90872 7.43347Z" />
              </svg>
            </button>

            <ul class="ml-[7.5rem] flex flex-wrap items-center gap-0 group-data-[open]:gap-4 md:items-baseline md:gap-6">
{chr(10).join(filter_lis)}
            </ul>
          </form>
        </div>
      </section>'''
    style = page_style('builder_list_style.css')
    extra_head = (f'<style>{style}</style>' + FONT_PRELOAD_ZMX
                  + f'<style>html{{visibility:hidden}}html.wf-ready{{visibility:visible}}</style>'
                  + WF_READY_SCRIPT
                  + '\n<script type="module" src="/_astro/hoisted.BrlQdvNY.js"></script>'
                    '<script type="module" src="/_astro/page.LS5KDvwX.js"></script>'
                  + '\n' + GTAG_BLOCK
                  + '\n<link rel="preload" href="/fonts/LXGWWenKai-subset.woff2" as="font" type="font/woff2" crossorigin>')
    tail = ('\n<script type="module" src="/builder/works.js"></script>'
            + render_footer_scripts(prefetch_slugs=slugs))
    page = (render_head('WORKS | HAO-Z Portfolio', 'William Liu -- Biomedical engineering student, AI researcher, and creative technologist based in Wuhan.', '/builder/',
                        extra_head=extra_head)
            + '<body class="font-serif text-black relative overflow-x-hidden" data-bg-color="bright"> <div class="fixed left-0 top-0 -z-10 h-screen w-screen bg-gray-texture" style="background-color:#C2B3A4"></div>   '
            + render_chrome('bright', 'false') + '    '
            + '<main id="main" class="container px-[1.4rem] md:px-20 overflow-x-hidden md:max-w-[calc(85rem+10rem)]">   '
            + content + '   </main>' + tail + '</body></html>')
    write('builder/index.html', page)

# ---------------------------------------------------------- 创客区详情页

def gen_title(p):
    return ('<section class="ts-horizontal-scroll-item mt-28 flex flex-col items-start justify-end '
            'px-5 md:mr-[23.75rem] md:mt-0 md:w-[40rem] md:px-0 md:pb-[7.5rem]">'
            '<h1 class="flex w-full flex-col gap-[0.2rem] whitespace-normal md:gap-4">'
            f'<span class="block w-full text-[2rem] font-medium md:text-5xl">{p["title"]}</span>'
            f'<span class="font-serif-en md:text-lg">{p["subtitle"]}</span></h1>'
            '<div class="mt-16 flex w-full flex-col gap-4 whitespace-normal md:mt-24 md:flex-row md:gap-20">'
            '<p class="flex flex-col gap-2"><span class="text-gray">ROLE</span>'
            f'<span> {p["role"]} </span></p>'
            f'<p class="flex flex-col gap-2"><span>{p["date"]}</span></p></div></section>')

def gen_image(src, alt):
    return f'<img src="{src}" alt="{alt}" class="ts-horizontal-scroll-item mt-9 block md:mt-0 md:h-full">'

def gen_overview(paragraphs):
    paras = ''.join(f'<p>{p}</p>' for p in paragraphs)
    return ('<section class="ts-horizontal-scroll-item flex items-center px-5 py-24 md:px-0 md:py-0">'
            '<div class="md:w-[40rem] md:px-[13.75rem] box-content">'
            '<div class="md:py-[calc(var(--header-height)+7.5rem)]">'
            '<h2 class="mb-9 font-serif-en text-lg md:mb-10 md:text-xl">(Overview)</h2>'
            f'<div class="whitespace-normal"><div class="flex flex-col gap-4 md:text-lg">{paras}</div></div>'
            '</div></div></section>')

def gen_text(tag, paragraphs):
    paras = ''.join(f'<p>{p}</p>' for p in paragraphs)
    return ('<section class="ts-horizontal-scroll-item flex items-center px-5 py-24 md:px-0 md:py-0">'
            '<div class="md:w-[40rem] md:px-[13.75rem] box-content">'
            '<div class="md:py-[calc(var(--header-height)+7.5rem)]">'
            f'<h2 class="mb-9 font-serif-en text-lg md:mb-10 md:text-xl">({tag})</h2>'
            f'<div class="whitespace-normal"><div class="flex flex-col gap-4 md:text-lg">{paras}</div></div>'
            '</div></div></section>')

def gen_image_panel(images):
    if len(images) == 1:
        img = images[0]
        return ('<div class="ts-horizontal-scroll-item flex flex-col items-start justify-center gap-4 '
                'px-5 py-16 md:px-10 md:py-0" style="min-width:360px;max-width:540px">'
                '<p class="font-serif-en text-base text-gray">(' + img.get('label', '') + ')</p>'
                f'<img src="{img["src"]}" alt="{img["alt"]}" style="width:100%;border-radius:8px;'
                'box-shadow:0 4px 20px rgba(0,0,0,0.15)">'
                f'<p class="whitespace-normal text-sm text-gray">{img.get("caption", "")}</p></div>')
    lis = ''
    for img in images:
        lis += ('<li><picture><img src="' + img['src'] + '" alt="' + img['alt'] +
                '" class="ts-image-white-in md:h-full w-full object-cover"></picture>'
                '<p class="mt-3 whitespace-normal text-sm text-gray md:text-base">'
                '<span class="font-serif-en">(' + img.get('label', '') + ') </span>'
                '<span>' + img.get('caption', '') + '</span></p></li>')
    return ('<div class="ts-horizontal-scroll-item md:h-full">'
            '<div class="h-full px-5 md:px-0 md:py-[7.5rem]">'
            '<ul class="md:h-full flex flex-col gap-4 md:gap-10 [&_picture]:block md:[&_picture]:h-full '
            'md:[&_img]:h-full md:flex-col md:[&>li]:h-[calc(50%-1.25rem)] [&>li]:w-full">'
            f'{lis}</ul></div></div>')

def gen_video(src, caption):
    return ('<div class="ts-horizontal-scroll-item flex flex-col items-start justify-center gap-4 px-5 py-16 '
            'md:px-10 md:py-0" style="min-width:360px;max-width:520px">'
            '<p class="font-serif-en text-base text-gray">(Video)</p>'
            '<div style="border-radius:8px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.15)">'
            '<video style="width:100%;display:block" preload="auto" webkit-playsinline playsinline muted autoplay loop>'
            f'<source src="{src}" type="video/mp4"></video></div>'
            f'<p class="whitespace-normal text-sm text-gray">{caption}</p></div>')

def gen_pdf(src, title):
    return ('<div class="ts-horizontal-scroll-item">'
            '<div class="h-full px-5 md:px-0 md:py-[7.5rem]">'
            '<ul class="md:h-full flex flex-wrap md:[&>li]:w-auto gap-4 md:gap-10 [&_picture]:block '
            'md:[&_picture]:h-full md:[&_img]:h-full md:[&_video]:h-full md:[&>li]:h-full flex-col [&>li]:w-full">'
            '<li style="min-width:480px;flex-shrink:0;height:100%">'
            '<div style="background:#1a1a1a;border-radius:12px;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,0.4);'
            'border:1px solid rgba(255,255,255,0.08);height:100%;display:flex;flex-direction:column">'
            '<div style="padding:10px 16px;background:#252525;display:flex;align-items:center;gap:8px;'
            'border-bottom:1px solid rgba(255,255,255,0.06);flex-shrink:0">'
            '<span style="width:10px;height:10px;border-radius:50%;background:#ff5f57;display:inline-block"></span>'
            '<span style="width:10px;height:10px;border-radius:50%;background:#febc2e;display:inline-block"></span>'
            '<span style="width:10px;height:10px;border-radius:50%;background:#28c840;display:inline-block"></span>'
            f'<span style="font-family:Cinzel,serif;font-size:11px;color:#888;margin-left:8px;'
            f'letter-spacing:0.1em">{title}</span></div>'
            '<div style="flex:1;min-height:0;position:relative;overflow:hidden">'
            f'<iframe src="{src}" style="position:absolute;top:0;left:0;width:100%;height:100%;border:none" '
            f'title="{title} PDF"></iframe>'
            '</div></div></li></ul></div></div>')

def gen_back():
    return ('<div class="flex items-center justify-center md:ml-[7.5rem]">'
            '<a href="/builder/" class="group mt-12 inline-flex items-center md:mt-0">'
            '<span class="flex h-[1.7rem] w-[1.7rem] items-center justify-center rounded-full border border-black '
            'text-black transition-colors duration-700 group-hover:bg-black group-hover:text-white '
            'md:h-[2rem] md:w-[2rem]" aria-hidden="true">'
            '<svg width="12" height="8" viewBox="0 0 12 8" fill="none" xmlns="http://www.w3.org/2000/svg">'
            '<path d="M6.07157 7.703C6.07157 7.703 5.14179 6.02754 4.92111 4.53711L11.3711 4.19307L11.3711 3.70154L4.92735 3.35749C5.17107 1.88795 6.18137 0.296124 6.18137 0.296124C6.18137 0.296124 3.11268 3.53318 0.303436 3.9515C3.1044 4.37403 6.07157 7.70312 6.07157 7.70312L6.07157 7.703Z" class="fill-current"></path>'
            '</svg></span>'
            '<span class="ml-4 md:text-lg">返回列表</span>'
            '</a></div>')

def gen_next(next_slug, next_title):
    media = f'projects/{next_slug}/media'
    def pick(name):
        return f'/projects/{next_slug}/media/{name}' if os.path.exists(f'{media}/{name}') \
            else f'/projects/{next_slug}/media/mv-pc.jpg'
    src_pc, src_sp = pick('mv-vertical.jpg'), pick('mv-sp.jpg')
    return (
        '<section class="ts-horizontal-scroll-item box-content flex items-end justify-end whitespace-normal '
        'md:w-[40rem] md:pl-[34rem] md:pr-20">'
        '<div class="px-5 pt-[7.43rem] md:px-0 md:py-[8.125rem]">'
        '<p class="font-serif-en md:text-lg">(NEXT PROJECT)</p>'
        f'<h2 class="text-[2rem] font-medium md:text-[2.625rem]"> {next_title} </h2></div></section>'
        f'<a href="/builder/{next_slug}/" aria-label="查看下一个项目" '
        'class="ts-horizontal-scroll-item relative mt-9 block md:mt-0 md:h-full">'
        '<picture class="ts-crossing-link block h-full">'
        f'<source srcset="{src_pc}" media="(min-width: 768px)" class="md:h-full">'
        f'<img src="{src_sp}" class="ts-image-white-in md:h-full">'
        '</picture>'
        '<span class="absolute bottom-[20px] right-5 flex h-[40px] w-[40px] items-center justify-center '
        'rounded-full bg-mine-shaft md:hidden">'
        '<svg width="14" height="10" viewBox="0 0 14 10" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M6.91412 0.679339C6.91412 0.679339 7.99884 2.63402 8.2563 4.37282L0.731445 4.7742V5.34764L8.24902 5.74902C7.96468 7.46346 6.78603 9.32056 6.78603 9.32056C6.78603 9.32056 10.3661 5.54406 13.6435 5.05602C10.3758 4.56308 6.91412 0.679199 6.91412 0.679199V0.679339Z" class="fill-taupe-gray"></path>'
        '</svg></span></a>')

def render_builder_detail(p):
    parts = [gen_title(p), gen_image(p['hero'], p.get('hero_alt', p['title'])),
             gen_overview(p['overview'])]
    for sec in p.get('sections', []):
        t = sec['type']
        if t == 'text':
            parts.append(gen_text(sec.get('tag', 'Unfold'), sec['paragraphs']))
        elif t == 'image':
            parts.append(gen_image(sec['src'], sec['alt']))
        elif t == 'image_panel':
            parts.append(gen_image_panel(sec['images']))
        elif t == 'video':
            parts.append(gen_video(sec['src'], sec['caption']))
        elif t == 'pdf':
            parts.append(gen_pdf(sec['src'], sec['title']))
        else:
            raise SystemExit(f"未知 section 类型: {t} (项目 {p['slug']})")
    parts.append(gen_back())
    nxt = p['_next']
    parts.append(gen_next(nxt['slug'], nxt['title']))
    content = ' '.join(parts)

    desc = (p['overview'][0][:150] + '…') if p['overview'] else p['title']
    extra_head = (FONT_PRELOADS_ALL
                  + '<link rel="preload" href="/_astro/about.j05OPDV1.css" as="style">'
                    '<link rel="preload" href="/_astro/about.Cl2ZlrCQ.css" as="style">'
                    '<link rel="preload" href="/_astro/font-override.css" as="style">'
                  + f'<link rel="preload" as="image" href="{p["hero"]}" fetchpriority="high">'
                  + '\n<script type="module" src="/_astro/hoisted.GaSC7j3R.js"></script>'
                    '<script type="module" src="/_astro/page.LS5KDvwX.js"></script>'
                  + '\n' + GTAG_BLOCK)
    tail = ('\n' + snippet('detail_tail.html') + '   </main>')
    page = (render_head(f'{p["title"]} | William Liu', desc, f'/builder/{p["slug"]}/',
                        extra_head=extra_head)
            + '<body class="font-serif text-black relative overflow-x-hidden" data-bg-color="bright"> '
            '<div class="fixed left-0 top-0 -z-10 h-screen w-screen bg-gray-texture" '
            'style="background-color:#C2B3A4"></div>   '
            + render_chrome('bright', 'false') + '    '
            + '<main id="main" class="mx-auto overflow-x-hidden false">   '
            + snippet('detail_scroll_open.html')
            + content + ' </div> </section> '
            + tail + '</body></html>')
    write(f'builder/{p["slug"]}/index.html', page)

# ---------------------------------------------------------- sitemap

def render_sitemap():
    base = 'https://haozi.dev'
    urls = ['/', '/about/', '/builder/', '/wanderer/', '/poet/', '/enthusiast/']
    urls += [f'/builder/{p["slug"]}/' for p in PROJECTS]
    items = '\n'.join(f'  <url><loc>{base}{u}</loc><changefreq>monthly</changefreq></url>' for u in urls)
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
           f'{items}\n</urlset>\n')
    write('sitemap.xml', xml)

# ---------------------------------------------------------- 主流程

def main():
    validate()
    log('开始生成页面…')
    render_home()
    render_about()
    render_wanderer()
    render_poet()
    render_enthusiast()
    render_builder_list()
    for p in PROJECTS:
        render_builder_detail(p)
    render_sitemap()
    log('完成。')

if __name__ == '__main__':
    if '--check' in sys.argv:
        validate()
    else:
        main()
