#!/usr/bin/env python3
"""一次性：从 about/index.html 抽取共享的页头+菜单(chrome)与页脚(footer)，做成带 token 的模板文件。"""
import os, re

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src = open('about/index.html', encoding='utf-8').read()

# ---- chrome: header + menu (从 <header 到 <main 之前) ----
h_start = src.find('<header id="header"')
m_end = src.find('<main')
chrome = src[h_start:m_end].rstrip()
# 修正复制走样的残缺引号
chrome = chrome.replace('data-astro-cid-z6iz25dn"', 'data-astro-cid-z6iz25dn')
# menu 块结尾是连续 </div>，保留原样直到最后一个 </div>
chrome = chrome.rstrip()
while not chrome.endswith('</div>'):
    chrome = chrome[:-1].rstrip()

# header 开标签参数化
old_head = '<header id="header" class="group-[header] fixed left-0 top-0 z-30 w-screen transition-all data-[color=dark]:text-pampas" data-color="dark" data-mix-blend-mode="false"'
new_head = '<header id="header" class="group-[header] fixed left-0 top-0 z-30 w-screen transition-all data-[color=dark]:text-pampas" data-color="{{HEADER_COLOR}}" data-mix-blend-mode="{{HEADER_BLEND}}"{{HEADER_STYLE}}'
assert old_head in chrome, 'header tag not found'
chrome = chrome.replace(old_head, new_head)

# 顶部导航 5 项 → token
nav_start = chrome.find('<ul class="ts-header-list')
nav_end = chrome.find('</ul>', nav_start) + len('</ul>')
nav_block = chrome[nav_start:nav_end]
lis = re.findall(r'<li class="relative".*?</li>', nav_block, re.S)
assert len(lis) == 5, f'nav lis = {len(lis)}'
chrome = chrome[:nav_start] + '{{HEADER_NAV}}' + chrome[nav_end:]

# 菜单 7 项 → token；ul max-width 参数化
ul_start = chrome.find('<ul style="display:flex;flex-direction:column;align-items:stretch')
ul_end = chrome.find('</ul>', ul_start) + len('</ul>')
menu_ul = chrome[ul_start:ul_end]
menu_lis = re.findall(r'<li style="list-style:none".*?</li>', menu_ul, re.S)
assert len(menu_lis) == 7, f'menu lis = {len(menu_lis)}'
chrome = chrome[:ul_start] + '{{MENU_ITEMS}}' + chrome[ul_end:]
chrome = chrome.replace('max-width:600px', 'max-width:{{MENU_MAX_WIDTH}}')

os.makedirs('templates/partials', exist_ok=True)
open('templates/partials/chrome.html', 'w', encoding='utf-8').write(chrome)
print('chrome.html written:', len(chrome), 'chars')

# ---- footer: 回到顶部 + 分割线 + 跑马灯 (从回顶 div 到 footer-wrapper 闭合) ----
f_start = src.find('<div style="text-align:center;padding:14rem 1rem 3rem">')
assert f_start > 0, 'back-to-top div not found'
fw_close = src.find('</div>', src.find('</footer>'))
assert fw_close > f_start, f'footer close {fw_close} before start {f_start}'
footer = src[f_start:fw_close + len('</div>')]

# 跑马灯两行内容相同：第一行原样保留结构、内嵌 {{MARQUEE_WORDS_ROW}}；第二行整行替换为同一 token 展开两份
row_pat = re.compile(r'<span style="display:flex;flex-shrink:0;white-space:nowrap;align-items:center">.*?</span> </span>', re.S)
rows = row_pat.findall(footer)
assert len(rows) == 2, f'marquee rows = {len(rows)}'
words = re.findall(r'font-weight:(\d+);color:([^;]+);letter-spacing:0\.03em;padding:0 5rem">([^<]+)</span>', rows[0])
assert len(words) == 6, f'marquee words = {len(words)}'
# 把第一行的词序列换成 token（保留行结构）
row0_tokens = row_pat.sub('{{MARQUEE_ROW}}', rows[0], count=1)
footer = footer.replace(rows[0], row0_tokens, 1)
footer = footer.replace(rows[1], '{{MARQUEE_ROW}}', 1)
open('templates/partials/footer.html', 'w', encoding='utf-8').write(footer)
print('footer.html written:', len(footer), 'chars')
print('words:', [w[2] for w in words])
