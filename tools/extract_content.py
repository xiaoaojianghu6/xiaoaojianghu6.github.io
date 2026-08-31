#!/usr/bin/env python3
"""一次性迁移脚本：从旧页面 HTML / projects.py 中提取内容，生成 content/*.yaml 与 projects/*/project.yaml。
仅用于本次重构，迁移完成后保留作为数据来源说明。"""
import os, re, sys, json, importlib.util, glob
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

def read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()

def dump_yaml(data, path, header=''):
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header)
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, width=100,
                  default_flow_style=False)

# ============ 1. content/wanderer.yaml ============
src = read('wanderer/index.html')
hero = re.search(r'wanderer-hero-img">\s*<img[^>]*src="([^"]+)"', src) or re.search(r'class="wanderer-hero-img">', src)
cover = re.search(r'<img fetchpriority="high" src="([^"]+)" alt="行者封面"', src).group(1)
hero_title = re.search(r'font-size:clamp\(5rem,13vw,9rem\);line-height:1\.3;letter-spacing:0\.15em;text-shadow:[^>]*>(.*?)</p>', src, re.S).group(1)
places = []
for m in re.finditer(r'<div class="place-block">(.*?)</div>\s*(?=<div class="place-block">|</div>\s*<div style="text-align:center)', src, re.S):
    b = m.group(1)
    en = re.search(r'<p class="section-title">(.*?)</p>', b).group(1)
    title = re.search(r'<div class="place-title">(.*?)</div>', b).group(1)
    sub = re.search(r'<div class="place-sub">(.*?)</div>', b).group(1)
    text = re.search(r'<p class="place-text">(.*?)</p>', b, re.S).group(1)
    photos = []
    layout = 'two' if 'place-photos two' in b else 'single'
    for im in re.finditer(r'<img src="([^"]+)" alt="([^"]*)"', b):
        photos.append({'src': im.group(1), 'alt': im.group(2)})
    places.append({'en': en, 'title': title, 'sub': sub, 'text': text,
                   'photos': photos, 'layout': layout})
dump_yaml({'hero': {'cover': cover,
                    'title_lines': [s.strip() for s in hero_title.split('<br>')]},
           'places': places},
          'content/wanderer.yaml',
          '# 行者区内容——改这里即可更新 /wanderer/ 页面\n'
          '# 每个地点是一个 places 列表项；photos 数组顺序 = 页面照片顺序\n'
          '# layout: two = 双图并排, single = 单图\n'
          '# 新增地点：复制一个列表项改文字和图片路径（图片放 top/wanderer/）\n\n')

# ============ 2. content/poems.yaml ============
src = read('poet/index.html')
cover = re.search(r'<img fetchpriority="high" src="([^"]+)" alt="诗人封面"', src).group(1)
hero_title = re.search(r'text-shadow:0 0 40px[^>]*>(.*?)</p>', src, re.S).group(1)
opening_label = re.search(r'<div class="opening-label">(.*?)</div>', src).group(1)
opening_title = re.search(r'<p class="opening-title">(.*?)</p>', src).group(1)
opening_date = re.search(r'<p class="opening-date">(.*?)</p>', src).group(1)
opening_lines = [m.group(1) for m in re.finditer(r'<p class="opening-line">(.*?)</p>', src)]

rows = []
for m in re.finditer(r'<div class="photo-poem-row( photo-poem-row--img-right)?">(.*?)\n  </div>\n\n?', src, re.S):
    side_right = bool(m.group(1))
    b = m.group(2)
    img = re.search(r'<img src="([^"]+)" alt="([^"]*)">\s*<span class="photo-breather__caption">(.*?)</span>', b, re.S)
    ids = re.findall(r'data-poem-id="(\w+)"', b)
    rows.append({'image': img.group(1), 'alt': img.group(2), 'caption': img.group(3),
                 'image_side': 'right' if side_right else 'left', 'poems': ids})

zone_ids = [pid for pid in re.findall(r'data-poem-id="(\w+)"',
            re.search(r'<div class="poetry-zone">(.*?)</div>\s*\n\s*<!--', src, re.S).group(1))
            if all(pid not in r['poems'] for r in rows)]
zone_ids = list(dict.fromkeys(zone_ids))
if not zone_ids:
    all_row_ids = {pid for r in rows for pid in r['poems']}
    zone_ids = [pid for pid in poems.keys() if pid not in all_row_ids]

pm = re.search(r'var poems = \{(.*?)\n\};', src, re.S).group(1)
poems = {}
for m in re.finditer(r'(\w+): \{\s*title: "((?:[^"\\]|\\.)*)",\s*date: "([^"]*)",\s*lines: "((?:[^"\\]|\\.)*)"\s*\}', pm):
    poems[m.group(1)] = {'title': m.group(2), 'date': m.group(3), 'lines': m.group(4)}

long_title = re.search(r'<h2 class="long-poem-title">(.*?)</h2>', src).group(1)
stanzas = [re.sub(r'\s+', ' ', m.group(1)).strip() for m in
           re.finditer(r'<div class="long-poem-stanza">\s*<p>(.*?)</p>', src, re.S)]

dump_yaml({'hero': {'cover': cover, 'title_lines': [s.strip() for s in hero_title.split('<br>')]},
           'opening': {'label': opening_label, 'title': opening_title, 'date': opening_date,
                       'lines': opening_lines},
           'photo_poem_rows': rows,
           'poetry_zone': {'poems': zone_ids},
           'poems': poems,
           'long_poem': {'title': long_title, 'stanzas': stanzas}},
          'content/poems.yaml',
          '# 诗人区内容——改这里即可更新 /poet/ 页面\n'
          '# poems: 每首诗的完整文本（lines 里的 \\n = 换行）；卡片上只显示摘要(自动取开头)\n'
          '# photo_poem_rows: 照片+诗卡并排行，image_side: left/right = 图在左/右\n'
          '# long_poem.stanzas: 长诗的每一节，<br> 表示强制换行\n\n')

# ============ 3. content/enthusiast.yaml ============
src = read('enthusiast/index.html')
cover = re.search(r'<img fetchpriority="high" src="([^"]+)" alt="赤子封面"', src).group(1)
hero_title = re.search(r'text-shadow:0 0 60px[^>]*>(.*?)</p>', src, re.S).group(1)

def two_col_items(block):
    items = []
    for m in re.finditer(r'<div class="two-col-item"(?: style="[^"]*")?><span class="two-col-idx">(\d*)</span><span class="two-col-name">(.*?)</span>(?:<span class="two-col-note">(.*?)</span>)?</div>', block):
        items.append({'name': m.group(2), 'note': m.group(3) or ''})
    return items

cinema_block = re.search(r'id="cinema"(.*?)<!-- MUSIC -->', src, re.S).group(1)
films = [{'src': m.group(1), 'alt': m.group(2), 'caption': m.group(3)} for m in
         re.finditer(r'<img src="([^"]+)" alt="([^"]*)"><span class="film-caption">(.*?)</span>', cinema_block)]
movies = two_col_items(re.search(r'MOVIES</span></div>(.*?)</div>\s*<div>\s*<div class="two-col-item" style="[^"]*"><span class="two-col-idx"></span><span class="two-col-name">TV SERIES', cinema_block, re.S).group(1))
after_tv = cinema_block.split('TV SERIES</span></div>', 1)[1]
tv = [m.group(1) for m in re.finditer(r'<div class="two-col-item"><span class="two-col-name">([^<]+)</span></div>',
      after_tv.split('TRILOGIES', 1)[0])]
trilogies = [m.group(1) for m in re.finditer(r'<div class="two-col-item"><span class="two-col-name">([^<]+)</span></div>',
      after_tv.split('TRILOGIES', 1)[1].split('DIRECTORS', 1)[0])]
directors = [m.group(1) for m in re.finditer(r'<div class="two-col-item"><span class="two-col-name">([^<]+)</span></div>',
      after_tv.split('DIRECTORS', 1)[1])]

music_block = re.search(r'id="music"(.*?)<!-- BOOKS -->', src, re.S).group(1)
music_items = two_col_items(music_block)
books_block = re.search(r'id="books"(.*?)<!-- FOOD -->', src, re.S).group(1)
books_items = two_col_items(books_block)

food_imgs = [{'src': m.group(1), 'alt': m.group(2)} for m in
             re.finditer(r'<img src="([^"]+)" alt="([^"]*)" class="h-full w-full object-cover">', src)]
sport_cards = []
for m in re.finditer(r'<div class="sport-card( dashed)?"><h4>(.*?)</h4><p>(.*?)</p></div>', src, re.S):
    sport_cards.append({'title': m.group(2), 'text': m.group(3), 'dashed': bool(m.group(1))})

texts = re.findall(r'<p class="body-text">(.*?)</p>', src, re.S)
dump_yaml({'hero': {'cover': cover, 'title_lines': [s.strip() for s in hero_title.split('<br>')]},
           'cinema': {'text': texts[0].strip(), 'films': films, 'movies': movies,
                      'tv_series': tv, 'trilogies': trilogies, 'directors': directors},
           'music': {'text': texts[1].strip(), 'items': music_items},
           'books': {'text': texts[2].strip(), 'items': books_items},
           'food': {'text': texts[3].strip(), 'images': food_imgs},
           'sport': {'text': texts[4].strip(), 'cards': sport_cards}},
          'content/enthusiast.yaml',
          '# 赤子区内容——改这里即可更新 /enthusiast/ 页面\n'
          '# films: 海报墙（src 路径 + caption 英文名）；movies/books/music 的 note 是右侧小字\n'
          '# food.images: 美食轮播图；sport.cards: dashed=true 显示虚线框（未开始的项目）\n\n')

# ============ 4. content/about.yaml ============
src = read('about/index.html')
bio_zh = re.search(r'<div id="about-text"[^>]*>\s*<p[^>]*>\s*(.*?)\s*</p>', src, re.S).group(1)
bio_en = re.search(r'<p class="mt-7 text-silver-chalice md:mt-12"[^>]*>\s*(.*?)\s*</p>', src, re.S).group(1)
slider = []
for m in re.finditer(r'<img(?: fetchpriority="high")? src="(/about/mv/[^"]+)" alt="([^"]*)" class="h-full w-full object-cover"', src):
    slider.append({'src': m.group(1), 'alt': m.group(2)})
profile_name = re.search(r'<p class="font-serif-en text-sm text-taupe-gray md:text-base"[^>]*>\s*(.*?)\s*</p>', src).group(1)
profile_big = re.search(r'<p class="mt-2 font-medium" style="font-size:clamp\(1\.5rem,4vw,2\.8rem\)"[^>]*>\s*(.*?)\s*</p>', src).group(1)
emails = re.findall(r'<a href="(mailto:[^"]+)" target="_blank" title="(?:QQ Mail|Gmail)"', src)
github = re.search(r'href="(https://github\.com/[^"]+)"', src).group(1)
wechat_id = re.search(r"font-family:'LXGW WenKai',serif;color:#B4AC97;font-size:14px\">([^<]+)</p>", src).group(1)
profile_p = re.search(r'<p class="mt-7 text-silver-chalice md:mt-10"[^>]*>(.*?)</p>', src, re.S).group(1)
profile_lines = [s.strip() for s in re.split(r'<br[^>]*>', profile_p) if s.strip()]
timeline = []
for m in re.finditer(r'<time datetime="([^"]+)"[^>]*>(.*?)</time> <span class="font-medium"[^>]*>(.*?) <span[^>]*>(.*?)</span></span> <p[^>]*>(.*?)</p> </li>', src, re.S):
    timeline.append({'time': m.group(2), 'city': m.group(3), 'city_sub': m.group(4), 'desc': m.group(5)})
dump_yaml({'bio_zh': bio_zh, 'bio_en': bio_en,
           'slider': slider,
           'profile': {'image': '/about/profile.jpg', 'name_en': profile_name, 'name': profile_big,
                        'lines': profile_lines, 'email_qq': emails[0], 'email_gmail': emails[1],
                        'github': github, 'wechat_qr': '/about/qr-wechat.jpg', 'wechat_id': wechat_id},
           'timeline': timeline},
          'content/about.yaml',
          '# 述己页内容——改这里即可更新 /about/ 页面\n'
          '# slider: 主视觉轮播图（顺序 = 轮播顺序）；timeline: 底部时间线\n\n')

# ============ 5. content/home.yaml ============
src = read('index.html')
hero_en = re.search(r'max-width:26rem;z-index:5;font-family:Cinzel,\'LXGW WenKai\',serif;[^>]*>\s*(.*?)\s*</p>', src, re.S).group(1).strip()
hero_tags = re.search(r'z-index:1;margin-left:clamp\(1rem,5vw,5rem\)[^>]*>\s*(.*?)\s*</p>', src, re.S).group(1).strip()
hero_title = re.search(r'font-size:clamp\(5rem,13vw,9rem\);letter-spacing:0\.15em;text-shadow[^>]*>(.*?)</h2>', src, re.S).group(1)
parallax = []
for m in re.finditer(r'<li class="ts-parallax-gallery-image (relative[^"]*)" data-astro-cid-obodp3z4> <picture class="block w-full mix-blend-multiply"[^>]*> <img src="([^"]+)" alt="([^"]*)"', src):
    cls = m.group(1)
    side = 'left' if 'items-start' in src[max(0, m.start()-200):m.start()] else 'right'
    parallax.append({'src': m.group(2), 'alt': m.group(3), 'side': side, 'class': cls})
# side 判定修正：按在左 ul 还是右 ul
left_part = src.find('<ul class="flex flex-1 flex-col items-start">')
right_part = src.find('<ul class="flex flex-1 flex-col items-end">')
parallax = []
for m in re.finditer(r'<li class="ts-parallax-gallery-image (relative[^"]*)" data-astro-cid-obodp3z4> <picture class="block w-full mix-blend-multiply"[^>]*> <img src="([^"]+)" alt="([^"]*)"', src):
    side = 'left' if (left_part < m.start() < right_part) else 'right'
    parallax.append({'src': m.group(2), 'alt': m.group(3), 'side': side, 'class': m.group(1)})
wanderer_fig = re.search(r'<img src="(/top/wanderer/wanderer-figure\.jpg)"', src).group(1)
quotes = []
for m in re.finditer(r'<div class="scroll-reveal" style="font-family:\'FZXingKai\',\'LXGW WenKai\',serif;font-size:clamp\(1\.3rem,4\.5vw,2\.8rem\);[^"]*margin-bottom:(3\.5rem|10rem);text-align:center"><p>(.*?)</p></div>', src):
    quotes.append({'text': m.group(2), 'last': m.group(2) == '10rem'})
# last 标记按位置重算
quote_data = []
for i, m in enumerate(re.finditer(r'<div class="scroll-reveal" style="font-family:\'FZXingKai\',\'LXGW WenKai\',serif;font-size:clamp\(1\.3rem,4\.5vw,2\.8rem\);[^"]*margin-bottom:(3\.5rem|10rem);text-align:center"><p>(.*?)</p></div>', src)):
    quote_data.append({'text': m.group(2), 'last': m.group(1) == '10rem'})
poet_label = re.search(r'<p class="opening-flash"[^>]*>([^<]+)</p>', src).group(1)
poet_title = re.search(r'margin-bottom:0\.4rem;color:var\(--color-pampas,#F9F9F6\)">([^<]+)</p>', src).group(1)
poet_date = re.search(r'margin-bottom:8rem;text-align:center">([^<]+)</p>', src).group(1)
poet_lines = [m.group(1) for m in re.finditer(r'<div class="scroll-reveal" style="font-family:\'FZXingKai\',\'LXGW WenKai\',serif;font-size:clamp\(2\.4rem,5vw,3\.5rem\);[^"]*"><p>(.*?)</p></div>', src)]
cards = []
for m in re.finditer(r'<a href="/enthusiast/#(\w+)" class="enthusiast-card card-(\w+)"> <span style="font-size:1\.8rem;opacity:0\.5">(.*?)</span> <span[^>]*class="card-en">(\w+)</span> <span[^>]*>(.*?)</span> <span[^>]*>(.*?)</span> </a>', src):
    cards.append({'anchor': m.group(1), 'color': m.group(2), 'emoji': m.group(3),
                  'en': m.group(4), 'zh': m.group(5), 'sub': m.group(6)})
dump_yaml({'hero': {'en_paragraph': hero_en, 'tags_line': hero_tags,
                    'title_lines': [s.strip() for s in hero_title.split('<br>')],
                    'vertical_label': '(PORTFOLIO)'},
           'parallax': parallax,
           'about_link': '/about',
           'wanderer_figure': wanderer_fig,
           'wanderer_quotes': quote_data,
           'poet_gallery': {'label': poet_label, 'title': poet_title, 'date': poet_date, 'lines': poet_lines},
           'enthusiast_cards': cards},
          'content/home.yaml',
          '# 首页内容——改这里即可更新首页各区块\n'
          '# parallax: 视差画廊图片（class 是布局参数：宽度/边距，改图片内容只需换 src）\n'
          '# wanderer_quotes: 行者区古文引用；poet_gallery: 首页背水诗\n'
          '# enthusiast_cards: 赤子区五张卡片\n\n')

# ============ 6. projects/<slug>/project.yaml ============
spec = importlib.util.spec_from_file_location('projects', 'builder/projects.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PROJECTS = mod.PROJECTS

list_src = read('builder/index.html')
props = json.loads(re.search(r"data-props='([^']*)'", list_src).group(1).replace('&quot;', '"'))
list_order = [it['slug'] for it in props['items']]
list_meta = {it['slug']: it for it in props['items']}

os.makedirs('projects/_template', exist_ok=True)
for slug, p in PROJECTS.items():
    meta = list_meta.get(slug, {})
    out = {'slug': slug, 'order': list_order.index(slug) + 1 if slug in list_order else 99}
    out.update(p)
    out.pop('next_slug', None); out.pop('next_title', None)
    out['year'] = meta.get('year', '')
    out['categories'] = meta.get('categories', [])
    out['tags'] = meta.get('tags', [])
    # list_image: mv-vertical（与 mv-pc 同图）→ 统一用 mv-pc
    img = meta.get('image', '')
    out['list_image'] = p['hero']
    os.makedirs(f'projects/{slug}', exist_ok=True)
    dump_yaml(out, f'projects/{slug}/project.yaml',
              f'# 项目数据——{slug}\n'
              '# order: 在列表/首页的排序位置（1 = 最前）；year/categories/tags 用于列表卡片与筛选\n'
              '# sections: 详情页正文区块，type 支持 text/image/video/pdf/image_panel，数组顺序 = 页面顺序\n'
              '# 图片视频放本项目 media/ 文件夹，路径写 /projects/' + slug + '/media/文件名\n\n')

# 模板
dump_yaml({'slug': 'my-new-project', 'order': 99, 'title': '项目名称', 'subtitle': 'ENGLISH SUBTITLE',
           'role': 'SOLO DEVELOPER', 'date': '2025', 'year': '2025',
           'categories': ['PERSONAL PROJECT'], 'tags': ['PERSONAL PROJECT', 'TAG'],
           'hero': '/projects/my-new-project/media/cover.jpg', 'hero_alt': '主视觉图',
           'overview': ['一段话简介。'],
           'sections': [
               {'type': 'text', 'tag': 'Unfold', 'paragraphs': ['正文段落…']},
               {'type': 'image', 'src': '/projects/my-new-project/media/图1.jpg', 'alt': '图片说明'},
               {'type': 'video', 'src': '/projects/my-new-project/media/demo.mp4', 'caption': '视频说明'},
               {'type': 'image_panel', 'images': [{'src': '/projects/my-new-project/media/a.jpg', 'alt': 'A', 'label': 'A', 'caption': ''}]},
               {'type': 'pdf', 'src': '/projects/my-new-project/media/paper.pdf', 'title': 'PDF 标题'},
           ]},
          'projects/_template/project.yaml',
          '# ===== 新项目模板 =====\n'
          '# 用法：复制整个 _template 文件夹，重命名为项目英文短名（如 my-new-project），\n'
          '# 图片/视频放进该文件夹的 media/ 子目录，然后修改本文件。改完运行 python3 build.py\n\n')

print('OK: content/*.yaml + projects/*/project.yaml written')
