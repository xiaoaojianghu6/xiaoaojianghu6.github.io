#!/usr/bin/env python3
"""Build Builder secondary pages from projects.py + templates."""
import os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from projects import PROJECTS

with open(f'{BASE}/_tpl_before.html', 'r') as f:
    TPL_BEFORE = f.read()
with open(f'{BASE}/_tpl_after.html', 'r') as f:
    TPL_AFTER = f.read()

NEXT_MAP = {
    'cardioagent': 'text-restructuring',
    'grab-car': 'signal-processing',
    'image-segmentation': 'cardioagent',
    'math-modeling': 'grab-car',
    'signal-processing': 'workflow',
    'text-restructuring': 'williamnotes',
    'williamnotes': 'math-modeling',
    'workflow': 'image-segmentation',
}


def gen_title(title, subtitle, role, date):
    return (
        f'<section class="ts-horizontal-scroll-item mt-28 flex flex-col items-start justify-end px-5 md:mr-[23.75rem] md:mt-0 md:w-[40rem] md:px-0 md:pb-[7.5rem]">'
        f'<h1 class="flex w-full flex-col gap-[0.2rem] whitespace-normal md:gap-4">'
        f'<span class="block w-full text-[2rem] font-medium md:text-5xl">{title}</span>'
        f'<span class="font-serif-en md:text-lg">{subtitle}</span>'
        f'</h1>'
        f'<div class="mt-16 flex w-full flex-col gap-4 whitespace-normal md:mt-24 md:flex-row md:gap-20">'
        f'<p class="flex flex-col gap-2"><span class="text-gray">ROLE</span><span> {role} </span></p>'
        f'<p class="flex flex-col gap-2"><span>{date}</span></p>'
        f'</div>'
        f'</section>'
    )


def gen_hero(src, alt):
    return f'<img src="{src}" alt="{alt}" class="ts-horizontal-scroll-item mt-9 block md:mt-0 md:h-full">'


def gen_overview(paragraphs):
    paras = ''.join(f'<p>{p}</p>' for p in paragraphs)
    return (
        f'<section class="ts-horizontal-scroll-item flex items-center px-5 py-24 md:px-0 md:py-0">'
        f'<div class="md:w-[40rem] md:px-[13.75rem] box-content">'
        f'<div class="md:py-[calc(var(--header-height)+7.5rem)]">'
        f'<h2 class="mb-9 font-serif-en text-lg md:mb-10 md:text-xl">(Overview)</h2>'
        f'<div class="whitespace-normal"><div class="flex flex-col gap-4 md:text-lg">{paras}</div></div>'
        f'</div></div></section>'
    )


def gen_text(tag, paragraphs):
    paras = ''.join(f'<p>{p}</p>' for p in paragraphs)
    return (
        f'<section class="ts-horizontal-scroll-item flex items-center px-5 py-24 md:px-0 md:py-0">'
        f'<div class="md:w-[40rem] md:px-[13.75rem] box-content">'
        f'<div class="md:py-[calc(var(--header-height)+7.5rem)]">'
        f'<h2 class="mb-9 font-serif-en text-lg md:mb-10 md:text-xl">({tag})</h2>'
        f'<div class="whitespace-normal"><div class="flex flex-col gap-4 md:text-lg">{paras}</div></div>'
        f'</div></div></section>'
    )


def gen_image(src, alt):
    return f'<img src="{src}" alt="{alt}" class="ts-horizontal-scroll-item mt-9 block md:mt-0 md:h-full">'


def gen_image_panel(images):
    """Generate image panel(s) - single image with caption, or stacked comparison."""
    if len(images) == 1:
        img = images[0]
        return (
            f'<div class="ts-horizontal-scroll-item flex flex-col items-start justify-center gap-4 px-5 py-16 md:px-10 md:py-0" style="min-width:360px;max-width:540px">'
            f'<p class="font-serif-en text-base text-gray">({img["label"]})</p>'
            f'<img src="{img["src"]}" alt="{img["alt"]}" style="width:100%;border-radius:8px;box-shadow:0 4px 20px rgba(0,0,0,0.15)">'
            f'<p class="whitespace-normal text-sm text-gray">{img.get("caption", "")}</p>'
            f'</div>'
        )
    else:
        # Stacked comparison (like image-segmentation A/B)
        lis = ''
        for img in images:
            lis += (
                f'<li><picture><img src="{img["src"]}" alt="{img["alt"]}" class="ts-image-white-in md:h-full w-full object-cover"></picture>'
                f'<p class="mt-3 whitespace-normal text-sm text-gray md:text-base">'
                f'<span class="font-serif-en">({img["label"]}) </span><span>{img.get("caption", "")}</span></p></li>'
            )
        return (
            f'<div class="ts-horizontal-scroll-item md:h-full">'
            f'<div class="h-full px-5 md:px-0 md:py-[7.5rem]">'
            f'<ul class="md:h-full flex flex-col gap-4 md:gap-10 [&_picture]:block md:[&_picture]:h-full md:[&_img]:h-full md:flex-col md:[&>li]:h-[calc(50%-1.25rem)] [&>li]:w-full">'
            f'{lis}</ul></div></div>'
        )


def gen_video(src, caption):
    return (
        f'<div class="ts-horizontal-scroll-item flex flex-col items-start justify-center gap-4 px-5 py-16 md:px-10 md:py-0" style="min-width:360px;max-width:520px">'
        f'<p class="font-serif-en text-base text-gray">(Video)</p>'
        f'<div style="border-radius:8px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.15)">'
        f'<video style="width:100%;display:block" webkit-playsinline playsinline muted autoplay loop>'
        f'<source src="{src}" type="video/mp4"></video></div>'
        f'<p class="whitespace-normal text-sm text-gray">{caption}</p>'
        f'</div>'
    )


def gen_pdf(src, title):
    return (
        f'<div class="ts-horizontal-scroll-item">'
        f'<div class="h-full px-5 md:px-0 md:py-[7.5rem]">'
        f'<ul class="md:h-full flex flex-wrap md:[&>li]:w-auto gap-4 md:gap-10 [&_picture]:block md:[&_picture]:h-full md:[&_img]:h-full md:[&_video]:h-full md:[&>li]:h-full flex-col [&>li]:w-full">'
        f'<li style="min-width:480px;flex-shrink:0;height:100%">'
        f'<div style="background:#1a1a1a;border-radius:12px;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,0.4);border:1px solid rgba(255,255,255,0.08);height:100%;display:flex;flex-direction:column">'
        f'<div style="padding:10px 16px;background:#252525;display:flex;align-items:center;gap:8px;border-bottom:1px solid rgba(255,255,255,0.06);flex-shrink:0">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:#ff5f57;display:inline-block"></span>'
        f'<span style="width:10px;height:10px;border-radius:50%;background:#febc2e;display:inline-block"></span>'
        f'<span style="width:10px;height:10px;border-radius:50%;background:#28c840;display:inline-block"></span>'
        f'<span style="font-family:Cinzel,serif;font-size:11px;color:#888;margin-left:8px;letter-spacing:0.1em">{title}</span>'
        f'</div>'
        f'<div style="flex:1;min-height:0;position:relative;overflow:hidden">'
        f'<iframe src="{src}" style="position:absolute;top:0;left:0;width:100%;height:100%;border:none" title="{title} PDF"></iframe>'
        f'</div></div></li></ul></div></div>'
    )


def gen_back():
    return (
        '<div class="flex items-center justify-center md:ml-[7.5rem]">'
        '<a href="/builder/" class="group mt-12 inline-flex items-center md:mt-0">'
        '<span class="flex h-[1.7rem] w-[1.7rem] items-center justify-center rounded-full border border-black text-black transition-colors duration-700 group-hover:bg-black group-hover:text-white md:h-[2rem] md:w-[2rem]" aria-hidden="true">'
        '<svg width="12" height="8" viewBox="0 0 12 8" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M6.07157 7.703C6.07157 7.703 5.14179 6.02754 4.92111 4.53711L11.3711 4.19307L11.3711 3.70154L4.92735 3.35749C5.17107 1.88795 6.18137 0.296124 6.18137 0.296124C6.18137 0.296124 3.11268 3.53318 0.303436 3.9515C3.1044 4.37403 6.07157 7.70312 6.07157 7.70312L6.07157 7.703Z" class="fill-current"></path>'
        '</svg></span>'
        '<span class="ml-4 md:text-lg">返回列表</span>'
        '</a></div>'
    )


def gen_next(slug, next_slug, next_title):
    return (
        f'<section class="ts-horizontal-scroll-item box-content flex items-end justify-end whitespace-normal md:w-[40rem] md:pl-[34rem] md:pr-20">'
        f'<div class="px-5 pt-[7.43rem] md:px-0 md:py-[8.125rem]">'
        f'<p class="font-serif-en md:text-lg">(NEXT PROJECT)</p>'
        f'<h2 class="text-[2rem] font-medium md:text-[2.625rem]"> {next_title} </h2>'
        f'</div></section>'
        f'<a href="/builder/{next_slug}/" aria-label="查看下一个项目" class="ts-horizontal-scroll-item relative mt-9 block md:mt-0 md:h-full">'
        f'<picture class="ts-crossing-link block h-full">'
        f'<source srcset="/detail/{next_slug}/mv-vertical.jpg" media="(min-width: 768px)" class="md:h-full">'
        f'<img src="/detail/{next_slug}/mv-sp.jpg" class="ts-image-white-in md:h-full">'
        f'</picture>'
        f'<span class="absolute bottom-[20px] right-5 flex h-[40px] w-[40px] items-center justify-center rounded-full bg-mine-shaft md:hidden">'
        f'<svg width="14" height="10" viewBox="0 0 14 10" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<path d="M6.91412 0.679339C6.91412 0.679339 7.99884 2.63402 8.2563 4.37282L0.731445 4.7742V5.34764L8.24902 5.74902C7.96468 7.46346 6.78603 9.32056 6.78603 9.32056C6.78603 9.32056 10.3661 5.54406 13.6435 5.05602C10.3758 4.56308 6.91412 0.679199 6.91412 0.679199V0.679339Z" class="fill-taupe-gray"></path>'
        f'</svg></span></a>'
    )


def build_page(slug):
    proj = PROJECTS[slug]
    next_slug = proj.get('next_slug', NEXT_MAP.get(slug, ''))
    next_title = proj.get('next_title', '')

    parts = []
    parts.append(gen_title(proj['title'], proj['subtitle'], proj['role'], proj['date']))
    parts.append(gen_hero(proj['hero'], proj['hero_alt']))
    parts.append(gen_overview(proj['overview']))

    for sec in proj.get('sections', []):
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

    parts.append(gen_back())
    parts.append(gen_next(slug, next_slug, next_title))

    content = ' '.join(parts)
    return TPL_BEFORE + content + ' </div> </section> ' + TPL_AFTER


def main():
    for slug in PROJECTS:
        html = build_page(slug)
        out_dir = f'{BASE}/{slug}'
        os.makedirs(out_dir, exist_ok=True)
        with open(f'{out_dir}/index.html', 'w') as f:
            f.write(html)
        print(f'  {slug}: {len(html)} chars')

    # Clean up template files
    for f in ['_tpl_before.html', '_tpl_after.html']:
        p = f'{BASE}/{f}'
        if os.path.exists(p):
            os.remove(p)
    print('Done. Templates cleaned up.')


if __name__ == '__main__':
    main()
