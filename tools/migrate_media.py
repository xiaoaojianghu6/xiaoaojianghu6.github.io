#!/usr/bin/env python3
"""一次性迁移：detail/<slug>/ → projects/<slug>/media/，中文文件名改英文，去除重复三联图，更新 project.yaml 路径。"""
import os, subprocess, hashlib, glob, sys
import yaml

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RENAME = {
    'cardioagent/cardioagent论文_副本.pdf': 'cardioagent-paper.pdf',
    'grab-car/小车结构设计图.JPG': 'car-structure.jpg',
    'grab-car/抓取功能小车设计.JPG': 'car-design.jpg',
    'grab-car/小车视频.mp4': 'car-demo.mp4',
    'image-segmentation/指尖创世图和 resurgsam 分割对比图.jpg': 'fingertip-vs-resurgsam.jpg',
    'image-segmentation/resurgsam 分割与指尖创世动作很像.png': 'resurgsam-mask-vs-fingertip.png',
    'image-segmentation/tim 分割.jpg': 'tim-segmentation.jpg',
    'math-modeling/美赛论文.pdf': 'mcm-paper.pdf',
    'math-modeling/国赛论文.pdf': 'cumcm-paper.pdf',
    'math-modeling/美赛.png': 'mcm.png',
    'math-modeling/国赛.png': 'cumcm.png',
    'signal-processing/独立成分分析-用于封面.png': 'ica-cover.png',
    'signal-processing/脑电信号频率提取.png': 'eeg-bands.png',
    'signal-processing/rppg 视频识别心率等生理信号.png': 'rppg-result.png',
    'text-restructuring/多项目对比1.png': 'comparison-1.png',
    'text-restructuring/多项目对比2.png': 'comparison-2.png',
    'text-restructuring/多项目对比3.png': 'comparison-3.png',
    'williamnotes/项目-williamnotes.png': 'williamnotes-app.png',
    'workflow/高德MCP.png': 'amap-mcp.png',
    'workflow/workflow 图表.png': 'workflow-chart.png',
}

for d in sorted(glob.glob('detail/*')):
    if not os.path.isdir(d):
        continue
    slug = d.split('/')[1]
    dst = f'projects/{slug}/media'
    os.makedirs(dst, exist_ok=True)
    for f in sorted(os.listdir(d)):
        if f == '.DS_Store':
            os.remove(os.path.join(d, f))
            continue
        subprocess.run(['git', 'mv', os.path.join(d, f), os.path.join(dst, f)], check=True)
    os.rmdir(d)

# 中文名重命名
for old, new in RENAME.items():
    src = f'projects/{old.split("/")[0]}/media/{old.split("/")[1]}'
    if os.path.exists(src):
        subprocess.run(['git', 'mv', src, f'projects/{old.split("/")[0]}/media/{new}'], check=True)

# 去重：mv-sp / mv-vertical 与 mv-pc 相同则删除
for d in sorted(glob.glob('projects/*/media')):
    pc = os.path.join(d, 'mv-pc.jpg')
    if not os.path.exists(pc):
        continue
    h_pc = hashlib.md5(open(pc, 'rb').read()).hexdigest()
    for dup in ('mv-sp.jpg', 'mv-vertical.jpg'):
        p = os.path.join(d, dup)
        if os.path.exists(p) and hashlib.md5(open(p, 'rb').read()).hexdigest() == h_pc:
            os.remove(p)
            subprocess.run(['git', 'rm', '--cached', p.replace('projects/', 'projects/'), '-q'], check=False)

# 更新 project.yaml 路径
for yml in glob.glob('projects/*/project.yaml'):
    slug = yml.split('/')[1]
    raw = open(yml, encoding='utf-8').read()
    raw = raw.replace(f'/detail/{slug}/', f'/projects/{slug}/media/')
    data = yaml.safe_load(raw)
    # 旧文本重命名映射（yaml 里的中文文件路径）
    for old, new in RENAME.items():
        o_slug, o_name = old.split('/')
        if o_slug == slug:
            raw = raw.replace(o_name, new)
    data = yaml.safe_load(raw)
    # 三联图去重引用
    for key in ('hero', 'list_image'):
        v = data.get(key, '')
        if isinstance(v, str) and ('mv-sp.jpg' in v or 'mv-vertical.jpg' in v):
            cand = f'/projects/{slug}/media/mv-pc.jpg'
            if os.path.exists('projects/' + slug + '/media/mv-pc.jpg'):
                data[key] = cand
    for sec in data.get('sections', []):
        if sec.get('type') == 'image' and isinstance(sec.get('src'), str):
            if 'mv-sp.jpg' in sec['src'] or 'mv-vertical.jpg' in sec['src']:
                sec['src'] = f'/projects/{slug}/media/mv-pc.jpg'
        if sec.get('type') in ('image_panel',):
            for img in sec.get('images', []):
                if 'mv-sp.jpg' in img.get('src', '') or 'mv-vertical.jpg' in img.get('src', ''):
                    img['src'] = f'/projects/{slug}/media/mv-pc.jpg'
    with open(yml, 'w', encoding='utf-8') as f:
        f.write(raw.split('\n\n', 1)[0] + '\n\n')
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, width=100)

print('media migrated')
for d in sorted(glob.glob('projects/*/media')):
    print(d, '->', sorted(os.listdir(d)))
