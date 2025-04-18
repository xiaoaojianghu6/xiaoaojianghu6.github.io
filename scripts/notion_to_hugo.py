import os
import zipfile
import shutil
import tempfile
import frontmatter
import re
from datetime import datetime

# === 配置路径 ===
NOTION_EXPORT_DIR = "/workspaces/xiaoaojianghu6.github.io/notion_zips/"
HUGO_CONTENT_DIR = '/workspaces/xiaoaojianghu6.github.io/content/post/'

def extract_latest_zip(export_dir):
    zips = [f for f in os.listdir(export_dir) if f.endswith('.zip')]
    if not zips:
        raise FileNotFoundError("❌ 没找到 Notion 导出的 zip 文件")
    latest_zip = max(zips, key=lambda x: os.path.getctime(os.path.join(export_dir, x)))
    zip_path = os.path.join(export_dir, latest_zip)
    print(f"📦 正在解压最新导出：{zip_path}")
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir

def clean_title(filename):
    title = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"\s+[a-f0-9]{32}$", "", title)


def convert_md(md_path, hugo_dir):
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # ✅ 先加载 markdown 内容为 post 对象
    post = frontmatter.loads(content)

    # 提取标题和日期
    raw_title = os.path.splitext(os.path.basename(md_path))[0]
    title = clean_title(raw_title)
    date_str = datetime.now().strftime('%Y-%m-%d')

    # ✅ 然后添加 front matter 字段
    post['title'] = title
    post['date'] = date_str
    post['authors'] = ['william']
    post['tags'] = []

    # 文章保存路径（例如 /content/post/标题/index.md）
    post_dir = os.path.join(hugo_dir, title)
    os.makedirs(post_dir, exist_ok=True)
    output_path = os.path.join(post_dir, 'index.md')

    # 保存带有 front matter 的 markdown 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(frontmatter.dumps(post))

    # 更新 Markdown 图片链接格式
    def replace_img(match):
        nonlocal img_counter
        img_counter += 1
        ext = os.path.splitext(match.group(1))[1]
        return f"![](output{img_counter}{ext})"

    img_counter = 0
    content = re.sub(r'!\[(.*?)\]\(([^)]+)\)', replace_img, content)

    # 添加 Front Matter
    post = frontmatter.loads(content)
    post['title'] = title
    post['date'] = date_str
    post['authors'] = ['william']
    post['tags'] = []

    # 创建目标目录
    target_dir = os.path.join(hugo_dir, title)
    os.makedirs(target_dir, exist_ok=True)

    # 保存 index.md
    index_path = os.path.join(target_dir, 'index.md')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(frontmatter.dumps(post))

    print(f"✅ 文章已处理：{title}")

    # 拷贝插图
    assets_src_dir = os.path.join(os.path.dirname(md_path), raw_title)
    if os.path.exists(assets_src_dir):
        for i, filename in enumerate(os.listdir(assets_src_dir), 1):
            ext = os.path.splitext(filename)[-1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.gif']:
                new_name = f"output{i}{ext}"
                shutil.copy(
                    os.path.join(assets_src_dir, filename),
                    os.path.join(target_dir, new_name)
                )
        print(f"🖼️ 插图已导入：{len(os.listdir(assets_src_dir))} 张")

def main():
    print("📦 正在查找最新 Notion 导出文件...")
    temp_dir = extract_latest_zip(NOTION_EXPORT_DIR)

    md_files = [f for f in os.listdir(temp_dir) if f.endswith('.md')]
    if not md_files:
        print("⚠️ 没有找到任何 Markdown 文件。")
        return

    for md in md_files:
        convert_md(os.path.join(temp_dir, md), HUGO_CONTENT_DIR)

    print("🚀 所有文章已处理完毕，可在 Codespace 中继续编辑后发布。")

if __name__ == '__main__':
    main()