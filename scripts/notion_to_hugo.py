import os
import zipfile
import shutil
import tempfile
import frontmatter
import re
from datetime import datetime
from pathlib import Path

# === 配置路径 ===
NOTION_EXPORT_DIR = "/workspaces/xiaoaojianghu6.github.io/notion_zips/"
HUGO_CONTENT_DIR = '/workspaces/xiaoaojianghu6.github.io/content/post/'
PROCESSED_ZIP_RECORD = os.path.join(NOTION_EXPORT_DIR, ".processed_zips.txt")

# === 获取已处理的 zip 文件列表 ===
def get_processed_zips():
    if os.path.exists(PROCESSED_ZIP_RECORD):
        with open(PROCESSED_ZIP_RECORD, 'r') as f:
            return set(line.strip() for line in f.readlines())
    return set()

# === 将 zip 标记为已处理 ===
def mark_zip_as_processed(zip_name):
    with open(PROCESSED_ZIP_RECORD, 'a') as f:
        f.write(zip_name + "\n")

# === 清理标题（去除 hash）===
def clean_title(filename):
    title = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"\s+[a-f0-9]{32}$", "", title)

# === 提取摘要（第一段非空文字）===
def extract_summary(markdown_text):
    # 跳过 front matter 和标题，提取第一个段落作为摘要
    in_frontmatter = False
    lines = []
    for line in markdown_text.splitlines():
        line = line.strip()
        if line == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue
        if line.startswith("#"):  # 跳过标题行
            continue
        if line:  # 第一个非空非标题的行，认为是正文段落
            return line
    return ""

# === 处理 markdown 文件：生成 frontmatter、插图重命名 ===
def convert_md(md_path, hugo_dir):
    with open(md_path, 'r', encoding='utf-8') as f:
        raw_content = f.read()

    raw_title = Path(md_path).stem
    title = clean_title(raw_title)
    date_str = datetime.now().strftime('%Y-%m-%d')

    # 替换图片链接
    img_counter = 0
    def replace_img(match):
        nonlocal img_counter
        img_counter += 1
        ext = os.path.splitext(match.group(2))[-1]
        return f"![](output{img_counter}{ext})"

    content = re.sub(r'!\[(.*?)\]\(([^)]+)\)', replace_img, raw_content)

    # 加载 frontmatter 并添加字段
    post = frontmatter.loads(content)
    post['title'] = title
    post['date'] = date_str
    post['authors'] = ['william']
    post['tags'] = []
    post['summary'] = extract_summary(raw_content)

    # 保存为 index.md
    target_dir = os.path.join(hugo_dir, title)
    os.makedirs(target_dir, exist_ok=True)
    index_path = os.path.join(target_dir, 'index.md')

    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(frontmatter.dumps(post))

    print(f"✅ 文章已处理：{title}")

    # 拷贝插图
    assets_src_dir = os.path.join(os.path.dirname(md_path), raw_title)
    if os.path.exists(assets_src_dir):
        img_files = [f for f in os.listdir(assets_src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        for i, filename in enumerate(img_files, 1):
            ext = os.path.splitext(filename)[-1].lower()
            new_name = f"output{i}{ext}"
            shutil.copy(
                os.path.join(assets_src_dir, filename),
                os.path.join(target_dir, new_name)
            )
        print(f"🖼️ 插图已导入：{len(img_files)} 张")

# === 主函数：批量处理所有未处理 zip 文件 ===
def main():
    print("📦 正在查找未处理的 Notion 导出 zip 文件...")
    processed_zips = get_processed_zips()
    all_zips = [f for f in os.listdir(NOTION_EXPORT_DIR) if f.endswith('.zip')]
    unprocessed_zips = [f for f in all_zips if f not in processed_zips]

    if not unprocessed_zips:
        print("✅ 所有 zip 文件都已经处理过啦，无需重复操作。")
        return

    print(f"📁 共发现 {len(unprocessed_zips)} 个未处理 zip 文件，开始逐个处理...")

    for zip_name in unprocessed_zips:
        zip_path = os.path.join(NOTION_EXPORT_DIR, zip_name)
        print(f"\n📦 正在处理：{zip_path}")
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        md_files = list(Path(temp_dir).rglob("*.md"))
        if not md_files:
            print(f"⚠️ 未找到 Markdown 文件：{zip_name}")
        else:
            print(f"📄 共找到 {len(md_files)} 篇文章，开始处理...")
            for md in md_files:
                convert_md(str(md), HUGO_CONTENT_DIR)
        
        mark_zip_as_processed(zip_name)
        shutil.rmtree(temp_dir)

    print("\n🚀 所有 zip 文件处理完毕，可以在 Codespace 中继续编辑发布。")

if __name__ == '__main__':
    main()