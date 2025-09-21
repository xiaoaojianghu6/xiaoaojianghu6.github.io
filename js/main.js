document.addEventListener('DOMContentLoaded', () => {

    console.log("JavaScript is active. Final version loaded.");

    const GITHUB_USERNAME = "xiaoaojianghu6";
    const GITHUB_REPO = "xiaoaojianghu6.github.io";
    let postsCache = null;

    const fetchAndParsePosts = async () => {
        if (postsCache) return postsCache;
        try {
            const api_url = `https://api.github.com/repos/${GITHUB_USERNAME}/${GITHUB_REPO}/contents/posts`;
            const response = await fetch(api_url);
            if (!response.ok) throw new Error(`GitHub API request failed: ${response.statusText}`);
            const files = await response.json();
            const markdownFiles = files.filter(file => file.name.endsWith('.md'));
            const postsPromises = markdownFiles.map(async (file) => {
                const postResponse = await fetch(file.download_url);
                const rawContent = await postResponse.text();
                return parseMarkdownFile(rawContent, file.name);
            });
            const posts = await Promise.all(postsPromises);
            posts.sort((a, b) => new Date(b.date) - new Date(a.date));
            postsCache = posts;
            return posts;
        } catch (error) {
            console.error("Error fetching posts:", error);
            return [];
        }
    };

    const parseMarkdownFile = (content, filename) => {
        let title = "Untitled Post";
        let date = new Date().toISOString().split('T')[0]; // Default to today's date
        let description = "No description available.";

        const frontmatterMatch = content.match(/^---\s*([\s\S]*?)\s*---/);
        if (frontmatterMatch) {
            const yaml = frontmatterMatch[1];
            const dateMatch = yaml.match(/^date:\s*(.*)/m);
            if (dateMatch) {
                // Use the date from frontmatter if it exists
                date = new Date(dateMatch[1].trim()).toISOString().split('T')[0];
            }
        }

        const titleMatch = content.match(/^#\s+(.*)/m);
        if (titleMatch) title = titleMatch[1].trim();

        const contentWithoutFrontmatter = content.replace(/^---\s*([\s\S]*?)\s*---/, '').trim();
        const firstParagraphMatch = contentWithoutFrontmatter.match(/^(?!#)(?!>)\s*(\S.+)/m);
        if (firstParagraphMatch) {
            description = firstParagraphMatch[1].trim().substring(0, 150) + "...";
        }

        return { title, date, description, filename, rawContent: content };
    };

    const renderPostList = async () => {
        const container = document.getElementById('post-list-container');
        if (!container) return;
        const posts = await fetchAndParsePosts();
        container.innerHTML = '';
        if (posts.length === 0) {
            container.innerHTML = '<p>No posts found or failed to load.</p>';
            return;
        }
        const postList = document.createElement('ul');
        postList.className = 'post-list';
        posts.forEach(post => {
            const listItem = document.createElement('li');
            listItem.className = 'post-item';
            const postLink = document.createElement('a');
            postLink.href = `./post.html?post=${post.filename}`;
            postLink.innerHTML = `<h2>${post.title}</h2><p class="post-date">${post.date}</p><p>${post.description}</p>`;
            listItem.appendChild(postLink);
            postList.appendChild(listItem);
        });
        container.appendChild(postList);
    };

    const renderPostContent = async () => {
        const container = document.getElementById('post-content-area');
        if (!container) return;
        const params = new URLSearchParams(window.location.search);
        const postFilename = params.get('post');
        if (!postFilename) {
            container.innerHTML = '<h1>Error</h1><p>Post not specified.</p>';
            return;
        }
        const posts = await fetchAndParsePosts();
        const post = posts.find(p => p.filename === postFilename);
        if (!post) {
            container.innerHTML = `<h1>Error</h1><p>Post not found.</p>`;
            return;
        }
        document.title = `${post.title} - William Liu`;
        const contentToRender = post.rawContent.replace(/^---\s*([\s\S]*?)\s*---/, '').replace(/^#\s+(.*)/m, '').trim();
        container.innerHTML = `
            <header class="post-header"><h1>${post.title}</h1><p class="post-date">${post.date}</p></header>
            <div class="post-content">${marked.parse(contentToRender)}</div>
        `;
    };

    const setYear = () => {
        const yearSpan = document.getElementById('current-year');
        if (yearSpan) yearSpan.textContent = new Date().getFullYear();
    };

    const setupThemeToggle = () => {
        const toggleButton = document.getElementById('theme-toggle');
        const body = document.body;
        if (!toggleButton) return;
        const applySavedTheme = () => {
            const savedTheme = localStorage.getItem('theme');
            body.classList.toggle('dark-mode', savedTheme === 'dark');
        };
        toggleButton.addEventListener('click', () => {
            body.classList.toggle('dark-mode');
            localStorage.setItem('theme', body.classList.contains('dark-mode') ? 'dark' : 'light');
        });
        applySavedTheme();
    };

    setYear();
    setupThemeToggle();
    renderPostList();
    renderPostContent();
});