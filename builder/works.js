// 创客区列表页交互 —— 横向滚动（桌面 GSAP pin / 移动原生滚动）+ 分类筛选
// 数据由 build.py 静态生成，本文件只负责交互，与旧 React 版行为一致
import { g as gsap } from '/_astro/index.DjKJqAo0.js';
import { S as ScrollTrigger } from '/_astro/ScrollTrigger.DZdR0iV_.js';

(function () {
gsap.registerPlugin(ScrollTrigger);

var wrapper = document.getElementById('works-wrapper');
var container = document.getElementById('works-container');
var progress = document.getElementById('works-progress');
var filterGroup = document.getElementById('works-filter-group');
var filterToggle = document.getElementById('works-filter-toggle');
var filterOverlay = document.getElementById('works-filter-overlay');
function getOverflow(el) {
  return el.scrollWidth - el.offsetWidth;
}
function getIsMobile() {
  return /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
}

var st = null;
var tl = null;

function initScroll() {
  if (st) { st.kill(); st = null; }
  if (tl) { tl.kill(); tl = null; }
  if (getIsMobile()) {
    wrapper.style.overflowX = 'auto';
    wrapper.style.scrollbarWidth = 'none';
    wrapper.style.overflowY = 'hidden';
    wrapper.addEventListener('scroll', function () {
      var sl = wrapper.scrollLeft;
      var sw = container.scrollWidth;
      var cw = container.clientWidth;
      progress.style.width = ((sl / (sw - cw)) * 100).toFixed(5) + '%';
    });
  } else {
    var overflow = getOverflow(container);
    if (overflow <= 0) return;
    tl = gsap.to(container, { x: function () { return getOverflow(container) * -1.3; }, ease: 'none' });
    st = ScrollTrigger.create({
      trigger: wrapper,
      pin: true,
      scrub: 1,
      start: 'top top',
      end: function () { return '+=' + getOverflow(container); },
      animation: tl,
      anticipatePin: 1,
      invalidateOnRefresh: true,
      onUpdate: function (self) {
        progress.style.width = ((self.progress * 100).toFixed(5)) + '%';
      },
    });
    setTimeout(function () { ScrollTrigger.refresh(); }, 500);
  }
}

// 等图片加载完再计算横向宽度（与旧版一致）
var images = container.querySelectorAll('img');
var loaded = 0;
var didInit = false;
function checkAllLoaded() {
  loaded++;
  if (loaded >= images.length && !didInit) { didInit = true; initScroll(); }
}
if (images.length > 0) {
  images.forEach(function (img) {
    if (img.complete) { loaded++; }
    else {
      img.addEventListener('load', checkAllLoaded, { once: true });
      img.addEventListener('error', checkAllLoaded, { once: true });
    }
  });
  if (loaded >= images.length && !didInit) { didInit = true; initScroll(); }
} else {
  initScroll();
}

var filterBar = filterGroup;
if (filterBar) {
  document.documentElement.style.setProperty('--filter-height', filterBar.offsetHeight + 'px');
}

// ===== 筛选 =====
function applyFilter(value) {
  document.querySelectorAll('#works-container [data-tags]').forEach(function (li) {
    var tags = (li.getAttribute('data-tags') || '').split(' ');
    var hidden = value !== 'ALL' && tags.indexOf(value) === -1;
    if (hidden) li.setAttribute('data-hidden', '');
    else li.removeAttribute('data-hidden');
  });
  document.querySelectorAll('#works-filter-group li[data-checked]').forEach(function (li) {
    var input = li.querySelector('input');
    if (input && input.value !== value) li.removeAttribute('data-checked');
    else li.setAttribute('data-checked', 'true');
  });
  setTimeout(function () {
    if (st) { st.kill(); st = null; }
    if (tl) { tl.kill(); tl = null; }
    didInit = false;
    loaded = 0;
    if (getIsMobile()) { initScroll(); }
    else {
      // 桌面端重新测量
      gsap.set(container, { x: 0 });
      initScroll();
      if (st) ScrollTrigger.refresh();
    }
  }, 1550);
}

document.querySelectorAll('input[name="works-filter"]').forEach(function (input) {
  input.addEventListener('change', function () { applyFilter(input.value); });
});

if (filterToggle && filterGroup) {
  filterToggle.addEventListener('click', function () {
    if (filterGroup.hasAttribute('data-open')) filterGroup.removeAttribute('data-open');
    else filterGroup.setAttribute('data-open', '');
  });
}
if (filterOverlay && filterGroup) {
  filterOverlay.addEventListener('click', function () {
    filterGroup.removeAttribute('data-open');
  });
}
})();
