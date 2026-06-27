import { useState, useRef, useLayoutEffect } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

/* ===== Title Block ===== */
function TitleBlock({ jp, en }) {
  return (
    <>
      <h1 className="px-[1.4rem] pt-16 md:px-0 md:pt-32">
        <span className="-mt-5 align-top text-4xl font-medium leading-none vertical-rl md:text-5xl">
          <span>{jp}</span>
        </span>
        <span className="-mt-5 ml-2 align-top font-serif-en vertical-rl md:ml-[0.8rem] md:text-xl md:leading-none">
          ({en})
        </span>
      </h1>
      {/* 金句 */}
      <p
        className="mt-16 font-splash text-center"
        style={{
          fontFamily: "'Zhi Mang Xing', cursive",
          fontSize: 'clamp(3rem, 8vw, 5.5rem)',
          lineHeight: '1.3',
          letterSpacing: '0.1em',
          opacity: '0.55',
        }}
      >
        格物致知
        <br />
        工巧造化
      </p>
    </>
  );
}

/* ===== Horizontal Scroll Hook ===== */
function getOverflow(el) {
  return el.scrollWidth - el.offsetWidth;
}

function getIsMobile() {
  return /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
}

function useHorizontalScroll(wrapperRef, containerRef, progressRef, filterRef, items) {
  const stRef = useRef(null);
  const tlRef = useRef(null);
  const didInit = useRef(false);

  useLayoutEffect(() => {
    const wrapper = wrapperRef.current;
    const container = containerRef.current;
    const progress = progressRef.current;
    if (!wrapper || !container || !progress) return;

    // Wait for images to load before calculating overflow
    const images = container.querySelectorAll('img');
    let loadedCount = 0;
    const totalImages = images.length;

    function initScroll() {
      // Clean up previous instances
      if (stRef.current) { stRef.current.kill(); stRef.current = null; }
      if (tlRef.current) { tlRef.current.kill(); tlRef.current = null; }
      didInit.current = true;

      if (getIsMobile()) {
        wrapper.style.overflowX = 'auto';
        wrapper.style.scrollbarWidth = 'none';
        wrapper.style.overflowY = 'hidden';

        const onScroll = () => {
          const sl = wrapper.scrollLeft;
          const sw = container.scrollWidth;
          const cw = container.clientWidth;
          const pct = (sl / (sw - cw)) * 100;
          progress.style.width = `${pct.toFixed(5)}%`;
        };
        wrapper.addEventListener('scroll', onScroll);
        return () => wrapper.removeEventListener('scroll', onScroll);
      } else {
        const overflow = getOverflow(container);
        if (overflow <= 0) return;

        const tl = gsap.to(container, {
          x: () => getOverflow(container) * -1.3,
          ease: 'none',
        });

        const st = ScrollTrigger.create({
          trigger: wrapper,
          pin: true,
          scrub: 1,
          start: 'top top',
          end: () => `+=${getOverflow(container)}`,
          animation: tl,
          anticipatePin: 1,
          invalidateOnRefresh: true,
          onUpdate: (self) => {
            progress.style.width = `${(self.progress * 100).toFixed(5)}%`;
          },
        });

        stRef.current = st;
        tlRef.current = tl;

        setTimeout(() => ScrollTrigger.refresh(), 500);

        return () => {
          st.kill();
          tl.kill();
          stRef.current = null;
          tlRef.current = null;
        };
      }
    }

    // If images exist, wait for them to load; otherwise init immediately
    if (totalImages > 0) {
      const checkAllLoaded = () => {
        loadedCount++;
        if (loadedCount >= totalImages && !didInit.current) {
          initScroll();
        }
      };
      images.forEach((img) => {
        if (img.complete) {
          loadedCount++;
        } else {
          img.addEventListener('load', checkAllLoaded, { once: true });
          img.addEventListener('error', checkAllLoaded, { once: true });
        }
      });
      if (loadedCount >= totalImages && !didInit.current) {
        initScroll();
      }
    } else if (!didInit.current) {
      initScroll();
    }

    const filterHeight = filterRef?.current?.offsetHeight;
    if (filterHeight) {
      document.documentElement.style.setProperty('--filter-height', `${filterHeight}px`);
    }

    // Cleanup on unmount
    return () => {
      if (stRef.current) { stRef.current.kill(); stRef.current = null; }
      if (tlRef.current) { tlRef.current.kill(); tlRef.current = null; }
    };
  }, [wrapperRef, containerRef, progressRef, filterRef, items]);
}

/* ===== Main Component ===== */
const FILTERS = ['ALL', 'INTERNSHIP', 'PERSONAL PROJECT', 'COMPETITION', 'COURSE PROJECT'];

export default function WorksSlide({ title, titleEn, items, path = '' }) {
  const [filter, setFilter] = useState('ALL');
  const [filterOpen, setFilterOpen] = useState(false);

  const wrapperRef = useRef(null);
  const containerRef = useRef(null);
  const progressRef = useRef(null);
  const filterRef = useRef(null);

  useHorizontalScroll(wrapperRef, containerRef, progressRef, filterRef, items);

  const handleFilterChange = (e) => {
    setFilter(e.target.value);
  };

  return (
    <>
      <section>
        {/* Horizontal Scroll Area */}
        <div ref={wrapperRef} className="pt-[var(--header-height)]">
          <ul
            ref={containerRef}
            className="flex h-[calc(100vh-calc(var(--header-height)+3.25rem))] flex-nowrap justify-start whitespace-nowrap pb-20 md:h-[calc(100vh-calc(var(--header-height)+var(--filter-height)))]"
          >
            {/* Title + Gold Quote */}
            <li className="mr-20 md:mr-60 flex flex-col items-center">
              <TitleBlock jp={title} en={titleEn} />
            </li>

            {/* Project Cards */}
            {items.map((item, idx) => {
              const hidden = filter !== 'ALL' && !item.tags.includes(filter);
              return (
                <li
                  key={`${item.slug}-${idx}`}
                  className="aspect-works-slide h-full max-w-full animate-text-focus-in whitespace-nowrap border-l border-silver p-[1.4rem] transition-all duration-[1500ms] last-of-type:border-r peer-last:border-r data-[hidden]:max-w-0 data-[hidden]:animate-text-blur-out data-[hidden]:px-0 md:px-12 md:pb-6 md:pt-5"
                  data-tags={item.tags.join(' ')}
                  data-hidden={hidden ? true : null}
                >
                  <a
                    href={`${path}/builder/${item.slug}/`}
                    className="ts-image-link flex h-full flex-col"
                  >
                    <div className="flex">
                      <picture className="flex-1">
                        <img
                          src={`${path}${item.image}`}
                          alt={item.title}
                          className="h-full w-full object-cover"
                        />
                      </picture>
                      <div className="ml-3 flex font-serif-en vertical-rl">
                        <ul className="flex flex-wrap gap-4">
                          {item.categories.map((cat, cIdx) => (
                            <li key={`${item.slug}-${cat}-${cIdx}`}>{cat}</li>
                          ))}
                        </ul>
                        <span className="mt-auto inline-block">{item.year}</span>
                      </div>
                    </div>
                    <div className="mt-4 flex-1 pr-8">
                      <h2
                        className="overflow-hidden text-ellipsis whitespace-normal text-xl font-medium md:text-[1.75rem]"
                        style={{
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                        }}
                      >
                        {item.title}
                      </h2>
                      <ul className="mt-6 flex min-h-[2.2rem] flex-wrap gap-x-4 gap-y-[0.2rem] font-serif-en text-gray">
                        {item.tags.map((tag, tIdx) => (
                          <li key={`${item.slug}-${tag}-${tIdx}`}>{tag}</li>
                        ))}
                      </ul>
                    </div>
                  </a>
                </li>
              );
            })}

            {/* End Link */}
            <li className="flex items-start justify-center pl-24 pr-10 md:pl-72">
              <a
                href="/about"
                className="group inline-flex flex-col items-center text-2xl font-medium md:text-[2rem]"
                style={{ paddingTop: '2rem' }}
              >
                <span className="vertical-rl">
                  <span>述己</span>
                </span>
                <span
                  className="mx-auto mt-6 flex h-[1.5rem] w-[1.5rem] items-center justify-center rounded-full border border-black text-black transition-colors duration-700 group-hover:bg-black group-hover:text-white md:mt-9 md:h-[2rem] md:w-[2rem]"
                  aria-hidden="true"
                >
                  <svg
                    width="12"
                    height="8"
                    viewBox="0 0 12 8"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M5.92843 0.296874C5.92843 0.296874 6.85821 1.97234 7.07889 3.46277L0.628906 3.80681L0.628906 4.29834L7.07265 4.64238C6.82893 6.11193 5.81863 7.70375 5.81863 7.70375C5.81863 7.70375 8.88732 4.4667 11.6966 4.04838C8.8956 3.62585 5.92843 0.296754 5.92843 0.296754L5.92843 0.296874Z"
                      className="fill-current"
                    />
                  </svg>
                </span>
              </a>
            </li>
          </ul>
        </div>

        {/* Filter Bar */}
        <div
          className="group fixed bottom-0 left-0 w-screen data-[open]:z-30"
          data-open={filterOpen || null}
          ref={filterRef}
        >
          <div className="h-[1px] w-full bg-silver">
            <div className="h-full w-0 bg-black" ref={progressRef} />
          </div>

          {/* Mobile overlay */}
          <div
            className="invisible fixed left-0 top-0 h-screen w-screen bg-[rgba(0,0,0,0.5)] opacity-0 transition-all duration-500 group-data-[open]:visible group-data-[open]:opacity-100 md:hidden"
            onClick={() => setFilterOpen(false)}
          />

          <form className="container relative flex items-start overflow-hidden px-5 py-[0.875rem] bg-gray-texture group-data-[open]:z-40 md:px-20 md:py-7">
            <button
              type="button"
              className="flex items-center font-serif-en text-[1.375rem] leading-none md:pointer-events-none"
              onClick={() => setFilterOpen(!filterOpen)}
            >
              FILTER
              <svg
                width="12"
                height="8"
                viewBox="0 0 12 8"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="ml-3 fill-current transition-transform duration-500 group-data-[open]:-rotate-[540deg] md:hidden"
              >
                <path d="M5.90872 7.43347L-0.183594 1.78197L0.907946 0.769409L5.90872 5.43189L10.9095 0.792959L12.001 1.80552L5.90872 7.43347Z" />
              </svg>
            </button>

            <ul className="ml-[7.5rem] flex flex-wrap items-center gap-0 group-data-[open]:gap-4 md:items-baseline md:gap-6">
              {FILTERS.map((f, idx) => {
                const checked = filter === f;
                return (
                  <li
                    key={`${f}-${idx}`}
                    data-checked={checked ? true : null}
                    className="invisible h-0 w-0 animate-text-blur-out opacity-0 transition-all data-[checked]:visible data-[checked]:h-auto data-[checked]:w-auto data-[checked]:animate-text-focus-in data-[checked]:opacity-100 group-data-[open]:visible group-data-[open]:h-auto group-data-[open]:w-auto group-data-[open]:animate-text-focus-in group-data-[open]:opacity-100 md:visible md:h-auto md:w-auto md:animate-none md:opacity-100 md:data-[checked]:animate-none"
                  >
                    <label className="flex items-center font-serif-en">
                      <input
                        className="relative m-0 box-content block h-4 w-4 appearance-none rounded-full border border-black p-0 before:absolute before:left-1/2 before:top-1/2 before:h-3 before:w-3 before:-translate-x-1/2 before:-translate-y-1/2 before:rounded-full checked:before:bg-black md:h-5 md:w-5 md:before:h-[0.875rem] md:before:w-[0.875rem]"
                        type="radio"
                        value={f}
                        checked={checked}
                        onChange={handleFilterChange}
                      />
                      <span className="ml-2 flex-1 md:text-[1.125rem]">{f}</span>
                    </label>
                  </li>
                );
              })}
            </ul>
          </form>
        </div>
      </section>
    </>
  );
}
