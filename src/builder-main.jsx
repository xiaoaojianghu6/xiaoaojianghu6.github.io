import { createRoot } from 'react-dom/client';
import WorksSlide from './components/WorksSlide';

const mountEl = document.getElementById('works-slide-root');
if (mountEl) {
  const propsJson = mountEl.getAttribute('data-props');
  if (propsJson) {
    const props = JSON.parse(propsJson);
    const root = createRoot(mountEl);
    root.render(<WorksSlide {...props} />);
  }
}
