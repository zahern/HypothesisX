import { useEffect, useRef } from "react";

/*
  Thin wrapper around the global Plotly (loaded via CDN in index.html).
  Re-plots whenever data/layout change and purges the graph on unmount so we
  don't leak DOM/listeners as the user switches sections.
*/
export default function Plot({ data, layout, config, style, className }) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || !window.Plotly) return;
    window.Plotly.react(el, data, layout || {}, config || { responsive: true });
  }, [data, layout, config]);

  useEffect(() => {
    const el = ref.current;
    return () => {
      if (el && window.Plotly) window.Plotly.purge(el);
    };
  }, []);

  return <div ref={ref} style={style} className={className} />;
}
