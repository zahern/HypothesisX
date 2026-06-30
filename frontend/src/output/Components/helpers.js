// Shared constants, significance-class helpers, and the Plotly layout theme.
// Colours are picked to read both on the dark card surface (#0f1117) and on
// the light card surface (#f9fafb).

import { useEffect, useState } from "react";

export const COLORS = ['#60a5fa', '#f87171', '#4ade80', '#c084fc', '#fbbf24'];

// ── Plotly layout fragments ──────────────────────────────────────────────
// Spread one of these into each chart's layout, then spread the axis defaults
// into any axis you override (so titles/ranges merge cleanly):
//   layout={{ ...PLOT_THEME, xaxis: { ...PLOT_THEME.xaxis, title: 'Iteration' }, ... }}
//
// `titleColor` and `obsLine` are project-specific extras (Plotly ignores
// unknown layout keys) so components can pick the right shade for chart
// titles / annotations / reference lines without hard-coding hex codes.

const PLOT_DARK = {
  paper_bgcolor: '#0f1117',
  plot_bgcolor:  '#0f1117',
  font: { color: '#cbd5e1', size: 12 },
  xaxis: { gridcolor: '#1e2533', zerolinecolor: '#1e2533', linecolor: '#1e2533', tickcolor: '#1e2533' },
  yaxis: { gridcolor: '#1e2533', zerolinecolor: '#1e2533', linecolor: '#1e2533', tickcolor: '#1e2533' },
  legend: { font: { color: '#cbd5e1' } },
  colorway: COLORS,
  titleColor: '#f1f5f9',
  obsLine:    '#e2e8f0',
};

const PLOT_LIGHT = {
  paper_bgcolor: '#f9fafb',
  plot_bgcolor:  '#f9fafb',
  font: { color: '#334155', size: 12 },
  xaxis: { gridcolor: '#e2e8f0', zerolinecolor: '#e2e8f0', linecolor: '#cbd5e1', tickcolor: '#cbd5e1' },
  yaxis: { gridcolor: '#e2e8f0', zerolinecolor: '#e2e8f0', linecolor: '#cbd5e1', tickcolor: '#cbd5e1' },
  legend: { font: { color: '#334155' } },
  colorway: COLORS,
  titleColor: '#0f172a',
  obsLine:    '#475569',
};

// Backward-compat constant alias. New code should call usePlotTheme() so the
// layout flips when the user toggles Dark/Light.
export const PLOT_THEME = PLOT_DARK;

function currentTheme() {
  return document.documentElement.getAttribute('dcm-studio-theme') === 'light' ? PLOT_LIGHT : PLOT_DARK;
}

// React hook: returns the Plotly layout fragment matching the active theme and
// re-renders the caller when the `dcm-studio-theme` attribute on <html> changes.
export function usePlotTheme() {
  const [t, setT] = useState(currentTheme);
  useEffect(() => {
    const obs = new MutationObserver(() => setT(currentTheme()));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['dcm-studio-theme'] });
    return () => obs.disconnect();
  }, []);
  return t;
}

export function sigCls(s) {
  /*
    returns a class based of the significance (represented as asterix ***)
  */
  return s === '***' ? 'sig-3' : s === '**' ? 'sig-2' : s === '*' ? 'sig-1' : s === '.' ? 'sig-dot' : 'sig-ns';
}

export function sbgCls(s) {
  return s === '***' ? 'sbg-3' : s === '**' ? 'sbg-2' : s === '*' ? 'sbg-1' : s === '.' ? 'sbg-dot' : 'sbg-ns';
}
