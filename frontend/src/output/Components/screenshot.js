// Card-to-PNG export helper. Wraps the CDN-loaded `window.htmlToImage`
// (see frontend/index.html) so any section component can offer "Save as PNG"
// without re-implementing the capture/download dance.
//
// Mark elements you don't want in the screenshot (toolbars, buttons) with the
// `no-export` class.

export async function captureCardAsPng(card, { filename, expandScrollable = false } = {}) {
  if (!window.htmlToImage || !card) return;
  const safeName = String(filename || "card").replace(/[^A-Za-z0-9._-]/g, "_");

  // Viewport overlay: gives the user immediate feedback on click and masks the
  // card-resize jump (when expandScrollable=true) behind a fading backdrop.
  // Static styling lives in All.css under .screenshot-overlay; opacity is
  // toggled below to drive the fade-in/fade-out.
  const overlay = document.createElement("div");
  overlay.className = "screenshot-overlay";
  overlay.textContent = "Generating screenshot…";
  document.body.appendChild(overlay);
  // Two RAFs: first commits the append, second lets the browser paint the
  // fade-in before we mutate the card and start the capture.
  await new Promise((r) => requestAnimationFrame(r));
  overlay.style.opacity = "1";
  await new Promise((r) => requestAnimationFrame(r));

  // When expandScrollable is true, temporarily lift the .table-scroll clip so
  // the entire table is captured (not just the slice currently visible). The
  // card grows with the table; styles are restored in `finally`.
  const tableScroll = expandScrollable ? card.querySelector(".table-scroll") : null;
  const orig = {
    cardWidth: card.style.width,
    cardMaxWidth: card.style.maxWidth,
    overflowX: tableScroll && tableScroll.style.overflowX,
  };
  if (expandScrollable) {
    if (tableScroll) tableScroll.style.overflowX = "visible";
    card.style.width = "max-content";
    card.style.maxWidth = "none";
  }

  // Match the active theme's card surface so the PNG looks like what's on
  // screen (dark card on dark theme, light card on light theme).
  const cardBg = getComputedStyle(document.documentElement)
    .getPropertyValue("--bg-elev").trim() || "#0f1117";

  try {
    const dataUrl = await window.htmlToImage.toPng(card, {
      backgroundColor: cardBg,
      pixelRatio: 2,
      filter: (node) => !(node.classList && node.classList.contains("no-export")),
    });
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = `${safeName}.png`;
    a.click();
  } finally {
    if (expandScrollable) {
      card.style.width = orig.cardWidth;
      card.style.maxWidth = orig.cardMaxWidth;
      if (tableScroll) tableScroll.style.overflowX = orig.overflowX;
    }
    overlay.style.opacity = "0";
    setTimeout(() => overlay.remove(), 150);
  }
}
