// Card-to-PNG export helper. Wraps the CDN-loaded `window.htmlToImage`
// (see frontend/index.html) so any section component can offer "Save as PNG"
// without re-implementing the capture/download dance.
//
// Mark elements you don't want in the screenshot (toolbars, buttons) with the
// `no-export` class.

export async function captureCardAsPng(card, { filename } = {}) {
  if (!window.htmlToImage || !card) return;
  const safeName = String(filename || "card").replace(/[^A-Za-z0-9._-]/g, "_");

  const overlay = document.createElement("div");
  overlay.className = "screenshot-overlay";
  overlay.textContent = "Generating screenshot…";
  document.body.appendChild(overlay);
  await new Promise((r) => requestAnimationFrame(r));
  overlay.style.opacity = "1";
  await new Promise((r) => requestAnimationFrame(r));

  // Any nested `.table-scroll` that's currently scrolled horizontally hides
  // content past its right edge. htmlToImage captures the card's box, so the
  // card must be wide enough to contain every child before we snapshot.
  //
  // The previous attempt relied on `width: max-content` + `overflow: visible`,
  // but `.table-scroll { width: 100% }` doesn't propagate the table's natural
  // width up to `.card`, so the card never actually grew. Here we read each
  // scroller's `scrollWidth` (which already includes the clipped overflow) and
  // apply explicit pixel widths — sidestepping intrinsic-sizing entirely.
  const scrollers = Array.from(card.querySelectorAll(".table-scroll"));
  const savedScrollers = scrollers.map((s) => ({
    overflowX: s.style.overflowX,
    width: s.style.width,
  }));
  const savedCard = {
    width: card.style.width,
    maxWidth: card.style.maxWidth,
  };

  let extraWidth = 0;
  for (const s of scrollers) {
    const diff = s.scrollWidth - s.clientWidth;
    if (diff > extraWidth) extraWidth = diff;
    s.style.width = `${s.scrollWidth}px`;
    s.style.overflowX = "visible";
  }
  if (extraWidth > 0) {
    card.style.width = `${card.clientWidth + extraWidth}px`;
    card.style.maxWidth = "none";
  }

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
    card.style.width = savedCard.width;
    card.style.maxWidth = savedCard.maxWidth;
    scrollers.forEach((s, i) => {
      s.style.overflowX = savedScrollers[i].overflowX;
      s.style.width = savedScrollers[i].width;
    });
    overlay.style.opacity = "0";
    setTimeout(() => overlay.remove(), 150);
  }
}
