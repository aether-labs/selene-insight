/**
 * Programmatic sharp-point starfield for CesiumJS skybox.
 *
 * Generates 6 cubemap face canvases with crisp 1-2px star dots
 * in realistic colors (blue-white-yellow-orange-red based on
 * spectral class distribution).
 */

const STAR_COLORS = [
  [0.6, 0.7, 1.0],   // O/B — blue-white (hot)
  [0.8, 0.85, 1.0],  // A — white-blue
  [1.0, 1.0, 1.0],   // F — white
  [1.0, 0.95, 0.8],  // G — yellow-white (Sun-like)
  [1.0, 0.8, 0.5],   // K — orange
  [1.0, 0.6, 0.3],   // M — red-orange (most common)
];

// Approximate spectral class distribution weights
const COLOR_WEIGHTS = [0.03, 0.05, 0.1, 0.15, 0.25, 0.42];

function pickStarColor() {
  let r = Math.random();
  for (let i = 0; i < COLOR_WEIGHTS.length; i++) {
    r -= COLOR_WEIGHTS[i];
    if (r <= 0) return STAR_COLORS[i];
  }
  return STAR_COLORS[STAR_COLORS.length - 1];
}

/**
 * Generate a single cubemap face as a canvas.
 */
function generateFace(size, starCount, seed) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");

  // Black background
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, size, size);

  // Seeded pseudo-random
  let s = seed;
  function rand() {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  }

  for (let i = 0; i < starCount; i++) {
    const x = rand() * size;
    const y = rand() * size;
    const brightness = rand();

    // Most stars are dim, few are bright (power law)
    const mag = Math.pow(brightness, 3);
    const alpha = 0.3 + mag * 0.7;
    const radius = mag > 0.8 ? 1.5 : mag > 0.4 ? 1.0 : 0.5;

    const [cr, cg, cb] = pickStarColor();

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${Math.floor(cr * 255)}, ${Math.floor(cg * 255)}, ${Math.floor(cb * 255)}, ${alpha})`;
    ctx.fill();

    // Bright stars get a subtle glow
    if (mag > 0.85) {
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${Math.floor(cr * 255)}, ${Math.floor(cg * 255)}, ${Math.floor(cb * 255)}, ${mag * 0.15})`;
      ctx.fill();
    }
  }

  return canvas;
}

/**
 * Create 6 cubemap face canvases for CesiumJS SkyBox.
 */
export function createStarfieldSkyboxSources(size = 2048, starsPerFace = 3000) {
  return {
    positiveX: generateFace(size, starsPerFace, 12345),
    negativeX: generateFace(size, starsPerFace, 23456),
    positiveY: generateFace(size, starsPerFace, 34567),
    negativeY: generateFace(size, starsPerFace, 45678),
    positiveZ: generateFace(size, starsPerFace, 56789),
    negativeZ: generateFace(size, starsPerFace, 67890),
  };
}
