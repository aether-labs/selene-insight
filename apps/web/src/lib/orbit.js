/**
 * Orbital positioning for the Earth-Moon-Orion system.
 *
 * Uses a simplified lunar ephemeris + trilateration from
 * earth_dist_km and moon_dist_km to place the spacecraft in 3D.
 */

// Constants
const R_EARTH_KM = 6371;
const MOON_MEAN_DIST_KM = 384400;
const MOON_ORBITAL_PERIOD_S = 27.321661 * 86400; // seconds
const MOON_INCLINATION_RAD = 5.145 * (Math.PI / 180); // to ecliptic
const TWO_PI = 2 * Math.PI;

// J2000 epoch reference: 2000-01-01T12:00:00Z
const J2000_MS = Date.UTC(2000, 0, 1, 12, 0, 0);

// Approximate Moon mean longitude at J2000 (radians)
const MOON_L0_RAD = 3.8104; // ~218.3°

/**
 * Approximate Moon position in Earth-centered inertial (ECI) frame.
 * Returns [x, y, z] in km.
 *
 * This is a mean-motion model — accurate to ~1° over weeks,
 * which is sufficient for visualization (not navigation).
 */
export function moonPositionECI(timestampSec) {
  const dt = (timestampSec * 1000 - J2000_MS) / 1000; // seconds since J2000
  const meanAnomaly = MOON_L0_RAD + (TWO_PI / MOON_ORBITAL_PERIOD_S) * dt;
  const angle = meanAnomaly % TWO_PI;

  // Moon in its orbital plane
  const xOrb = MOON_MEAN_DIST_KM * Math.cos(angle);
  const yOrb = MOON_MEAN_DIST_KM * Math.sin(angle);

  // Rotate by inclination (around x-axis) to get ECI
  const x = xOrb;
  const y = yOrb * Math.cos(MOON_INCLINATION_RAD);
  const z = yOrb * Math.sin(MOON_INCLINATION_RAD);

  return [x, y, z];
}

/**
 * Trilaterate spacecraft position from Earth and Moon distances.
 *
 * Given:
 *   - Earth at origin
 *   - Moon at position M
 *   - d_e = distance from Earth (km)
 *   - d_m = distance from Moon (km)
 *
 * Finds the point P such that |P| = d_e and |P - M| = d_m.
 * Two solutions exist (mirror across Earth-Moon line); we pick
 * the one on the "outbound" side (positive cross-product with
 * Moon's orbital angular momentum).
 *
 * Returns [x, y, z] in km (ECI frame), or null if geometry is invalid.
 */
export function trilaterateSpacecraft(moonECI, earthDistKm, moonDistKm) {
  const [mx, my, mz] = moonECI;
  const moonDist = Math.sqrt(mx * mx + my * my + mz * mz);

  if (moonDist < 1) return null;

  // Unit vector Earth → Moon
  const ux = mx / moonDist;
  const uy = my / moonDist;
  const uz = mz / moonDist;

  // Project spacecraft onto Earth-Moon line:
  // x_along = (d_e² + D² - d_m²) / (2D)
  // where D = |Earth - Moon|
  const D = moonDist;
  const de2 = earthDistKm * earthDistKm;
  const dm2 = moonDistKm * moonDistKm;
  const D2 = D * D;

  const xAlong = (de2 + D2 - dm2) / (2 * D);

  // Perpendicular distance from the Earth-Moon line
  const hSq = de2 - xAlong * xAlong;
  if (hSq < 0) {
    // Distances don't form a valid triangle — clamp to the line
    const clamped = Math.max(0, Math.min(D, xAlong));
    return [ux * clamped, uy * clamped, uz * clamped];
  }

  const h = Math.sqrt(hSq);

  // Build a perpendicular vector in the orbital plane.
  // Use cross product of Moon direction with z-hat, then normalize.
  // If Moon direction is nearly parallel to z, use x-hat instead.
  let px, py, pz;
  if (Math.abs(uz) < 0.9) {
    // cross(u, z_hat) = (uy, -ux, 0)
    const len = Math.sqrt(ux * ux + uy * uy);
    px = uy / len;
    py = -ux / len;
    pz = 0;
  } else {
    // cross(u, x_hat) = (0, uz, -uy)
    const len = Math.sqrt(uz * uz + uy * uy);
    px = 0;
    py = uz / len;
    pz = -uy / len;
  }

  // Spacecraft position: along the line + perpendicular offset
  // We pick the solution with positive perpendicular component
  // (prograde side of the orbit)
  const x = ux * xAlong + px * h;
  const y = uy * xAlong + py * h;
  const z = uz * xAlong + pz * h;

  return [x, y, z];
}

/**
 * Convert ECI km coordinates to Cesium Cartesian3.
 *
 * Maps ECI (x, y, z) to Cesium's Earth-fixed frame:
 *   ECI x → Cesium x (through 0°N 0°E)
 *   ECI y → Cesium y (through 0°N 90°E)
 *   ECI z → Cesium z (through North Pole)
 *
 * Note: we ignore Earth rotation (sidereal time) — the visualization
 * rotates with the inertial frame, which is fine for showing the
 * trajectory shape relative to Earth and Moon.
 */
export function eciToCartesian3(eciKm, Cartesian3) {
  return new Cartesian3(
    eciKm[0] * 1000,
    eciKm[1] * 1000,
    eciKm[2] * 1000,
  );
}

/**
 * Full pipeline: telemetry point → Cartesian3 position.
 */
export function telemetryToPosition(point, Cartesian3) {
  if (!point || !point.earth_dist_km || !point.moon_dist_km) return null;

  const ts = point.timestamp || Date.now() / 1000;
  const moonECI = moonPositionECI(ts);
  const craftECI = trilaterateSpacecraft(
    moonECI,
    point.earth_dist_km,
    point.moon_dist_km,
  );

  if (!craftECI) return null;
  return eciToCartesian3(craftECI, Cartesian3);
}

/**
 * Moon position as Cartesian3 at a given timestamp.
 */
export function moonToPosition(timestampSec, Cartesian3) {
  const eci = moonPositionECI(timestampSec);
  return eciToCartesian3(eci, Cartesian3);
}
