/**
 * Artemis II reference trajectory generator.
 *
 * Generates a parametric free-return trajectory in ECI coordinates.
 * Uses direct 3D curve generation instead of distance-based trilateration
 * to avoid geometric artifacts (heart shapes from flipping offsets).
 *
 * Artemis II profile (~10 days):
 *   Day 0-4:   Outbound (Earth → Moon)
 *   Day 4-5:   Lunar flyby (loop behind Moon)
 *   Day 5-10:  Return (Moon → Earth)
 */

import { moonPositionECI, eciToCartesian3 } from "./orbit.js";

const MISSION_DAYS = 10;
const STEPS = 600;
const R_EARTH_KM = 6371;
const MOON_DIST_KM = 384400;
const FLYBY_ALT_KM = 100; // closest approach above Moon surface
const MOON_RADIUS_KM = 1737;

/**
 * Generate reference trajectory as array of Cartesian3 positions.
 */
export function generateReferenceTrajectory(launchTimestamp, Cartesian3) {
  const positions = [];
  const dt = (MISSION_DAYS * 86400) / STEPS;

  for (let i = 0; i <= STEPS; i++) {
    const ts = launchTimestamp + i * dt;
    const t = i / STEPS; // 0..1 normalized mission time

    const moonECI = moonPositionECI(ts);
    const moonDist = Math.sqrt(
      moonECI[0] ** 2 + moonECI[1] ** 2 + moonECI[2] ** 2
    );

    // Unit vector Earth → Moon
    const ux = moonECI[0] / moonDist;
    const uy = moonECI[1] / moonDist;
    const uz = moonECI[2] / moonDist;

    // Perpendicular vector (in orbital plane)
    let px, py, pz;
    if (Math.abs(uz) < 0.9) {
      const len = Math.sqrt(ux * ux + uy * uy);
      px = -uy / len;
      py = ux / len;
      pz = 0;
    } else {
      const len = Math.sqrt(uz * uz + uy * uy);
      px = 0;
      py = -uz / len;
      pz = uy / len;
    }

    let x, y, z;

    if (t < 0.40) {
      // Outbound: Earth to Moon, slight curve
      const p = t / 0.40;
      const r = R_EARTH_KM + (moonDist - R_EARTH_KM) * smoothStep(p);
      // Gentle curve off the Earth-Moon line (prograde side)
      const lateral = moonDist * 0.02 * Math.sin(p * Math.PI);
      x = ux * r + px * lateral;
      y = uy * r + py * lateral;
      z = uz * r + pz * lateral;
    } else if (t < 0.50) {
      // Lunar flyby: loop behind Moon
      const p = (t - 0.40) / 0.10;
      const angle = -Math.PI * 0.3 + p * Math.PI * 1.6; // sweep ~290°
      const loopRadius = MOON_RADIUS_KM + FLYBY_ALT_KM + 2000;

      // Out-of-plane component for 3D flyby
      const nz = ux * py - uy * px; // normal to orbital plane
      const ny = -(ux * pz - uz * px);
      const nx = uy * pz - uz * py;
      const nLen = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;

      x = moonECI[0] + (ux * Math.cos(angle) + px * Math.sin(angle)) * loopRadius;
      y = moonECI[1] + (uy * Math.cos(angle) + py * Math.sin(angle)) * loopRadius;
      z = moonECI[2] + (uz * Math.cos(angle) + pz * Math.sin(angle)) * loopRadius;
    } else {
      // Return: Moon back to Earth, wider curve (different side)
      const p = (t - 0.50) / 0.50;
      const r = moonDist - (moonDist - R_EARTH_KM) * smoothStep(p);
      // Return on opposite side of Earth-Moon line
      const lateral = -moonDist * 0.03 * Math.sin(p * Math.PI);
      x = ux * r + px * lateral;
      y = uy * r + py * lateral;
      z = uz * r + pz * lateral;
    }

    positions.push(eciToCartesian3([x, y, z], Cartesian3));
  }

  return positions;
}

/**
 * Generate Moon orbit ring (circle at Moon's distance).
 */
export function generateMoonOrbit(timestampSec, Cartesian3, steps = 200) {
  const positions = [];
  const period = 27.321661 * 86400;

  for (let i = 0; i <= steps; i++) {
    const ts = timestampSec - period / 2 + (period * i) / steps;
    const moonECI = moonPositionECI(ts);
    positions.push(eciToCartesian3(moonECI, Cartesian3));
  }

  return positions;
}

function smoothStep(t) {
  return t * t * (3 - 2 * t);
}

/**
 * Estimate launch timestamp from current telemetry.
 */
export function estimateLaunchTime(telemetryPoint) {
  if (!telemetryPoint) return null;
  const met = telemetryPoint.met;
  if (!met) return null;

  let totalSeconds = 0;

  const matchNew = met.match(/T\+(\d+)d\s+(\d+):(\d+):(\d+)/);
  if (matchNew) {
    totalSeconds =
      parseInt(matchNew[1]) * 86400 +
      parseInt(matchNew[2]) * 3600 +
      parseInt(matchNew[3]) * 60 +
      parseInt(matchNew[4]);
  } else {
    const matchOld = met.match(/(\d+):(\d+):(\d+):(\d+)/);
    if (matchOld) {
      totalSeconds =
        parseInt(matchOld[1]) * 86400 +
        parseInt(matchOld[2]) * 3600 +
        parseInt(matchOld[3]) * 60 +
        parseInt(matchOld[4]);
    }
  }

  if (totalSeconds === 0) return null;
  return telemetryPoint.timestamp - totalSeconds;
}
