/**
 * Artemis II reference trajectory generator.
 *
 * Generates an approximate free-return trajectory around the Moon
 * based on mission parameters. This is a visualization aid, not
 * a navigation solution.
 *
 * Artemis II profile (~10 days):
 *   Day 0:     Launch + TLI burn → depart Earth
 *   Day 0-4:   Outbound coast (Earth→Moon)
 *   Day 4-5:   Lunar flyby (closest approach ~100 km above surface)
 *   Day 5-9:   Return coast (Moon→Earth)
 *   Day 10:    Re-entry
 */

import { moonPositionECI, trilaterateSpacecraft, eciToCartesian3 } from "./orbit.js";

// Mission duration in days
const MISSION_DAYS = 10;
const STEPS = 500; // trajectory resolution

// Approximate distance profile (km from Earth center)
// Based on Artemis II free-return trajectory shape
function earthDistanceProfile(t) {
  // t: 0..1 normalized mission time
  // Outbound: accelerating away from Earth
  if (t < 0.42) {
    const p = t / 0.42;
    // Smooth acceleration from LEO to lunar distance
    return 6571 + (384400 - 6571) * (1 - Math.cos(p * Math.PI)) / 2;
  }
  // Lunar flyby: closest to Moon at t=0.42-0.48
  if (t < 0.48) {
    const p = (t - 0.42) / 0.06;
    // Swing around Moon — distance from Earth stays near lunar distance
    const base = 384400;
    const dip = 20000; // slight dip as trajectory curves
    return base - dip * Math.sin(p * Math.PI);
  }
  // Return: decelerating back toward Earth
  const p = (t - 0.48) / 0.52;
  return 384400 - (384400 - 6571) * (1 - Math.cos(p * Math.PI)) / 2;
}

function moonDistanceProfile(earthDist) {
  // Approximate: Moon is ~384400 km from Earth
  // This is simplified — real distance depends on angle, not just radial distance
  // For the reference line, derive from earth dist and moon dist constraint
  const moonDist = Math.abs(384400 - earthDist) + 100; // minimum 100 km flyby
  return Math.max(moonDist, 100 + 1737); // above lunar surface
}

/**
 * Generate reference trajectory as array of ECI positions.
 *
 * @param {number} launchTimestamp - Unix timestamp of mission start
 * @param {object} Cartesian3 - Cesium Cartesian3 class
 * @returns {Array} Array of Cartesian3 positions
 */
export function generateReferenceTrajectory(launchTimestamp, Cartesian3) {
  const positions = [];
  const dt = (MISSION_DAYS * 86400) / STEPS;

  for (let i = 0; i <= STEPS; i++) {
    const ts = launchTimestamp + i * dt;
    const t = i / STEPS; // normalized 0..1

    const earthDist = earthDistanceProfile(t);
    const moonDist = moonDistanceProfile(earthDist);

    const moonECI = moonPositionECI(ts);
    const craftECI = trilaterateSpacecraft(moonECI, earthDist, moonDist);

    if (craftECI) {
      positions.push(eciToCartesian3(craftECI, Cartesian3));
    }
  }

  return positions;
}

/**
 * Estimate launch timestamp from current telemetry.
 * Uses MET string to calculate when T+0 was.
 */
export function estimateLaunchTime(telemetryPoint) {
  if (!telemetryPoint) return null;

  const met = telemetryPoint.met;
  if (!met) return null;

  // Parse MET formats: "T+1d 11:27:31" or "001:11:27:31"
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
