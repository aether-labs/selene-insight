/**
 * Selene-Insight — Open MCT application entry point.
 *
 * Open MCT is loaded globally via <script> tag (UMD bundle).
 * Plugins are ES modules that use the global `openmct` instance.
 */

import { TelemetryPlugin } from "./plugins/telemetryPlugin.js";
import { CesiumViewPlugin } from "./plugins/cesiumViewPlugin.js";
import { AlertPlugin } from "./plugins/alertPlugin.js";

const mct = window.openmct;

mct.setAssetPath("/node_modules/openmct/dist");

// Dark theme
mct.install(mct.plugins.Espresso());

// Our plugins
mct.install(TelemetryPlugin());
mct.install(CesiumViewPlugin());
mct.install(AlertPlugin());

mct.start();
