/* PathSnap M4 evaluator UI. No model predictions or reference labels are hardcoded. */
'use strict';
const $ = id => document.getElementById(id);
const state = { api: '', map: null, gpsLayer: null, routeLayer: null, tileLayer: null };
const connectButton = $('connectBtn');
const predictButton = $('predictBtn');
const storedUrl = localStorage.getItem('pathsnap_m4_backend');
if (storedUrl) $('apiAddress').value = storedUrl;

function message(text, tone = '') {
  const element = $('statusMessage');
  element.textContent = text;
  element.className = `info-box ${tone}`;
}
function connection(connected, isError = false, mode = '') {
  const pill = $('connectionPill');
  pill.className = `status-pill ${connected ? 'ready' : (isError ? 'error' : '')}`;
  pill.innerHTML = '<i></i>' + (connected ? `Connected · ${mode}` : (isError ? 'Connection failed' : 'Disconnected'));
}
function getApi() {
  const value = $('apiAddress').value.trim().replace(/\/+$/, '');
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Use a valid http:// or https:// URL.');
  return url.origin;
}
async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try {
    const resp = await fetch(state.api + path, { signal: controller.signal, ...options });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || `API returned ${resp.status}`);
    return data;
  } finally { clearTimeout(timeout); }
}
function initMap() {
  if (typeof L === 'undefined') {
    message('Leaflet did not load. Check internet connectivity for the map libraries.', 'error');
    return false;
  }
  if (!state.map) {
    state.map = L.map('map', { preferCanvas: true }).setView([12.95, 77.61], 13);
    state.tileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'
    }).addTo(state.map);
    state.gpsLayer = L.layerGroup().addTo(state.map);
    state.routeLayer = L.layerGroup().addTo(state.map);
    L.control.layers({}, { 'Noisy GPS': state.gpsLayer, 'Predicted route': state.routeLayer }, { collapsed: false }).addTo(state.map);
  }
  return true;
}
async function connect() {
  connectButton.disabled = true;
  predictButton.disabled = true;
  message('Contacting the backend on Laptop 1…');
  try {
    state.api = getApi();
    const health = await request('/api/health');
    const listing = await request('/api/trips');
    connection(true, false, health.mode);
    localStorage.setItem('pathsnap_m4_backend', state.api);
    const select = $('tripSelect');
    select.replaceChildren();
    for (const trip of listing.trips) {
      const opt = document.createElement('option');
      opt.value = trip.id;
      opt.textContent = trip.id;
      select.appendChild(opt);
    }
    const n = listing.trips.length;
    $('tripCount').textContent = `${n} trip${n === 1 ? '' : 's'} available${health.mode === 'demo' ? ' · SYNTHETIC DEMO' : ''}`;
    if (!n) {
      const opt = document.createElement('option');
      opt.textContent = 'No .track/.nodes/.arcs trip files found'; opt.value = '';
      select.appendChild(opt);
    }
    select.disabled = n === 0;
    predictButton.disabled = n === 0 || !health.model_configured;
    message(health.status === 'ready'
      ? `Server is ready (${health.mode}). ${health.data_hint || ''} Choose a trip and run map matching.`
      : `Server connected but not ready. ${health.data_hint || 'Check trip files and the M3 predictor configuration.'}`,
      health.status === 'ready' ? 'success' : '');
    $('predictorLabel').textContent = `Backend: ${state.api} • Mode: ${health.mode}`;
    initMap();
  } catch (e) {
    connection(false, true);
    message(`Could not connect: ${e.message}. Verify Laptop 1 IP address, port 8000, firewall and CORS.`, 'error');
  } finally { connectButton.disabled = false; }
}
function drawRoute(payload) {
  if (!initMap()) return;
  state.gpsLayer.clearLayers();
  state.routeLayer.clearLayers();
  const raw = payload.gps_points;
  const segments = payload.predicted_route.segments;
  if (raw.length) {
    L.polyline(raw, { color: '#2678eb', weight: 2, opacity: .63, dashArray: '5 5' }).addTo(state.gpsLayer);
    const stride = Math.max(1, Math.ceil(raw.length / 300));
    for (let i = 0; i < raw.length; i += stride)
      L.circleMarker(raw[i], { radius: 3, fillColor: '#2678eb', fillOpacity: .8,
        color: '#fff', weight: .7 }).addTo(state.gpsLayer);
    L.circleMarker(raw[0], { radius: 6, color: '#1357ae', fillColor: '#1357ae', fillOpacity: 1 }).bindTooltip('GPS start').addTo(state.gpsLayer);
    L.circleMarker(raw[raw.length-1], { radius: 6, color: '#174ca2', fillColor: '#fff', fillOpacity: 1 }).bindTooltip('GPS end').addTo(state.gpsLayer);
  }
  for (const part of segments) {
    L.polyline(part.coordinates, { color: '#009f73', weight: 6, opacity: .92, lineCap: 'round' })
      .bindTooltip(`Predicted road arc: ${part.arc_id}`).addTo(state.routeLayer);
  }
  if (raw.length) state.map.fitBounds(L.latLngBounds(raw).pad(.22), { maxZoom: 17 });
  $('mapFallback').classList.add('hidden');
  setTimeout(() => state.map.invalidateSize(), 50);
}
async function predict() {
  const tripId = $('tripSelect').value;
  if (!tripId) return;
  predictButton.disabled = true;
  message(`Processing trip ${tripId} on Laptop 1…`);
  try {
    const payload = await request('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trip_id: tripId })
    });
    $('activeTrip').textContent = payload.trip_id;
    $('observationsValue').textContent = payload.observation_count.toLocaleString();
    const score = payload.evaluation;
    $('accuracyValue').textContent = score.available ? `${score.route_accuracy_percent}%` : 'N/A';
    $('continuityValue').textContent = `${payload.predicted_route.continuity_percent}%`;
    $('arcsValue').textContent = payload.predicted_route.ordered_arcs.length.toLocaleString();
    $('predictorLabel').textContent = `Predictor: ${payload.predictor} · Mode: ${payload.mode}`;
    drawRoute(payload);
    const quality = payload.predicted_route.continuity_percent;
    message(`Trip ${payload.trip_id}: inference complete. ${score.available
      ? `Server-side sequence score: ${score.route_accuracy_percent}%.`
      : 'No reference route is available, so scoring was not performed.'}
      ${quality < 100 ? 'Some predicted road segments are not directly connected.' : 'Predicted route segments are continuous.'}`,
      score.available ? 'success' : '');
  } catch (err) { message(`Prediction error: ${err.message}`, 'error'); }
  finally { predictButton.disabled = false; }
}
connectButton.addEventListener('click', connect);
predictButton.addEventListener('click', predict);
$('apiAddress').addEventListener('keydown', e => { if (e.key === 'Enter') connect(); });
