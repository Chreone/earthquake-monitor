// Map Setup
const map = L.map('map').setView([7.5, 125.0], 7);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '© OpenStreetMap',
  subdomains: 'abcd',
  maxZoom: 18
}).addTo(map);

let heatLayer = null;
const markerLayer = L.layerGroup();
const predictionLayer = L.layerGroup();

// Chart instances
let trendsChart = null;
let magnitudeChart = null;

// Navigation
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    const view = item.dataset.view;
    document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
    item.classList.add('active');
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(`${view}View`).classList.add('active');
    if (view === 'analytics') loadCharts();
  });
});

// Load Data
async function loadStats() {
  const res = await fetch('/api/stats');
  const data = await res.json();
  
  document.getElementById('totalEvents').innerText = data.total_records.toLocaleString();
  document.getElementById('maxMag').innerText = `M${data.max_magnitude.toFixed(1)}`;
  document.getElementById('avgMag').innerText = data.avg_magnitude.toFixed(1);
  document.getElementById('avgDepth').innerText = `${data.depth_avg.toFixed(0)} km`;
  document.getElementById('maxDepth').innerText = `${data.depth_max.toFixed(0)} km`;
  
  // Top provinces
  const provinceHtml = Object.entries(data.by_province).slice(0, 6).map(([name, count]) => `
    <div class="province-item">
      <span class="province-name">${name}</span>
      <span class="province-count">${count.toLocaleString()} events</span>
    </div>
  `).join('');
  document.getElementById('topProvincesList').innerHTML = provinceHtml;
  
  // Store for charts
  window.statsData = data;
}

// Load Charts
async function loadCharts() {
  if (!window.statsData) return;
  const data = window.statsData;
  
  // Trends Chart
  if (trendsChart) trendsChart.destroy();
  const ctx1 = document.getElementById('trendsChart').getContext('2d');
  trendsChart = new Chart(ctx1, {
    type: 'line',
    data: {
      labels: data.monthly_trends.months,
      datasets: [{
        label: 'Earthquake Count',
        data: data.monthly_trends.counts,
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.05)',
        fill: true,
        tension: 0.4
      }]
    },
    options: { responsive: true, maintainAspectRatio: true }
  });
  
  // Magnitude Distribution
  if (magnitudeChart) magnitudeChart.destroy();
  const ctx2 = document.getElementById('magnitudeChart').getContext('2d');
  magnitudeChart = new Chart(ctx2, {
    type: 'bar',
    data: {
      labels: Object.keys(data.magnitude_distribution),
      datasets: [{
        label: 'Frequency',
        data: Object.values(data.magnitude_distribution),
        backgroundColor: '#f59e0b',
        borderRadius: 8
      }]
    },
    options: { responsive: true, maintainAspectRatio: true }
  });
}

// Load Map Layers
fetch('/api/earthquakes')
  .then(r => r.json())
  .then(points => {
    const heatPoints = points.map(p => [p[0], p[1], Math.min(p[2] / 6, 0.8)]);
    heatLayer = L.heatLayer(heatPoints, { radius: 20, blur: 15 });
    heatLayer.addTo(map);
  });

fetch('/api/significant')
  .then(r => r.json())
  .then(quakes => {
    quakes.forEach(q => {
      L.circleMarker([q.Latitude, q.Longitude], {
        radius: 6 + q.Magnitude / 2,
        color: '#ef4444',
        fillColor: '#ef4444',
        fillOpacity: 0.5,
        weight: 1.5
      }).bindPopup(`
        <b>M${q.Magnitude.toFixed(1)}</b><br>
        ${q.Specific_Location}<br>
        ${new Date(q.Datetime).toLocaleDateString()}
      `).addTo(markerLayer);
    });
    markerLayer.addTo(map);
  });

fetch('/api/predictions')
  .then(r => r.json())
  .then(data => {
    if (data.cells?.length) {
      document.getElementById('forecastDate').innerText = `As of ${new Date(data.as_of).toLocaleDateString()}`;
      const grid = document.getElementById('forecastGrid');
      grid.innerHTML = data.cells.slice(0, 12).map(cell => {
        const risk = cell.risk_probability;
        let riskClass = 'low';
        if (risk > 0.6) riskClass = 'critical';
        else if (risk > 0.35) riskClass = 'high';
        else if (risk > 0.15) riskClass = 'moderate';
        return `
          <div class="forecast-cell risk-${riskClass}">
            <div style="font-weight: 600; margin-bottom: 8px;">Region ${cell.bounds[0][0].toFixed(1)}°N, ${cell.bounds[0][1].toFixed(1)}°E</div>
            <div style="font-size: 1.5rem; font-weight: 700;">${(risk * 100).toFixed(0)}%</div>
            <div style="font-size: 0.7rem; color: #8a92a6;">Expected M${cell.expected_max_magnitude.toFixed(1)} max</div>
          </div>
        `;
      }).join('');
      
      data.cells.forEach(cell => {
        const risk = cell.risk_probability;
        const color = risk < 0.15 ? '#10b981' : risk < 0.35 ? '#f59e0b' : risk < 0.6 ? '#f97316' : '#ef4444';
        L.rectangle(cell.bounds, {
          color: color,
          weight: 1,
          fillColor: color,
          fillOpacity: 0.2
        }).addTo(predictionLayer);
      });
      predictionLayer.addTo(map);
    }
  });

// Layer Toggles
document.getElementById('toggleHeat').addEventListener('change', e => {
  if (e.target.checked && heatLayer) heatLayer.addTo(map);
  else if (heatLayer) map.removeLayer(heatLayer);
});
document.getElementById('toggleMarkers').addEventListener('change', e => {
  e.target.checked ? markerLayer.addTo(map) : map.removeLayer(markerLayer);
});
document.getElementById('togglePredictions').addEventListener('change', e => {
  e.target.checked ? predictionLayer.addTo(map) : map.removeLayer(predictionLayer);
});

// Chat
let chatOpen = true;
window.toggleChat = () => {
  chatOpen = !chatOpen;
  document.getElementById('chatWidget').classList.toggle('collapsed', !chatOpen);
};

const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');

function addMessage(text, sender) {
  const div = document.createElement('div');
  div.className = `message ${sender}`;
  div.textContent = text;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function sendMessage() {
  const msg = chatInput.value.trim();
  if (!msg) return;
  addMessage(msg, 'user');
  chatInput.value = '';
  
  const loading = document.createElement('div');
  loading.className = 'message bot';
  loading.textContent = '...';
  chatMessages.appendChild(loading);
  
  fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: msg })
  })
  .then(r => r.json())
  .then(data => {
    loading.remove();
    addMessage(data.reply, 'bot');
  })
  .catch(() => {
    loading.remove();
    addMessage('Connection error. Please try again.', 'bot');
  });
}

document.getElementById('chatSendBtn').onclick = sendMessage;
chatInput.onkeypress = (e) => { if (e.key === 'Enter') sendMessage(); };

// Initialize
loadStats();