/**
 * SENTINEL Dashboard JavaScript
 * Live feed + stats logic with SSE and Sparkline
 */

(function() {
    'use strict';

    // ═══════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════
    
    const state = {
        connected: false,
        sessionStart: null,
        stats: {},
        fpsHistory: [],
        vehicleHistory: [],
        latencyHistory: [],
    };

    // ═══════════════════════════════════════════════════════════
    // SPARKLINE CLASS
    // ═══════════════════════════════════════════════════════════
    
    class Sparkline {
        constructor(canvas, options = {}) {
            this.canvas = canvas;
            this.ctx = canvas.getContext('2d');
            this.data = [];
            this.maxPoints = options.maxPoints || 60;
            this.color = options.color || '#00d4ff';
            this.fillColor = options.fillColor || 'rgba(0, 212, 255, 0.1)';
            this.lineWidth = options.lineWidth || 2;
            this.minValue = options.minValue;
            this.maxValue = options.maxValue;
            
            this.resize();
            window.addEventListener('resize', () => this.resize());
        }
        
        resize() {
            const rect = this.canvas.parentElement.getBoundingClientRect();
            this.canvas.width = rect.width;
            this.canvas.height = rect.height;
        }
        
        push(value) {
            this.data.push(value);
            if (this.data.length > this.maxPoints) {
                this.data.shift();
            }
            this.draw();
        }
        
        draw() {
            const ctx = this.ctx;
            const w = this.canvas.width;
            const h = this.canvas.height;
            
            // Clear
            ctx.clearRect(0, 0, w, h);
            
            if (this.data.length < 2) return;
            
            // Calculate bounds
            let min = this.minValue ?? Math.min(...this.data);
            let max = this.maxValue ?? Math.max(...this.data);
            
            // Ensure some range
            if (max === min) {
                max = min + 1;
            }
            
            const range = max - min;
            const padding = h * 0.1;
            const graphH = h - padding * 2;
            
            // Calculate points
            const points = this.data.map((v, i) => ({
                x: (i / (this.maxPoints - 1)) * w,
                y: padding + graphH - ((v - min) / range) * graphH,
            }));
            
            // Draw fill
            ctx.beginPath();
            ctx.moveTo(points[0].x, h);
            points.forEach(p => ctx.lineTo(p.x, p.y));
            ctx.lineTo(points[points.length - 1].x, h);
            ctx.closePath();
            ctx.fillStyle = this.fillColor;
            ctx.fill();
            
            // Draw line
            ctx.beginPath();
            ctx.moveTo(points[0].x, points[0].y);
            points.forEach(p => ctx.lineTo(p.x, p.y));
            ctx.strokeStyle = this.color;
            ctx.lineWidth = this.lineWidth;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.stroke();
            
            // Draw end dot
            const last = points[points.length - 1];
            ctx.beginPath();
            ctx.arc(last.x, last.y, 3, 0, Math.PI * 2);
            ctx.fillStyle = this.color;
            ctx.fill();
        }
    }

    // ═══════════════════════════════════════════════════════════
    // SSE CONNECTION
    // ═══════════════════════════════════════════════════════════
    
    let eventSource = null;
    let reconnectTimeout = null;
    
    function connectSSE() {
        if (eventSource) {
            eventSource.close();
        }
        
        eventSource = new EventSource('/stream');
        
        eventSource.onopen = function() {
            state.connected = true;
            updateConnectionStatus(true);
            console.log('[SSE] Connected');
        };
        
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                handleStatsUpdate(data);
            } catch (e) {
                console.error('[SSE] Parse error:', e);
            }
        };
        
        eventSource.onerror = function(error) {
            state.connected = false;
            updateConnectionStatus(false);
            console.error('[SSE] Error:', error);
            
            // Reconnect after delay
            eventSource.close();
            clearTimeout(reconnectTimeout);
            reconnectTimeout = setTimeout(connectSSE, 3000);
        };
    }
    
    function updateConnectionStatus(connected) {
        const statusDot = document.querySelector('.status-dot[data-system="sse"]');
        if (statusDot) {
            statusDot.classList.toggle('error', !connected);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // UI UPDATE
    // ═══════════════════════════════════════════════════════════
    
    let pendingUpdate = false;
    
    function handleStatsUpdate(data) {
        state.stats = data;
        
        // Store history for sparklines
        if (typeof data.fps === 'number') {
            state.fpsHistory.push(data.fps);
            if (state.fpsHistory.length > 60) state.fpsHistory.shift();
        }
        
        if (typeof data.vehicles === 'number') {
            state.vehicleHistory.push(data.vehicles);
            if (state.vehicleHistory.length > 60) state.vehicleHistory.shift();
        }
        
        // Batch DOM updates
        if (!pendingUpdate) {
            pendingUpdate = true;
            requestAnimationFrame(() => {
                updateUI(data);
                pendingUpdate = false;
            });
        }
    }
    
    function updateUI(data) {
        // Update FPS
        const fpsEl = document.getElementById('fps-value');
        if (fpsEl) fpsEl.textContent = `${data.fps || 0} FPS`;
        
        // Update vehicle count
        const vehicleEl = document.getElementById('vehicle-count');
        if (vehicleEl) vehicleEl.textContent = data.vehicles || 0;
        
        // Update plate count
        const plateEl = document.getElementById('plate-count');
        if (plateEl) plateEl.textContent = data.plates || 0;
        
        // Update threat level
        updateThreatLevel(data.threat_level);
        
        // Update camera status
        updateCameraStatus(data.camera_status);
        
        // Update active tracks
        updateActiveTracks(data.active_tracks || []);
        
        // Update detection counts
        updateDetectionCounts(data);
        
        // Update detection log
        updateDetectionLog(data.detections || []);
        
        // Update sparklines
        updateSparklines();
    }
    
    function updateThreatLevel(level) {
        const indicator = document.getElementById('threat-indicator');
        if (!indicator) return;
        
        indicator.className = 'threat-indicator';
        
        switch (level) {
            case 'HIGH':
                indicator.classList.add('high');
                indicator.innerHTML = '⚠ HIGH THREAT';
                break;
            case 'ELEVATED':
                indicator.classList.add('elevated');
                indicator.innerHTML = '◈ ELEVATED';
                break;
            default:
                indicator.classList.add('clear');
                indicator.innerHTML = '◉ CLEAR';
        }
    }
    
    function updateCameraStatus(status) {
        const dot = document.querySelector('.status-dot[data-system="camera"] .dot');
        const label = document.querySelector('.status-dot[data-system="camera"]');
        
        if (dot) {
            dot.style.background = status === 'connected' ? 'var(--accent-clear)' : 
                                   status === 'error' ? 'var(--accent-threat)' : 'var(--accent-warning)';
        }
        
        const statusRow = document.getElementById('camera-status-value');
        if (statusRow) {
            statusRow.textContent = status || 'unknown';
            statusRow.className = 'value ' + (status === 'connected' ? 'connected' : 'error');
        }
    }
    
    function updateActiveTracks(tracks) {
        const container = document.getElementById('active-tracks');
        if (!container) return;
        
        if (tracks.length === 0) {
            container.innerHTML = '<div class="empty-state"><div class="empty-state-text">No active tracks</div></div>';
            return;
        }
        
        container.innerHTML = tracks.slice(0, 10).map(track => `
            <div class="track-item ${track.class_name === 'military_vehicle' ? 'military' : ''}">
                <div class="track-id">#${track.track_id}</div>
                <div class="track-type">${formatClassName(track.class_name)}</div>
                <div class="track-conf">${(track.confidence * 100).toFixed(0)}%</div>
            </div>
        `).join('');
    }
    
    function updateDetectionCounts(data) {
        const commercial = data.commercial_count || 0;
        const military = data.military_count || 0;
        const total = commercial + military || 1;
        
        // Update commercial bar
        const commercialValue = document.getElementById('commercial-count');
        const commercialFill = document.getElementById('commercial-fill');
        if (commercialValue) commercialValue.textContent = commercial;
        if (commercialFill) commercialFill.style.width = `${(commercial / total) * 100}%`;
        
        // Update military bar
        const militaryValue = document.getElementById('military-count');
        const militaryFill = document.getElementById('military-fill');
        if (militaryValue) militaryValue.textContent = military;
        if (militaryFill) militaryFill.style.width = `${(military / total) * 100}%`;
    }
    
    function updateDetectionLog(detections) {
        const container = document.getElementById('detection-log');
        if (!container) return;
        
        if (detections.length === 0) {
            container.innerHTML = '<div class="empty-state"><div class="empty-state-text">No detections yet</div></div>';
            return;
        }
        
        const newHTML = detections.slice(0, 20).map(det => `
            <div class="log-entry ${det.type === 'military_vehicle' ? 'military' : ''}">
                <div class="log-entry-id">${det.track_id ? '#' + det.track_id : '—'}</div>
                <div class="log-entry-info">
                    <div class="log-entry-type">${formatClassName(det.type)}</div>
                    ${det.plate ? `<div class="log-entry-plate">${det.plate}</div>` : ''}
                </div>
                <div class="log-entry-time">${det.time}</div>
            </div>
        `).join('');
        
        // Only update if content changed
        if (container.innerHTML !== newHTML) {
            container.innerHTML = newHTML;
        }
    }
    
    function formatClassName(name) {
        if (!name) return 'Unknown';
        return name.replace(/[-_]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    }

    // ═══════════════════════════════════════════════════════════
    // SPARKLINES
    // ═══════════════════════════════════════════════════════════
    
    let fpsSparkline, vehicleSparkline, latencySparkline;
    
    function initSparklines() {
        const fpsCanvas = document.getElementById('fps-sparkline');
        const vehicleCanvas = document.getElementById('vehicle-sparkline');
        const latencyCanvas = document.getElementById('latency-sparkline');
        
        if (fpsCanvas) {
            fpsSparkline = new Sparkline(fpsCanvas, {
                color: '#00d4ff',
                fillColor: 'rgba(0, 212, 255, 0.1)',
                minValue: 0,
            });
        }
        
        if (vehicleCanvas) {
            vehicleSparkline = new Sparkline(vehicleCanvas, {
                color: '#00ff88',
                fillColor: 'rgba(0, 255, 136, 0.1)',
                minValue: 0,
            });
        }
        
        if (latencyCanvas) {
            latencySparkline = new Sparkline(latencyCanvas, {
                color: '#ffcc00',
                fillColor: 'rgba(255, 204, 0, 0.1)',
                minValue: 0,
            });
        }
    }
    
    function updateSparklines() {
        if (fpsSparkline && state.fpsHistory.length > 0) {
            fpsSparkline.data = [...state.fpsHistory];
            fpsSparkline.draw();
            
            const avgFps = state.fpsHistory.reduce((a, b) => a + b, 0) / state.fpsHistory.length;
            const fpsAvgEl = document.getElementById('fps-avg');
            if (fpsAvgEl) fpsAvgEl.textContent = avgFps.toFixed(1);
        }
        
        if (vehicleSparkline && state.vehicleHistory.length > 0) {
            vehicleSparkline.data = [...state.vehicleHistory];
            vehicleSparkline.draw();
            
            const total = state.vehicleHistory.reduce((a, b) => a + b, 0);
            const vehicleTotalEl = document.getElementById('vehicle-total');
            if (vehicleTotalEl) vehicleTotalEl.textContent = total;
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MISSION TIMER
    // ═══════════════════════════════════════════════════════════
    
    function startMissionTimer() {
        const timerEl = document.getElementById('mission-timer');
        if (!timerEl) return;
        
        const startTime = Date.now();
        
        function updateTimer() {
            const elapsed = Date.now() - startTime;
            const hours = Math.floor(elapsed / 3600000);
            const minutes = Math.floor((elapsed % 3600000) / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            
            timerEl.textContent = 
                String(hours).padStart(2, '0') + ':' +
                String(minutes).padStart(2, '0') + ':' +
                String(seconds).padStart(2, '0');
        }
        
        updateTimer();
        setInterval(updateTimer, 1000);
    }

    // ═══════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═══════════════════════════════════════════════════════════
    
    function init() {
        console.log('[SENTINEL] Dashboard initializing...');
        
        // Start SSE connection
        connectSSE();
        
        // Initialize sparklines
        initSparklines();
        
        // Start mission timer
        startMissionTimer();
        
        // Load initial config
        loadConfig();
        
        console.log('[SENTINEL] Dashboard ready');
    }
    
    // Config loading for config panel
    async function loadConfig() {
        try {
            const response = await fetch('/config');
            const config = await response.json();
            
            // Update slider values if present
            Object.entries(config.vehicle_conf || {}).forEach(([key, value]) => {
                const slider = document.querySelector(`input[data-config="vehicle_conf.${key}"]`);
                if (slider) {
                    slider.value = value;
                    const display = slider.parentElement.querySelector('.slider-value');
                    if (display) display.textContent = value.toFixed(2);
                }
            });
            
            Object.entries(config.plate_conf || {}).forEach(([key, value]) => {
                const slider = document.querySelector(`input[data-config="plate_conf.${key}"]`);
                if (slider) {
                    slider.value = value;
                    const display = slider.parentElement.querySelector('.slider-value');
                    if (display) display.textContent = value.toFixed(2);
                }
            });
        } catch (e) {
            console.error('[Config] Load error:', e);
        }
    }
    
    // Export for global access
    window.SENTINEL = {
        state,
        connectSSE,
        Sparkline,
    };
    
    // Start when DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
