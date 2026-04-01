/**
 * SENTINEL Analytics JavaScript
 * Charts and analytics page logic using Chart.js
 */

(function() {
    'use strict';

    // ═══════════════════════════════════════════════════════════
    // CHART.JS CONFIGURATION
    // ═══════════════════════════════════════════════════════════
    
    // Global Chart.js defaults for SENTINEL theme
    Chart.defaults.color = '#7a9ab8';
    Chart.defaults.borderColor = '#1a2535';
    Chart.defaults.font.family = "'Inter', sans-serif";
    
    const COLORS = {
        operative: '#00d4ff',
        threat: '#ff4444',
        clear: '#00ff88',
        plate: '#ffcc00',
        warning: '#ff8800',
        surface: '#080d13',
        border: '#1a2535',
    };

    // ═══════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════
    
    const state = {
        autoRefresh: false,
        refreshInterval: null,
        dateRange: 'today',
        charts: {},
    };

    // ═══════════════════════════════════════════════════════════
    // DATA FETCHING
    // ═══════════════════════════════════════════════════════════
    
    async function fetchAnalytics() {
        try {
            const response = await fetch('/api/analytics');
            return await response.json();
        } catch (e) {
            console.error('[Analytics] Fetch error:', e);
            return null;
        }
    }
    
    async function fetchSessions() {
        try {
            const response = await fetch('/api/sessions');
            const data = await response.json();
            return data.sessions || [];
        } catch (e) {
            console.error('[Sessions] Fetch error:', e);
            return [];
        }
    }
    
    async function exportCSV() {
        try {
            const response = await fetch('/api/detections?limit=10000');
            const data = await response.json();
            
            if (!data.detections || data.detections.length === 0) {
                alert('No data to export');
                return;
            }
            
            // Convert to CSV
            const headers = Object.keys(data.detections[0]);
            const csv = [
                headers.join(','),
                ...data.detections.map(row => 
                    headers.map(h => JSON.stringify(row[h] ?? '')).join(',')
                )
            ].join('\n');
            
            // Download
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `sentinel-detections-${new Date().toISOString().split('T')[0]}.csv`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (e) {
            console.error('[Export] Error:', e);
            alert('Export failed');
        }
    }

    // ═══════════════════════════════════════════════════════════
    // CHARTS
    // ═══════════════════════════════════════════════════════════
    
    function createTimelineChart(canvas, data) {
        const timeline = data.timeline || [];
        
        // Reverse to show oldest first
        const reversed = [...timeline].reverse();
        
        return new Chart(canvas, {
            type: 'line',
            data: {
                labels: reversed.map(d => d.minute),
                datasets: [
                    {
                        label: 'Commercial',
                        data: reversed.map(d => d.commercial || 0),
                        borderColor: COLORS.clear,
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        fill: true,
                        tension: 0.3,
                    },
                    {
                        label: 'Military',
                        data: reversed.map(d => d.military || 0),
                        borderColor: COLORS.threat,
                        backgroundColor: 'rgba(255, 68, 68, 0.1)',
                        fill: true,
                        tension: 0.3,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index',
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 20,
                        },
                    },
                    tooltip: {
                        backgroundColor: COLORS.surface,
                        borderColor: COLORS.border,
                        borderWidth: 1,
                        padding: 12,
                    },
                },
                scales: {
                    x: {
                        grid: { color: COLORS.border },
                    },
                    y: {
                        beginAtZero: true,
                        grid: { color: COLORS.border },
                    },
                },
            },
        });
    }
    
    function createDistributionChart(canvas, data) {
        const dist = data.type_distribution || {};
        const commercial = dist['commercial-vehicle'] || 0;
        const military = dist['military_vehicle'] || 0;
        const total = commercial + military;
        
        return new Chart(canvas, {
            type: 'doughnut',
            data: {
                labels: ['Commercial', 'Military'],
                datasets: [{
                    data: [commercial, military],
                    backgroundColor: [COLORS.clear, COLORS.threat],
                    borderColor: COLORS.surface,
                    borderWidth: 3,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            padding: 20,
                        },
                    },
                    tooltip: {
                        backgroundColor: COLORS.surface,
                        borderColor: COLORS.border,
                        borderWidth: 1,
                    },
                },
            },
            plugins: [{
                id: 'centerText',
                beforeDraw: function(chart) {
                    const ctx = chart.ctx;
                    const centerX = (chart.chartArea.left + chart.chartArea.right) / 2;
                    const centerY = (chart.chartArea.top + chart.chartArea.bottom) / 2;
                    
                    ctx.save();
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    
                    // Total number
                    ctx.font = 'bold 32px Inter';
                    ctx.fillStyle = '#e2e8f0';
                    ctx.fillText(total, centerX, centerY - 8);
                    
                    // Label
                    ctx.font = '11px Inter';
                    ctx.fillStyle = '#7a9ab8';
                    ctx.fillText('TOTAL', centerX, centerY + 18);
                    
                    ctx.restore();
                },
            }],
        });
    }
    
    function createConfidenceChart(canvas, data) {
        const hist = data.confidence_histogram || {};
        const labels = Object.keys(hist).sort();
        const values = labels.map(k => hist[k]);
        
        // Create gradient colors from yellow to green
        const colors = labels.map((_, i) => {
            const ratio = i / (labels.length - 1 || 1);
            return `hsl(${60 + ratio * 60}, 100%, 50%)`;
        });
        
        return new Chart(canvas, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Detections',
                    data: values,
                    backgroundColor: colors,
                    borderRadius: 4,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                },
                scales: {
                    x: {
                        grid: { display: false },
                    },
                    y: {
                        beginAtZero: true,
                        grid: { color: COLORS.border },
                    },
                },
            },
        });
    }
    
    function createPerformanceChart(canvas, data) {
        const perf = data.performance || {};
        
        return new Chart(canvas, {
            type: 'bar',
            data: {
                labels: ['Min', 'Avg', 'Max'],
                datasets: [{
                    data: [perf.min_fps || 0, perf.avg_fps || 0, perf.max_fps || 0],
                    backgroundColor: [COLORS.threat, COLORS.operative, COLORS.clear],
                    borderRadius: 4,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: { display: false },
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        grid: { color: COLORS.border },
                    },
                    y: {
                        grid: { display: false },
                    },
                },
            },
        });
    }

    // ═══════════════════════════════════════════════════════════
    // HEATMAP
    // ═══════════════════════════════════════════════════════════
    
    function renderHeatmap(container, data) {
        const heatmap = data.hourly_heatmap || [];
        const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
        
        // Build matrix
        const matrix = {};
        let maxCount = 1;
        
        heatmap.forEach(item => {
            const key = `${item.day}-${item.hour}`;
            matrix[key] = item.count;
            if (item.count > maxCount) maxCount = item.count;
        });
        
        // Render
        let html = '<div class="heatmap-grid">';
        
        // Header row
        html += '<div class="heatmap-label"></div>';
        for (let h = 0; h < 24; h++) {
            html += `<div class="heatmap-label">${h}</div>`;
        }
        
        // Data rows
        days.forEach(day => {
            html += `<div class="heatmap-label">${day}</div>`;
            for (let h = 0; h < 24; h++) {
                const count = matrix[`${day}-${h}`] || 0;
                const level = Math.ceil((count / maxCount) * 5);
                html += `<div class="heatmap-cell level-${level}" title="${day} ${h}:00 - ${count} detections"></div>`;
            }
        });
        
        html += '</div>';
        container.innerHTML = html;
    }

    // ═══════════════════════════════════════════════════════════
    // TABLES
    // ═══════════════════════════════════════════════════════════
    
    function renderPlatesTable(container, data) {
        const plates = data.top_plates || [];
        
        if (plates.length === 0) {
            container.innerHTML = '<div class="empty-state"><div class="empty-state-text">No plates detected</div></div>';
            return;
        }
        
        let html = `
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Plate</th>
                        <th>Type</th>
                        <th>Count</th>
                        <th>Last Seen</th>
                        <th>Conf</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        plates.forEach(plate => {
            const typeClass = plate.vehicle_type === 'military_vehicle' ? 'military' : 'commercial';
            html += `
                <tr>
                    <td class="mono plate">${plate.plate_text || 'UNREAD'}</td>
                    <td class="${typeClass}">${formatType(plate.vehicle_type)}</td>
                    <td class="mono">${plate.count}</td>
                    <td>${formatTime(plate.last_seen)}</td>
                    <td class="mono">${(plate.avg_conf * 100).toFixed(0)}%</td>
                </tr>
            `;
        });
        
        html += '</tbody></table>';
        container.innerHTML = html;
    }
    
    function renderSessionsTable(container, sessions) {
        if (sessions.length === 0) {
            container.innerHTML = '<div class="empty-state"><div class="empty-state-text">No sessions recorded</div></div>';
            return;
        }
        
        let html = `
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Session</th>
                        <th>Start</th>
                        <th>Detections</th>
                        <th>Military</th>
                        <th>Plates</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        sessions.forEach(session => {
            const plateRate = session.total_detections > 0 
                ? ((session.plate_read_count / session.total_detections) * 100).toFixed(0) 
                : 0;
            
            html += `
                <tr>
                    <td class="mono">${session.session_id?.slice(0, 8) || '—'}...</td>
                    <td>${formatTime(session.start_time)}</td>
                    <td class="mono">${session.total_detections}</td>
                    <td class="mono military">${session.military_count}</td>
                    <td class="mono">${plateRate}%</td>
                </tr>
            `;
        });
        
        html += '</tbody></table>';
        container.innerHTML = html;
    }

    // ═══════════════════════════════════════════════════════════
    // UTILITIES
    // ═══════════════════════════════════════════════════════════
    
    function formatType(type) {
        if (!type) return 'Unknown';
        return type.replace(/[-_]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    }
    
    function formatTime(timestamp) {
        if (!timestamp) return '—';
        try {
            const date = new Date(timestamp);
            return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        } catch {
            return timestamp;
        }
    }

    // ═══════════════════════════════════════════════════════════
    // MAIN UPDATE
    // ═══════════════════════════════════════════════════════════
    
    async function updateDashboard() {
        const [analytics, sessions] = await Promise.all([
            fetchAnalytics(),
            fetchSessions(),
        ]);
        
        if (!analytics) return;
        
        // Update stat cards
        const plateRate = document.getElementById('plate-rate');
        if (plateRate) {
            plateRate.textContent = `${(analytics.plate_detection_rate * 100).toFixed(0)}%`;
        }
        
        // Update or create charts
        const timelineCanvas = document.getElementById('timeline-chart');
        if (timelineCanvas) {
            if (state.charts.timeline) state.charts.timeline.destroy();
            state.charts.timeline = createTimelineChart(timelineCanvas, analytics);
        }
        
        const distCanvas = document.getElementById('distribution-chart');
        if (distCanvas) {
            if (state.charts.distribution) state.charts.distribution.destroy();
            state.charts.distribution = createDistributionChart(distCanvas, analytics);
        }
        
        const confCanvas = document.getElementById('confidence-chart');
        if (confCanvas) {
            if (state.charts.confidence) state.charts.confidence.destroy();
            state.charts.confidence = createConfidenceChart(confCanvas, analytics);
        }
        
        const perfCanvas = document.getElementById('performance-chart');
        if (perfCanvas) {
            if (state.charts.performance) state.charts.performance.destroy();
            state.charts.performance = createPerformanceChart(perfCanvas, analytics);
        }
        
        // Update heatmap
        const heatmapContainer = document.getElementById('heatmap-container');
        if (heatmapContainer) {
            renderHeatmap(heatmapContainer, analytics);
        }
        
        // Update tables
        const platesContainer = document.getElementById('plates-table');
        if (platesContainer) {
            renderPlatesTable(platesContainer, analytics);
        }
        
        const sessionsContainer = document.getElementById('sessions-table');
        if (sessionsContainer) {
            renderSessionsTable(sessionsContainer, sessions);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // CONTROLS
    // ═══════════════════════════════════════════════════════════
    
    function setupControls() {
        // Date range selector
        const rangeSelect = document.getElementById('date-range');
        if (rangeSelect) {
            rangeSelect.addEventListener('change', (e) => {
                state.dateRange = e.target.value;
                updateDashboard();
            });
        }
        
        // Auto-refresh toggle
        const refreshToggle = document.getElementById('auto-refresh');
        if (refreshToggle) {
            refreshToggle.addEventListener('click', () => {
                state.autoRefresh = !state.autoRefresh;
                refreshToggle.classList.toggle('active', state.autoRefresh);
                
                if (state.autoRefresh) {
                    state.refreshInterval = setInterval(updateDashboard, 30000);
                } else {
                    clearInterval(state.refreshInterval);
                }
            });
        }
        
        // Export button
        const exportBtn = document.getElementById('export-btn');
        if (exportBtn) {
            exportBtn.addEventListener('click', exportCSV);
        }
        
        // Manual refresh
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', updateDashboard);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═══════════════════════════════════════════════════════════
    
    function init() {
        console.log('[SENTINEL] Analytics initializing...');
        
        setupControls();
        updateDashboard();
        
        console.log('[SENTINEL] Analytics ready');
    }
    
    // Start when DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
