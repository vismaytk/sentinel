/**
 * SENTINEL Config JavaScript
 * Configuration panel logic
 */

(function() {
    'use strict';

    // ═══════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════
    
    let config = {};
    let dirty = false;
    let resizeRaf = null;

    // ═══════════════════════════════════════════════════════════
    // API
    // ═══════════════════════════════════════════════════════════
    
    async function loadConfig() {
        try {
            const response = await fetch('/config');
            config = await response.json();
            updateUI();
            return config;
        } catch (e) {
            console.error('[Config] Load error:', e);
            return null;
        }
    }
    
    async function saveConfig() {
        try {
            const response = await fetch('/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            });
            
            const result = await response.json();
            
            if (result.status === 'ok') {
                dirty = false;
                showNotification('Configuration saved', 'success');
            } else {
                showNotification('Error: ' + (result.errors?.join(', ') || 'Unknown'), 'error');
            }
            
            return result;
        } catch (e) {
            console.error('[Config] Save error:', e);
            showNotification('Failed to save configuration', 'error');
            return null;
        }
    }

    // ═══════════════════════════════════════════════════════════
    // UI UPDATE
    // ═══════════════════════════════════════════════════════════
    
    function updateUI() {
        // Vehicle confidence sliders
        if (config.vehicle_conf) {
            Object.entries(config.vehicle_conf).forEach(([key, value]) => {
                const slider = document.querySelector(`input[data-config="vehicle_conf.${key}"]`);
                if (slider) {
                    slider.value = value;
                    updateSliderDisplay(slider);
                }
            });
        }
        
        // Plate confidence sliders
        if (config.plate_conf) {
            Object.entries(config.plate_conf).forEach(([key, value]) => {
                const slider = document.querySelector(`input[data-config="plate_conf.${key}"]`);
                if (slider) {
                    slider.value = value;
                    updateSliderDisplay(slider);
                }
            });
        }
        
        // Weapon confidence sliders
        if (config.weapon_conf) {
            Object.entries(config.weapon_conf).forEach(([key, value]) => {
                const slider = document.querySelector(`input[data-config="weapon_conf.${key}"]`);
                if (slider) {
                    slider.value = value;
                    updateSliderDisplay(slider);
                }
            });
        }
        
        // YOLO image size
        const imgsizeSelect = document.querySelector('[data-config="yolo_imgsz"]');
        if (imgsizeSelect && config.yolo_imgsz) {
            imgsizeSelect.value = config.yolo_imgsz;
        }
        
        // Detect every N
        const detectNInput = document.querySelector('[data-config="detect_every_n"]');
        if (detectNInput && config.detect_every_n) {
            detectNInput.value = config.detect_every_n;
        }
        
        // Toggle switches
        const ocrToggle = document.querySelector('[data-config="enable_ocr"]');
        if (ocrToggle) {
            ocrToggle.classList.toggle('active', config.enable_ocr);
        }
        
        const trackingToggle = document.querySelector('[data-config="enable_tracking"]');
        if (trackingToggle) {
            trackingToggle.classList.toggle('active', config.enable_tracking);
        }
        
        // Weapon detection toggles
        const gunToggle = document.querySelector('[data-config="enable_gun_detection"]');
            if (gunToggle) {
                gunToggle.classList.toggle('active', config.enable_gun_detection);
                const gunSlider = document.getElementById('gun-conf-slider');
                if (gunSlider) gunSlider.classList.toggle('is-hidden', !config.enable_gun_detection);
            }
        
        const grenadeToggle = document.querySelector('[data-config="enable_grenade_detection"]');
            if (grenadeToggle) {
                grenadeToggle.classList.toggle('active', config.enable_grenade_detection);
                const grenadeSlider = document.getElementById('grenade-conf-slider');
                if (grenadeSlider) grenadeSlider.classList.toggle('is-hidden', !config.enable_grenade_detection);
            }
    }
    
    function updateSliderDisplay(slider) {
        const display = slider.parentElement.querySelector('.slider-value');
        if (display) {
            display.textContent = parseFloat(slider.value).toFixed(2);
        }
    }

    // ═══════════════════════════════════════════════════════════
    // EVENT HANDLERS
    // ═══════════════════════════════════════════════════════════
    
    function setupEventHandlers() {
        // Slider inputs
        document.querySelectorAll('input[type="range"][data-config]').forEach(slider => {
            slider.addEventListener('input', (e) => {
                updateSliderDisplay(e.target);
                dirty = true;
            });
            
            slider.addEventListener('change', (e) => {
                const [category, key] = e.target.dataset.config.split('.');
                if (config[category]) {
                    config[category][key] = parseFloat(e.target.value);
                }
                saveConfig();
            });
        });
        
        // Select inputs
        document.querySelectorAll('select[data-config]').forEach(select => {
            select.addEventListener('change', (e) => {
                const key = e.target.dataset.config;
                config[key] = parseInt(e.target.value);
                saveConfig();
            });
        });
        
        // Number inputs
        document.querySelectorAll('input[type="number"][data-config]').forEach(input => {
            input.addEventListener('change', (e) => {
                const key = e.target.dataset.config;
                config[key] = parseInt(e.target.value);
                saveConfig();
            });
        });
        
        // Toggle switches
        document.querySelectorAll('.toggle-switch[data-config]').forEach(toggle => {
            toggle.addEventListener('click', (e) => {
                const key = e.target.dataset.config;
                config[key] = !config[key];
                e.target.classList.toggle('active', config[key]);
                
                // Show/hide weapon confidence sliders when weapon toggles change
                if (key === 'enable_gun_detection') {
                    const slider = document.getElementById('gun-conf-slider');
                    if (slider) slider.classList.toggle('is-hidden', !config[key]);
                }
                if (key === 'enable_grenade_detection') {
                    const slider = document.getElementById('grenade-conf-slider');
                    if (slider) slider.classList.toggle('is-hidden', !config[key]);
                }
                
                saveConfig();
            });
        });
        
        // Collapsible config panel
        const configHeader = document.querySelector('.config-header');
        const configBody = document.querySelector('.config-body');
        const configToggle = document.querySelector('.config-toggle');
        
        if (configHeader && configBody) {
            let isConfigOpen = true;

            const applyConfigPanelState = () => {
                configBody.classList.toggle('open', isConfigOpen);
                configHeader.setAttribute('aria-expanded', isConfigOpen ? 'true' : 'false');
                if (configToggle) configToggle.classList.toggle('open', isConfigOpen);
            };

            const reconcileHeight = () => {
                // Keep panel content reachable after viewport changes.
                if (window.matchMedia('(max-width: 768px)').matches) {
                    configBody.style.maxHeight = '';
                    return;
                }
                const viewportAllowance = Math.max(260, window.innerHeight - 220);
                if (isConfigOpen) {
                    configBody.style.maxHeight = `${Math.min(viewportAllowance, 760)}px`;
                } else {
                    configBody.style.maxHeight = '0px';
                }
            };

            const updateConfigPanelState = (forceOpen = null) => {
                if (forceOpen !== null) {
                    isConfigOpen = forceOpen;
                }
                applyConfigPanelState();
                reconcileHeight();
            };

            updateConfigPanelState(true);

            const toggleConfigPanel = () => {
                updateConfigPanelState(!isConfigOpen);
            };

            configHeader.addEventListener('click', toggleConfigPanel);
            configHeader.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    toggleConfigPanel();
                }
            });

            window.addEventListener('resize', () => {
                if (resizeRaf) cancelAnimationFrame(resizeRaf);
                resizeRaf = requestAnimationFrame(() => {
                    applyConfigPanelState();
                    reconcileHeight();
                });
            });
        }
    }

    // ═══════════════════════════════════════════════════════════
    // NOTIFICATIONS
    // ═══════════════════════════════════════════════════════════
    
    function showNotification(message, type = 'info') {
        // Remove existing notification
        const existing = document.querySelector('.config-notification');
        if (existing) existing.remove();
        
        const notification = document.createElement('div');
        notification.className = `config-notification ${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 12px 20px;
            background: ${type === 'success' ? 'var(--glow-clear)' : 'var(--glow-threat)'};
            border: 1px solid ${type === 'success' ? 'var(--accent-clear)' : 'var(--accent-threat)'};
            color: ${type === 'success' ? 'var(--accent-clear)' : 'var(--accent-threat)'};
            border-radius: var(--radius-md);
            font-size: 12px;
            z-index: 1000;
            animation: slideIn 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    // ═══════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═══════════════════════════════════════════════════════════
    
    function init() {
        console.log('[Config] Initializing...');
        
        setupEventHandlers();
        loadConfig();
        
        console.log('[Config] Ready');
    }
    
    // Export for global access
    window.SENTINELConfig = {
        load: loadConfig,
        save: saveConfig,
        get: () => config,
    };
    
    // Start when DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
