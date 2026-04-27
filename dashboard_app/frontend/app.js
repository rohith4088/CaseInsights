/* ===========================
   NeuroSight — app.js
   =========================== */
document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const uploadZone = document.getElementById('upload-zone');
    const loadingOverlay = document.getElementById('loading-overlay');

    let globalData = [];
    let activeFilter = null;
    let currentCaseNumber = null;
    let validCategories = [];

    // --- Drag & Drop ---
    uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
    uploadZone.addEventListener('drop', e => {
        e.preventDefault(); uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFileUpload(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', e => { if (e.target.files.length) handleFileUpload(e.target.files[0]); });

    // --- File Upload ---
    function handleFileUpload(file) {
        if (!file.name.endsWith('.xlsx') && !file.name.endsWith('.csv')) {
            alert('Please upload a valid .xlsx or .csv file'); return;
        }
        const formData = new FormData();
        formData.append('file', file);
        loadingOverlay.style.display = 'flex';

        fetch('http://localhost:8000/api/upload', { method: 'POST', body: formData })
            .then(res => { if (!res.ok) throw new Error('Upload failed'); return res.json(); })
            .then(data => {
                loadingOverlay.style.display = 'none';
                collapseUpload(file.name);

                validCategories = data.valid_categories || [];
                globalData = data.table || [];

                // Populate override dropdown
                const sel = document.getElementById('override-select');
                sel.innerHTML = validCategories.map(c => `<option value="${c}">${c}</option>`).join('');

                // Show all sections
                ['stats-grid','charts-section','analytics-section',
                 'leaderboard-section','benchmark-section','table-section'].forEach(id => {
                    document.getElementById(id).style.display = id === 'stats-grid' ? 'grid' :
                        (id === 'charts-section' || id === 'analytics-section') ? 'grid' : 'block';
                });

                updateStats(data.summary, data.benchmark);
                renderCharts(data.summary);
                renderAnalytics(data.analytics);
                renderAgents(data.analytics?.agents || []);
                if (data.benchmark) renderBenchmark(data.benchmark);
                renderTable(globalData, null);
            })
            .catch(err => {
                loadingOverlay.style.display = 'none';
                alert('Error: ' + err.message + '\nMake sure the backend is running on port 8000.');
            });
    }

    function collapseUpload(name) {
        uploadZone.style.padding = '20px 40px';
        uploadZone.querySelector('h3').innerText = '✓ Loaded: ' + name;
        uploadZone.querySelector('p').style.display = 'none';
        uploadZone.querySelector('.btn-primary').style.display = 'none';
        uploadZone.querySelector('.upload-icon').style.display = 'none';
        // Show BERT panel and fetch initial status
        document.getElementById('bert-panel').style.display = 'flex';
        fetchBERTStatus();
    }

    // =============================================
    // BERT Engine
    // =============================================
    let bertPollInterval = null;

    function fetchBERTStatus() {
        fetch('http://localhost:8000/api/bert/status')
            .then(r => r.json())
            .then(s => updateBERTUI(s))
            .catch(() => {});
    }

    function updateBERTUI(s) {
        const panel = document.getElementById('bert-panel');
        const statusBadge = document.getElementById('bert-status-badge');
        const deviceBadge = document.getElementById('bert-device-badge');
        const progressWrap = document.getElementById('bert-progress-wrap');
        const progressBar = document.getElementById('bert-progress-bar');
        const progressText = document.getElementById('bert-progress-text');
        const progressPct = document.getElementById('bert-progress-pct');
        const trainBtn = document.getElementById('bert-train-btn');
        const applyBtn = document.getElementById('bert-apply-btn');
        const lossBadge = document.getElementById('bert-loss-badge');

        // Device badge
        deviceBadge.innerText = s.device || 'CPU';
        if (s.device && s.device.includes('MPS')) deviceBadge.style.cssText = 'background:rgba(16,185,129,.15);color:#10b981;border:1px solid rgba(16,185,129,.3);';
        else if (s.device && s.device.includes('CUDA')) deviceBadge.style.cssText = 'background:rgba(99,102,241,.15);color:#a5b4fc;border:1px solid rgba(99,102,241,.3);';

        if (s.status === 'idle' || s.status === 'unavailable') {
            panel.classList.remove('training');
            statusBadge.innerText = 'Not Trained';
            statusBadge.style.cssText = 'background:rgba(255,255,255,.08);color:#94a3b8;';
            progressWrap.style.display = 'none';
            trainBtn.style.display = 'inline-block';
            applyBtn.style.display = 'none';
        } else if (s.status === 'training') {
            panel.classList.add('training');
            statusBadge.innerText = '⚡ Training';
            statusBadge.style.cssText = 'background:rgba(168,85,247,.15);color:#a855f7;border:1px solid rgba(168,85,247,.3);';
            progressWrap.style.display = 'block';
            progressBar.style.width = (s.progress || 0) + '%';
            progressText.innerText = `Training... Epoch ${s.current_epoch || 1}/${s.total_epochs || 3}`;
            progressPct.innerText = (s.progress || 0) + '%';
            trainBtn.style.display = 'none';
        } else if (s.status === 'ready') {
            panel.classList.remove('training');
            statusBadge.innerText = '✓ Trained';
            statusBadge.style.cssText = 'background:rgba(16,185,129,.15);color:#10b981;border:1px solid rgba(16,185,129,.3);';
            progressBar.style.width = '100%';
            progressText.innerText = 'Training complete!';
            progressPct.innerText = '100%';
            trainBtn.style.display = 'inline-block';
            trainBtn.innerText = '↺ Retrain BERT';
            applyBtn.style.display = 'inline-block';
            if (s.train_loss) {
                lossBadge.style.display = 'inline-block';
                lossBadge.innerText = `Loss: ${s.train_loss}`;
            }
            stopBERTPoll();
        } else if (s.status === 'error') {
            panel.classList.remove('training');
            statusBadge.innerText = '✗ Error';
            statusBadge.style.cssText = 'background:rgba(239,68,68,.15);color:#ef4444;border:1px solid rgba(239,68,68,.3);';
            progressText.innerText = s.error || 'Training failed';
            trainBtn.style.display = 'inline-block';
            stopBERTPoll();
        }
    }

    function startBERTPoll() {
        if (bertPollInterval) return;
        bertPollInterval = setInterval(() => {
            fetch('http://localhost:8000/api/bert/status')
                .then(r => r.json())
                .then(s => {
                    updateBERTUI(s);
                    if (s.status === 'ready' || s.status === 'error') stopBERTPoll();
                })
                .catch(() => {});
        }, 3000);
    }

    function stopBERTPoll() {
        if (bertPollInterval) { clearInterval(bertPollInterval); bertPollInterval = null; }
    }

    window.startBERTTraining = function () {
        const btn = document.getElementById('bert-train-btn');
        btn.disabled = true;
        btn.innerText = 'Starting...';
        document.getElementById('bert-progress-wrap').style.display = 'block';
        document.getElementById('bert-progress-bar').style.width = '2%';
        fetch('http://localhost:8000/api/bert/train', { method: 'POST' })
            .then(r => r.json())
            .then(resp => {
                if (resp.status === 'already_training') {
                    btn.disabled = false; btn.innerText = '⚡ Train BERT on this data';
                    startBERTPoll();
                    return;
                }
                startBERTPoll();
                updateBERTUI({ status: 'training', progress: 2, current_epoch: 1, total_epochs: 3, device: resp.device });
            })
            .catch(err => {
                btn.disabled = false; btn.innerText = '⚡ Train BERT on this data';
                alert('Could not start BERT training: ' + err.message);
            });
    };

    window.applyBERTPredictions = function () {
        const btn = document.getElementById('bert-apply-btn');
        btn.disabled = true;
        btn.innerText = '⏳ Applying BERT predictions...';
        document.getElementById('loading-overlay').style.display = 'flex';
        document.getElementById('loading-msg').innerText = 'Running DistilBERT inference...';

        // ── Snapshot current (LinearSVC) predictions BEFORE overwriting ──
        const svcSnapshot = {};
        globalData.forEach(r => {
            const key = String(r['Case Number'] || '');
            if (key) svcSnapshot[key] = {
                category: r.Predicted_Category || '',
                confidence: parseFloat(r.Prediction_Confidence) || 0,
                subject: r.Subject || ''
            };
        });

        fetch('http://localhost:8000/api/bert/predict', { method: 'POST' })
            .then(r => { if (!r.ok) throw new Error('Prediction failed'); return r.json(); })
            .then(data => {
                document.getElementById('loading-overlay').style.display = 'none';
                btn.disabled = false;
                btn.innerText = '✓ BERT Active';
                globalData = data.table || [];
                renderCharts(data.summary);
                renderTable(globalData, null);
                if (data.summary) {
                    document.getElementById('stat-total').innerText = data.summary.total_cases.toLocaleString();
                    document.getElementById('stat-high').innerText = data.summary.high_confidence.toLocaleString();
                    document.getElementById('stat-low').innerText = data.summary.low_confidence.toLocaleString();
                    document.getElementById('stat-accuracy').innerText = '🧠 BERT';
                }
                const sb = document.getElementById('bert-status-badge');
                sb.innerText = '🧠 Active Engine';
                sb.style.cssText = 'background:linear-gradient(135deg,rgba(99,102,241,.2),rgba(168,85,247,.2));color:#a5b4fc;border:1px solid rgba(99,102,241,.3);font-weight:600;';

                // ── Compute and show diff ──
                renderDiff(svcSnapshot, globalData);
            })
            .catch(err => {
                document.getElementById('loading-overlay').style.display = 'none';
                btn.disabled = false; btn.innerText = '✓ Apply BERT Predictions';
                alert('BERT prediction error: ' + err.message);
            });
    };

    // =============================================
    // BERT Diff Engine
    // =============================================
    let diffData = [];       // all changed rows
    let diffFiltered = [];   // after search filter
    let diffPage = 0;
    const DIFF_PAGE_SIZE = 30;

    function renderDiff(svcSnapshot, bertRows) {
        const changed = [];
        bertRows.forEach(row => {
            const key = String(row['Case Number'] || '');
            const old = svcSnapshot[key];
            if (!old) return;
            const bertCat = row.Predicted_Category || '';
            const bertConf = parseFloat(row.Prediction_Confidence) || 0;
            if (old.category !== bertCat) {
                changed.push({
                    case_number: key,
                    subject: row.Subject || old.subject || '',
                    svc_cat: old.category,
                    bert_cat: bertCat,
                    svc_conf: old.confidence,
                    bert_conf: bertConf,
                    delta_conf: bertConf - old.confidence
                });
            }
        });

        diffData = changed;
        diffFiltered = [...changed];
        diffPage = 0;

        const panel = document.getElementById('bert-diff-panel');
        panel.style.display = 'block';
        panel.scrollIntoView({ behavior: 'smooth' });

        // ── Stat cards ──
        const totalChanged = changed.length;
        const totalCases = bertRows.length;
        const pct = totalCases > 0 ? ((totalChanged / totalCases) * 100).toFixed(1) : 0;
        const confUp = changed.filter(c => c.delta_conf > 0.05).length;
        const confDown = changed.filter(c => c.delta_conf < -0.05).length;
        document.getElementById('diff-stats').innerHTML = [
            { label: 'Cases Changed', value: totalChanged.toLocaleString(), color: '' },
            { label: '% of Total', value: pct + '%', color: '' },
            { label: 'Conf. Improved ↑', value: confUp.toLocaleString(), color: 'text-green' },
            { label: 'Conf. Dropped ↓', value: confDown.toLocaleString(), color: 'text-red' },
        ].map(s => `<div class="diff-stat-card"><div class="label">${s.label}</div>
            <div class="value ${s.color}">${s.value}</div></div>`).join('');

        // ── Category flow cards (top 10 transitions) ──
        const flowMap = {};
        changed.forEach(c => {
            const key = `${c.svc_cat}|||${c.bert_cat}`;
            flowMap[key] = (flowMap[key] || 0) + 1;
        });
        const topFlows = Object.entries(flowMap)
            .sort((a, b) => b[1] - a[1]).slice(0, 10);
        document.getElementById('diff-flow-grid').innerHTML = topFlows.map(([k, count]) => {
            const [from, to] = k.split('|||');
            return `<div class="diff-flow-card">
                <div class="flow-cats">
                    <span class="flow-from" title="${from}">${from.slice(0, 28)}</span>
                    <span class="flow-arrow">→</span>
                    <span class="flow-to" title="${to}">${to.slice(0, 28)}</span>
                </div>
                <div class="flow-count">${count} case${count !== 1 ? 's' : ''} reassigned</div>
            </div>`;
        }).join('');

        renderDiffTable();

        // Search
        document.getElementById('diff-search').addEventListener('input', e => {
            const q = e.target.value.toLowerCase();
            diffFiltered = q ? diffData.filter(d =>
                d.case_number.toLowerCase().includes(q) ||
                d.subject.toLowerCase().includes(q) ||
                d.svc_cat.toLowerCase().includes(q) ||
                d.bert_cat.toLowerCase().includes(q)
            ) : [...diffData];
            diffPage = 0;
            renderDiffTable();
        });
    }

    function renderDiffTable() {
        const tbody = document.getElementById('diff-tbody');
        tbody.innerHTML = '';
        document.getElementById('diff-count').innerText = `(${diffFiltered.length})`;
        const start = diffPage * DIFF_PAGE_SIZE;
        const page = diffFiltered.slice(start, start + DIFF_PAGE_SIZE);

        if (page.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--text-secondary)">No changed cases match the filter.</td></tr>';
        } else {
            page.forEach(d => {
                const deltaAmt = (d.delta_conf * 100).toFixed(1);
                const deltaClass = d.delta_conf > 0.05 ? 'delta-up' : d.delta_conf < -0.05 ? 'delta-down' : 'delta-same';
                const deltaLabel = d.delta_conf > 0 ? `+${deltaAmt}%` : `${deltaAmt}%`;
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="white-space:nowrap;">${d.case_number}</td>
                    <td class="subject-cell" title="${d.subject}">${d.subject || '—'}</td>
                    <td style="color:var(--danger);font-size:12px;">${d.svc_cat}<br><span style="color:var(--text-secondary);font-size:11px;">${Math.round(d.svc_conf * 100)}% conf</span></td>
                    <td style="color:var(--success);font-size:12px;"><strong>${d.bert_cat}</strong><br><span style="color:var(--text-secondary);font-size:11px;">${Math.round(d.bert_conf * 100)}% conf</span></td>
                    <td class="${deltaClass}">${deltaLabel}</td>
                    <td><button class="btn-secondary" style="padding:4px 10px;font-size:12px;" onclick="openModal('${d.case_number}')">View</button></td>`;
                tbody.appendChild(tr);
            });
        }

        // Pagination
        const pag = document.getElementById('diff-pagination');
        const totalPages = Math.ceil(diffFiltered.length / DIFF_PAGE_SIZE);
        pag.innerHTML = '';
        if (totalPages > 1) {
            for (let p = 0; p < totalPages; p++) {
                const btn = document.createElement('button');
                btn.className = 'page-btn' + (p === diffPage ? ' active' : '');
                btn.innerText = p + 1;
                btn.onclick = () => { diffPage = p; renderDiffTable(); };
                pag.appendChild(btn);
            }
        }
    }

    window.exportDiff = function () {
        const cols = ['case_number', 'subject', 'svc_cat', 'bert_cat', 'svc_conf', 'bert_conf', 'delta_conf'];
        const header = ['Case Number', 'Subject', 'LinearSVC Category', 'BERT Category', 'SVC Confidence', 'BERT Confidence', 'Delta'];
        const rows = [header.join(','), ...diffFiltered.map(d =>
            cols.map(c => `"${String(d[c] || '').replace(/"/g, '""')}"`).join(',')
        )];
        const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'BERT_vs_LinearSVC_Changes.csv';
        a.click();
    };


    // --- Stats ---
    function updateStats(summary, benchmark) {
        document.getElementById('stat-total').innerText = summary.total_cases.toLocaleString();
        document.getElementById('stat-high').innerText = summary.high_confidence.toLocaleString();
        document.getElementById('stat-low').innerText = summary.low_confidence.toLocaleString();
        if (benchmark) {
            document.getElementById('stat-accuracy').innerText = benchmark.accuracy_pct + '%';
            document.getElementById('bench-accuracy').innerText = benchmark.accuracy_pct + '%';
            const sideAcc = document.getElementById('sidebar-accuracy');
            sideAcc.style.display = 'block';
            document.getElementById('sidebar-acc-val').innerText = benchmark.accuracy_pct + '%';
        }
        // Render MoM section
        if (summary.mom_changes && Object.keys(summary.mom_changes).length > 0) {
            renderMoM(summary.mom_changes);
        }
    }

    // --- Charts (Pie + Bar) ---
    let pieChart, barChart, trendChart, severityChart;
    const COLORS = ['#6366f1','#a855f7','#ec4899','#10b981','#f59e0b','#3b82f6','#14b8a6','#f43f5e','#8b5cf6'];

    function renderCharts(summary) {
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'Outfit', sans-serif";

        if (pieChart) pieChart.destroy();
        pieChart = new Chart(document.getElementById('pieChart').getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: summary.main_categories.labels,
                datasets: [{ data: summary.main_categories.values, backgroundColor: COLORS, borderWidth: 0, hoverOffset: 8 }]
            },
            options: {
                responsive: true, maintainAspectRatio: false, cutout: '62%',
                plugins: { legend: { position: 'right' } },
                onClick: (evt, els) => els.length ? applyFilter('Main_Category', summary.main_categories.labels[els[0].index]) : clearFilter(),
                onHover: (evt, els) => { evt.native.target.style.cursor = els.length ? 'pointer' : 'default'; }
            }
        });

        if (barChart) barChart.destroy();
        barChart = new Chart(document.getElementById('barChart').getContext('2d'), {
            type: 'bar',
            data: {
                labels: summary.top_categories.labels,
                datasets: [{ label: 'Cases', data: summary.top_categories.values, backgroundColor: 'rgba(99,102,241,.8)', borderRadius: 4 }]
            },
            options: {
                responsive: true, maintainAspectRatio: false, indexAxis: 'y',
                plugins: { legend: { display: false } },
                scales: { x: { grid: { color: 'rgba(255,255,255,.05)' } }, y: { grid: { display: false } } },
                onClick: (evt, els) => els.length ? applyFilter('Predicted_Category', summary.top_categories.labels[els[0].index]) : clearFilter(),
                onHover: (evt, els) => { evt.native.target.style.cursor = els.length ? 'pointer' : 'default'; }
            }
        });
    }

    // --- Analytics Charts (Trends + Severity) ---
    let resolutionChart = null;

    function renderAnalytics(analytics) {
        if (!analytics) return;

        // Trend Chart
        if (analytics.trends && analytics.trends.labels && analytics.trends.labels.length > 0) {
            if (trendChart) trendChart.destroy();
            trendChart = new Chart(document.getElementById('trendChart').getContext('2d'), {
                type: 'line',
                data: analytics.trends,
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { position: 'top' } },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,.05)' }, ticks: { maxTicksLimit: 12 } },
                        y: { grid: { color: 'rgba(255,255,255,.05)' }, beginAtZero: true }
                    }
                }
            });
        }

        // Severity Chart
        if (analytics.severity && analytics.severity.labels && analytics.severity.labels.length > 0) {
            if (severityChart) severityChart.destroy();
            severityChart = new Chart(document.getElementById('severityChart').getContext('2d'), {
                type: 'bar',
                data: analytics.severity,
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { position: 'top' } },
                    scales: {
                        x: { stacked: true, grid: { display: false }, ticks: { maxRotation: 30, font: { size: 10 } } },
                        y: { stacked: true, grid: { color: 'rgba(255,255,255,.05)' } }
                    }
                }
            });
        }

        // Resolution Time Chart
        if (analytics.resolution_time && analytics.resolution_time.length > 0) {
            document.getElementById('resolution-section').style.display = 'block';
            const labels = analytics.resolution_time.map(r => r.category);
            const avgs = analytics.resolution_time.map(r => r.avg_days);
            const medians = analytics.resolution_time.map(r => r.median_days);
            if (resolutionChart) resolutionChart.destroy();
            resolutionChart = new Chart(document.getElementById('resolutionChart').getContext('2d'), {
                type: 'bar',
                data: {
                    labels,
                    datasets: [
                        { label: 'Avg Days', data: avgs, backgroundColor: 'rgba(99,102,241,.8)', borderRadius: 4 },
                        { label: 'Median Days', data: medians, backgroundColor: 'rgba(168,85,247,.6)', borderRadius: 4 }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false, indexAxis: 'y',
                    plugins: { legend: { position: 'top' } },
                    scales: { x: { grid: { color: 'rgba(255,255,255,.05)' } }, y: { grid: { display: false }, ticks: { font: { size: 11 } } } }
                }
            });
        }
    }

    // --- Agent Leaderboard ---
    function renderAgents(agents) {
        const tbody = document.getElementById('agent-tbody');
        tbody.innerHTML = '';
        agents.forEach((a, i) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${i + 1}</td><td>${a.name}</td><td><strong>${a.count.toLocaleString()}</strong></td><td><span class="badge badge-medium">${a.top_category}</span></td>`;
            tbody.appendChild(tr);
        });
    }

    // --- MoM Trends ---
    function renderMoM(momChanges) {
        const grid = document.getElementById('mom-grid');
        const momSec = document.getElementById('mom-section');
        if (!momChanges || Object.keys(momChanges).length === 0) return;
        momSec.style.display = 'block';
        grid.innerHTML = '';
        Object.entries(momChanges).sort((a,b) => Math.abs(b[1].change_pct) - Math.abs(a[1].change_pct)).slice(0,10).forEach(([cat, d]) => {
            const up = d.change_pct >= 0;
            const div = document.createElement('div');
            div.className = 'mom-card';
            div.innerHTML = `
                <div class="mom-cat" title="${cat}">${cat}</div>
                <div class="mom-change ${up ? 'text-red' : 'text-green'}">${up ? '+' : ''}${d.change_pct}%</div>
                <div class="mom-detail">${d.prev} → ${d.last} cases</div>`;
            grid.appendChild(div);
        });
    }

    // --- Benchmark ---
    let allMismatches = [];
    let filteredMismatches = [];
    let mismatchSortCol = 'confidence_pct';
    let mismatchSortAsc = true;
    let mismatchPage = 0;
    const MISMATCH_PAGE_SIZE = 25;

    function renderBenchmark(bm) {
        document.getElementById('bench-correct').innerText = bm.correct.toLocaleString();
        document.getElementById('bench-wrong').innerText = bm.incorrect.toLocaleString();
        document.getElementById('bench-total').innerText = bm.evaluated.toLocaleString();

        allMismatches = bm.top_mismatches || [];
        filteredMismatches = [...allMismatches];

        renderMismatchTable();
        setupMismatchInteractivity();
    }

    function applyMismatchFilters() {
        const q = (document.getElementById('mismatch-search').value || '').toLowerCase();
        const confRange = document.getElementById('mismatch-conf-filter').value;
        filteredMismatches = allMismatches.filter(m => {
            const textOk = !q || m.subject.toLowerCase().includes(q) || m.actual.toLowerCase().includes(q) || m.predicted.toLowerCase().includes(q) || m.case_number.toLowerCase().includes(q);
            let confOk = true;
            if (confRange !== 'all') {
                const [lo, hi] = confRange.split('-').map(Number);
                confOk = m.confidence_pct >= lo && m.confidence_pct < hi;
            }
            return textOk && confOk;
        });
        // Apply current sort
        filteredMismatches.sort((a, b) => {
            const av = a[mismatchSortCol]; const bv = b[mismatchSortCol];
            if (typeof av === 'number') return mismatchSortAsc ? av - bv : bv - av;
            return mismatchSortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
        });
        mismatchPage = 0;
        renderMismatchTable();
    }

    function renderMismatchTable() {
        const tbody = document.getElementById('mismatch-tbody');
        tbody.innerHTML = '';
        const start = mismatchPage * MISMATCH_PAGE_SIZE;
        const page = filteredMismatches.slice(start, start + MISMATCH_PAGE_SIZE);

        document.getElementById('mismatch-count').innerText = `(${filteredMismatches.length} cases)`;

        if (page.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:24px;color:var(--text-secondary)">No mismatches match the current filter.</td></tr>';
        } else {
            page.forEach(m => {
                const tr = document.createElement('tr');
                let bc = 'badge-low';
                if (m.confidence_pct >= 75) bc = 'badge-medium';
                else if (m.confidence_pct >= 50) bc = 'badge-medium';
                tr.innerHTML = `
                    <td style="white-space:nowrap;">${m.case_number || '—'}</td>
                    <td class="subject-cell" title="${m.subject}">${m.subject || '—'}</td>
                    <td>${m.severity || '—'}</td>
                    <td style="color:var(--success);font-size:12px;">${m.actual}</td>
                    <td style="color:var(--danger);font-size:12px;"><strong>${m.predicted}</strong></td>
                    <td><span class="badge badge-low">${m.confidence_pct}%</span></td>
                    <td><button class="btn-secondary" style="padding:4px 10px;font-size:12px;" onclick="openModal('${m.case_number}')">View & Fix</button></td>`;
                tbody.appendChild(tr);
            });
        }

        // Pagination
        const pagDiv = document.getElementById('mismatch-pagination');
        const totalPages = Math.ceil(filteredMismatches.length / MISMATCH_PAGE_SIZE);
        pagDiv.innerHTML = '';
        if (totalPages > 1) {
            for (let p = 0; p < totalPages; p++) {
                const btn = document.createElement('button');
                btn.className = 'page-btn' + (p === mismatchPage ? ' active' : '');
                btn.innerText = p + 1;
                btn.onclick = () => { mismatchPage = p; renderMismatchTable(); };
                pagDiv.appendChild(btn);
            }
        }
    }

    function setupMismatchInteractivity() {
        // Search
        document.getElementById('mismatch-search').addEventListener('input', applyMismatchFilters);
        // Confidence filter
        document.getElementById('mismatch-conf-filter').addEventListener('change', applyMismatchFilters);
        // Column sorting
        document.querySelectorAll('#mismatch-table th.sortable').forEach(th => {
            th.addEventListener('click', () => {
                const col = th.dataset.col;
                if (mismatchSortCol === col) {
                    mismatchSortAsc = !mismatchSortAsc;
                } else {
                    mismatchSortCol = col;
                    mismatchSortAsc = col === 'confidence_pct'; // default asc for confidence
                }
                document.querySelectorAll('#mismatch-table th.sortable').forEach(t => t.classList.remove('active-sort', 'sort-asc', 'sort-desc'));
                th.classList.add('active-sort', mismatchSortAsc ? 'sort-asc' : 'sort-desc');
                applyMismatchFilters();
            });
        });
    }

    // Export mismatches
    window.exportMismatches = function () {
        const cols = ['case_number', 'subject', 'severity', 'actual', 'predicted', 'confidence_pct'];
        const rows = [cols.join(','), ...filteredMismatches.map(r => cols.map(c => `"${String(r[c] || '').replace(/"/g, '""')}"`).join(','))];
        const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'CaseInsights_Mismatches.csv';
        a.click();
    };

    // --- Table ---
    function renderTable(data, filterLabel) {
        const tbody = document.getElementById('table-body');
        tbody.innerHTML = '';
        const titleEl = document.getElementById('table-title');
        const clearBtn = document.getElementById('clear-filter-btn');
        if (filterLabel) {
            titleEl.innerHTML = `Cases &nbsp;<span class="badge badge-medium">${filterLabel}</span>`;
            clearBtn.style.display = 'inline-block';
        } else {
            titleEl.innerText = 'Categorized Cases';
            clearBtn.style.display = 'none';
        }
        const display = data.slice(0, 250);
        document.getElementById('table-row-count').innerText = `${display.length} of ${data.length} shown`;

        if (display.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:32px;color:var(--text-secondary)">No cases found.</td></tr>';
            return;
        }
        display.forEach(row => {
            const tr = document.createElement('tr');
            const conf = row.Prediction_Confidence;
            let bc = 'badge-low';
            if (conf >= 0.8) bc = 'badge-high';
            else if (conf >= 0.5) bc = 'badge-medium';
            const pct = Math.round(conf * 100) + '%';
            const reviewed = row._reviewed ? '<span class="badge badge-reviewed">✓ Reviewed</span>' : '';
            tr.innerHTML = `
                <td>${row['Case Number'] || '—'}</td>
                <td class="subject-cell" title="${row.Subject || ''}">${row.Subject || '—'}</td>
                <td>${row.Severity || '—'}</td>
                <td>${row.Main_Category || 'Others'}</td>
                <td><strong>${row.Predicted_Category || 'Others'}</strong>${reviewed}</td>
                <td><span class="badge ${bc}">${pct}</span></td>
                <td>${row['Case Owner'] || '—'}</td>`;
            tr.addEventListener('click', () => openModal(row['Case Number']));
            tbody.appendChild(tr);
        });
    }

    // --- Filter ---
    function applyFilter(field, value) {
        activeFilter = { field, value };
        const filtered = globalData.filter(r => r[field] === value);
        renderTable(filtered, value);
        document.getElementById('table-section').scrollIntoView({ behavior: 'smooth' });
    }

    window.clearFilter = function () {
        activeFilter = null;
        document.getElementById('search-input').value = '';
        renderTable(globalData, null);
    };

    document.getElementById('search-input').addEventListener('input', e => {
        const q = e.target.value.toLowerCase();
        const src = activeFilter ? globalData.filter(r => r[activeFilter.field] === activeFilter.value) : globalData;
        renderTable(src.filter(r =>
            (r['Case Number'] && String(r['Case Number']).toLowerCase().includes(q)) ||
            (r.Subject && r.Subject.toLowerCase().includes(q)) ||
            (r.Predicted_Category && r.Predicted_Category.toLowerCase().includes(q))
        ), activeFilter ? activeFilter.value : null);
    });

    // --- Export CSV ---
    window.exportCSV = function () {
        const src = activeFilter ? globalData.filter(r => r[activeFilter.field] === activeFilter.value) : globalData;
        const cols = ['Case Number', 'Subject', 'Severity', 'Priority', 'Region', 'Main_Category', 'Predicted_Category', 'Prediction_Confidence', 'Case Owner', 'Status'];
        const rows = [cols.join(','), ...src.map(r => cols.map(c => `"${String(r[c] || '').replace(/"/g, '""')}"`).join(','))];
        const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'NeuroSight_Export.csv';
        a.click();
    };

    // --- Case Detail Modal ---
    window.openModal = function (caseNumber) {
        currentCaseNumber = String(caseNumber);
        document.getElementById('modal-overlay').style.display = 'flex';
        document.getElementById('modal-title').innerText = 'Case #' + currentCaseNumber;
        document.getElementById('feedback-msg').style.display = 'none';
        const body = document.getElementById('modal-body');
        body.innerHTML = '<div class="spinner" style="margin:32px auto;"></div>';

        fetch(`http://localhost:8000/api/case/${encodeURIComponent(currentCaseNumber)}`)
            .then(r => { if (!r.ok) throw new Error('Not found'); return r.json(); })
            .then(data => buildModalBody(data))
            .catch(() => { body.innerHTML = '<p style="padding:16px">Could not load case details.</p>'; });
    };

    function buildModalBody(data) {
        const body = document.getElementById('modal-body');
        const conf = parseFloat(data.Prediction_Confidence || 0);
        const confPct = Math.round(conf * 100);
        let barColor = '#ef4444';
        if (conf >= 0.8) barColor = '#10b981';
        else if (conf >= 0.5) barColor = '#f59e0b';

        // Build top-3 HTML from row data (stored in globalData)
        const rowData = globalData.find(r => String(r['Case Number']) === currentCaseNumber) || {};
        const top3 = rowData.top_3 || [];
        const top3Html = top3.length > 0 ? `
            <div class="modal-section">
                <h4>Top-3 AI Predictions</h4>
                ${top3.map((p, i) => {
                    const pct = Math.round(p.confidence * 100);
                    const color = i === 0 ? '#10b981' : i === 1 ? '#f59e0b' : '#94a3b8';
                    return `<div class="modal-field" style="flex-direction:column;gap:4px;">
                        <div style="display:flex;justify-content:space-between;font-size:13px;">
                            <span style="color:${color};">${i+1}. ${p.category}</span>
                            <span style="color:var(--text-secondary);">${pct}%</span>
                        </div>
                        <div class="conf-bar-wrap"><div class="conf-bar" style="width:${pct}%;background:${color}"></div></div>
                    </div>`;
                }).join('')}
            </div>` : '';

        const sections = {
            'AI Prediction': [
                ['Predicted Category', `<strong>${data.Predicted_Category || '—'}</strong>`],
                ['Main Category', data.Main_Category || '—'],
                ['Confidence', `${confPct}% <div class="conf-bar-wrap"><div class="conf-bar" style="width:${confPct}%;background:${barColor}"></div></div>`]
            ],
            'Case Info': [
                ['Case Number', data['Case Number'] || '—'],
                ['Status', data.Status || '—'],
                ['Severity', data.Severity || '—'],
                ['Priority', data.Priority || '—'],
                ['Opened Date', data['Opened Date'] || '—'],
                ['Closed Date', data['Closed Date'] || '—'],
                ['Region', data.Region || '—'],
            ],
            'People': [
                ['Account Name', data['Account Name'] || '—'],
                ['Case Owner', data['Case Owner'] || '—'],
                ['Case Owner Manager', data['Case Owner Manager'] || '—'],
                ['Case Contact', data['Case Contact Email'] || '—'],
            ],
            'Issue Details': [
                ['Subject', data.Subject || '—'],
                ['Product Line', data['Product Line'] || '—'],
                ['Issue Type', data['Issue Type'] || '—'],
                ['Issue Category', data['Issue Category'] || '—'],
                ['Issue', (data['Issue Plain Text'] || '—').slice(0, 400)],
                ['Cause', (data.Cause || '—').slice(0, 300)],
            ],
            'Resolution': [
                ['Resolution Type', data['Resolution Type'] || '—'],
                ['Resolution Code', data['Resolution Code'] || '—'],
                ['Resolution', (data.Resolution || '—').slice(0, 400)],
            ]
        };

        body.innerHTML = top3Html + Object.entries(sections).map(([sec, fields]) => `
            <div class="modal-section">
                <h4>${sec}</h4>
                ${fields.map(([label, val]) => `
                    <div class="modal-field">
                        <span class="modal-field-label">${label}</span>
                        <span class="modal-field-value">${val}</span>
                    </div>`).join('')}
            </div>`).join('');

        const sel = document.getElementById('override-select');
        for (let i = 0; i < sel.options.length; i++) {
            if (sel.options[i].value === data.Predicted_Category) { sel.selectedIndex = i; break; }
        }
    }

    window.closeModal = function (e) {
        if (!e || e.target === document.getElementById('modal-overlay')) {
            document.getElementById('modal-overlay').style.display = 'none';
        }
    };

    // --- Feedback / Override ---
    window.submitFeedback = function () {
        const category = document.getElementById('override-select').value;
        fetch('http://localhost:8000/api/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ case_number: currentCaseNumber, corrected_category: category })
        }).then(r => r.json()).then(resp => {
            document.getElementById('feedback-msg').style.display = 'block';
            // Update active learning counter in sidebar
            const fbEl = document.getElementById('sidebar-fb-val');
            const fbSec = document.getElementById('sidebar-feedback');
            fbSec.style.display = 'block';
            fbEl.innerText = `${resp.total_feedback} correction${resp.total_feedback !== 1 ? 's' : ''}`;
            // Update row in table
            const row = globalData.find(r => String(r['Case Number']) === currentCaseNumber);
            if (row) { row.Predicted_Category = category; row._reviewed = true; }
            const src = activeFilter ? globalData.filter(r => r[activeFilter.field] === activeFilter.value) : globalData;
            renderTable(src, activeFilter ? activeFilter.value : null);
        }).catch(() => alert('Could not save feedback.'));
    };

    // --- Force Retrain (Active Learning) ---
    window.forceRetrain = function () {
        const btn = document.querySelector('[onclick="forceRetrain()"]');
        if (btn) { btn.innerText = '⚡ Retraining...'; btn.disabled = true; }
        fetch('http://localhost:8000/api/retrain', { method: 'POST' })
            .then(r => r.json())
            .then(resp => {
                if (btn) { btn.innerText = '✓ Retrained!'; btn.disabled = false; }
                console.log('Retrain result:', resp);
                setTimeout(() => { if (btn) btn.innerText = '⚡ Retrain Now'; }, 3000);
            })
            .catch(() => { if (btn) { btn.innerText = '⚡ Retrain Now'; btn.disabled = false; } });
    };
});
