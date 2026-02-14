// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global state
let currentFileId = null;
let currentColumns = [];
let currentJobId = null;
let pollInterval = null;

// DOM Elements
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const datasetStats = document.getElementById('dataset-stats');
const datasetPreview = document.getElementById('dataset-preview');
const problemType = document.getElementById('problem-type');
const targetColumn = document.getElementById('target-column');
const dateColumn = document.getElementById('date-column');
const testSize = document.getElementById('test-size');
const testSizeValue = document.getElementById('test-size-value');
const sequenceLength = document.getElementById('sequence-length');
const sequenceLengthValue = document.getElementById('sequence-length-value');
const trainBtn = document.getElementById('train-btn');
const trainingProgress = document.getElementById('training-progress');
const progressFill = document.getElementById('progress-fill');
const progressStatus = document.getElementById('progress-status');
const resultsContent = document.getElementById('results-content');
const modelsList = document.getElementById('models-list');

// Tab Navigation
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;

        // Update active tab button
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Update active tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(`${tabName}-tab`).classList.add('active');

        // Load models when switching to models tab
        if (tabName === 'models') {
            loadModels();
        }
    });
});

// File Upload Handlers
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFileUpload(file);
});

// Handle File Upload
async function handleFileUpload(file) {
    if (!file.name.endsWith('.csv')) {
        alert('Please upload a CSV file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const data = await response.json();
        currentFileId = data.file_id;
        currentColumns = data.columns;

        displayFileInfo(data);
        populateColumnSelectors(data.columns);
        trainBtn.disabled = false;

        // Switch to train tab
        document.querySelector('[data-tab="train"]').click();

    } catch (error) {
        alert('Error uploading file: ' + error.message);
    }
}

// Display File Info
function displayFileInfo(data) {
    fileInfo.style.display = 'block';

    // Stats
    datasetStats.innerHTML = `
        <div class="stat-card">
            <div class="stat-label">Rows</div>
            <div class="stat-value">${data.rows.toLocaleString()}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Columns</div>
            <div class="stat-value">${data.columns.length}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">File</div>
            <div class="stat-value">${data.filename}</div>
        </div>
    `;

    // Preview Table
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    const tbody = document.createElement('tbody');

    // Header
    const headerRow = document.createElement('tr');
    data.columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);

    // Rows
    data.preview.forEach(row => {
        const tr = document.createElement('tr');
        data.columns.forEach(col => {
            const td = document.createElement('td');
            td.textContent = row[col] ?? 'null';
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });

    table.appendChild(thead);
    table.appendChild(tbody);
    datasetPreview.innerHTML = '';
    datasetPreview.appendChild(table);
}

// Populate Column Selectors
function populateColumnSelectors(columns) {
    targetColumn.innerHTML = '<option value="">Select target column...</option>';
    dateColumn.innerHTML = '<option value="">Auto-detect</option>';

    columns.forEach(col => {
        const option1 = document.createElement('option');
        option1.value = col;
        option1.textContent = col;
        targetColumn.appendChild(option1);

        const option2 = document.createElement('option');
        option2.value = col;
        option2.textContent = col;
        dateColumn.appendChild(option2);
    });
}

// Problem Type Change
problemType.addEventListener('change', () => {
    const isTimeSeries = problemType.value === 'timeseries';
    document.getElementById('date-column-group').style.display = isTimeSeries ? 'block' : 'none';
    document.getElementById('sequence-length-group').style.display = isTimeSeries ? 'block' : 'none';
});

// Range Sliders
testSize.addEventListener('input', (e) => {
    testSizeValue.textContent = `${e.target.value}%`;
});

sequenceLength.addEventListener('input', (e) => {
    sequenceLengthValue.textContent = e.target.value;
});

// Train Model
trainBtn.addEventListener('click', async () => {
    if (!targetColumn.value) {
        alert('Please select a target column');
        return;
    }

    const requestData = {
        problem_type: problemType.value,
        target_column: targetColumn.value,
        date_column: dateColumn.value || null,
        test_size: parseFloat(testSize.value) / 100,
        sequence_length: parseInt(sequenceLength.value)
    };

    try {
        trainBtn.disabled = true;
        trainingProgress.style.display = 'block';
        progressFill.style.width = '0%';
        progressStatus.textContent = 'Starting training...';

        const response = await fetch(`${API_BASE_URL}/api/train?file_id=${currentFileId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) throw new Error('Training failed');

        const data = await response.json();
        currentJobId = data.job_id;

        // Start polling for job status
        pollJobStatus();

    } catch (error) {
        alert('Error starting training: ' + error.message);
        trainBtn.disabled = false;
        trainingProgress.style.display = 'none';
    }
});

// Poll Job Status
async function pollJobStatus() {
    if (!currentJobId) return;

    pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/jobs/${currentJobId}`);
            const job = await response.json();

            // Update progress
            progressFill.style.width = `${job.progress}%`;
            progressFill.textContent = `${job.progress}%`;
            progressStatus.textContent = job.status === 'running' ? 'Training models...' : job.status;

            if (job.status === 'completed') {
                clearInterval(pollInterval);
                displayResults(job.result);
                trainBtn.disabled = false;
                trainingProgress.style.display = 'none';

                // Switch to results tab
                document.querySelector('[data-tab="results"]').click();
            } else if (job.status === 'failed') {
                clearInterval(pollInterval);
                alert('Training failed: ' + job.error);
                trainBtn.disabled = false;
                trainingProgress.style.display = 'none';
            }
        } catch (error) {
            console.error('Error polling job status:', error);
        }
    }, 2000); // Poll every 2 seconds
}

// Display Results
function displayResults(result) {
    const metrics = result.metrics;
    const problemType = result.problem_type;

    let metricsHTML = '<div class="metric-grid">';

    // Display relevant metrics
    for (const [key, value] of Object.entries(metrics)) {
        if (key !== 'model_name' && typeof value === 'number') {
            metricsHTML += `
                <div class="metric-card">
                    <div class="metric-label">${key.toUpperCase()}</div>
                    <div class="metric-value">${value.toFixed(4)}</div>
                </div>
            `;
        }
    }

    metricsHTML += '</div>';

    resultsContent.innerHTML = `
        <h3>🏆 Best Model: ${result.best_model}</h3>
        ${metricsHTML}
        <div class="visualization">
            <h3>Model Comparison</h3>
            <img src="${API_BASE_URL}/api/artifacts/${result.model_id}/model_comparison.png" 
                 alt="Model Comparison" onerror="this.style.display='none'">
        </div>
    `;
}

// Load Models
async function loadModels() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/models`);
        const data = await response.json();

        if (data.models.length === 0) {
            modelsList.innerHTML = '<p class="empty-state">No models saved yet.</p>';
            return;
        }

        modelsList.innerHTML = data.models.map(model => `
            <div class="model-card">
                <div class="model-header">
                    <div class="model-name">${model.best_model}</div>
                    <div class="model-date">${new Date(model.created_at).toLocaleDateString()}</div>
                </div>
                <div class="model-metrics">
                    <div class="model-metric">
                        <strong>Type:</strong> ${model.problem_type}
                    </div>
                    ${Object.entries(model.metrics).slice(0, 3).map(([key, value]) => `
                        <div class="model-metric">
                            <strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(4) : value}
                        </div>
                    `).join('')}
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Error loading models:', error);
        modelsList.innerHTML = '<p class="empty-state">Error loading models.</p>';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('AutoTabML Frontend Loaded');
});
