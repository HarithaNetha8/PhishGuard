// PhishGuard JavaScript functionality

document.addEventListener('DOMContentLoaded', function() {
    // Initialize form handlers
    initializeFormHandlers();
    initializeFilePreview();
    initializeFeatureExtraction();
});

function initializeFormHandlers() {
    // URL form handler
    const urlForm = document.getElementById('urlForm');
    if (urlForm) {
        urlForm.addEventListener('submit', handleUrlSubmission);
    }

    // Screenshot form handler
    const screenshotForm = document.getElementById('screenshotForm');
    if (screenshotForm) {
        screenshotForm.addEventListener('submit', handleScreenshotSubmission);
    }
}

function initializeFilePreview() {
    const fileInput = document.getElementById('screenshotFile');
    const preview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');

    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Validate file size (16MB max)
                if (file.size > 16 * 1024 * 1024) {
                    showAlert('File size must be less than 16MB', 'danger');
                    fileInput.value = '';
                    return;
                }

                // Validate file type
                const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp', 'image/webp'];
                if (!allowedTypes.includes(file.type)) {
                    showAlert('Please select a valid image file (PNG, JPG, JPEG, GIF, BMP, WebP)', 'danger');
                    fileInput.value = '';
                    return;
                }

                // Show preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImg.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            } else {
                preview.style.display = 'none';
            }
        });
    }
}

function initializeFeatureExtraction() {
    const extractBtn = document.getElementById('extractFeaturesBtn');
    const urlInput = document.getElementById('urlInput');

    if (extractBtn && urlInput) {
        extractBtn.addEventListener('click', function() {
            const url = urlInput.value.trim();
            if (!url) {
                showAlert('Please enter a URL first', 'warning');
                return;
            }

            // Validate URL format
            try {
                new URL(url);
            } catch (e) {
                showAlert('Please enter a valid URL (e.g., https://example.com)', 'warning');
                return;
            }

            extractFeatures(url);
        });
    }
}

async function extractFeatures(url) {
    const extractBtn = document.getElementById('extractFeaturesBtn');
    const originalText = extractBtn.innerHTML;
    
    try {
        // Show loading state
        extractBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Extracting...';
        extractBtn.disabled = true;

        const response = await fetch('/extract_features', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url })
        });

        const data = await response.json();

        if (data.error) {
            showAlert(`Error extracting features: ${data.error}`, 'danger');
            return;
        }

        // Populate form fields with extracted features
        const features = data.features;
        const featureNames = data.feature_names;
        
        featureNames.forEach((name, index) => {
            const input = document.querySelector(`[name="${name}"]`);
            if (input && features[index] !== undefined) {
                input.value = features[index];
            }
        });

        showAlert('Features extracted successfully! You can now analyze the URL.', 'success');

        // Expand the additional features section if it's collapsed
        const collapseElement = document.getElementById('collapseFeatures');
        if (collapseElement && !collapseElement.classList.contains('show')) {
            const accordionButton = document.querySelector('[data-bs-target="#collapseFeatures"]');
            if (accordionButton) {
                accordionButton.click();
            }
        }

    } catch (error) {
        console.error('Error:', error);
        showAlert('Failed to extract features. Please try again.', 'danger');
    } finally {
        // Restore button state
        extractBtn.innerHTML = originalText;
        extractBtn.disabled = false;
    }
}

async function handleUrlSubmission(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const features = [];
    
    // Feature names in the correct order
    const featureNames = [
        'length_url', 'length_hostname', 'ip', 'nb_dots', 'nb_hyphens',
        'nb_at', 'nb_qm', 'nb_and', 'nb_or', 'nb_eq', 'nb_underscore',
        'nb_tilde', 'nb_percent', 'nb_slash', 'nb_star', 'nb_colon',
        'nb_comma', 'nb_semicolon', 'nb_dollar', 'nb_space', 'nb_www',
        'nb_com', 'nb_dslash', 'http_in_path', 'https_token', 'ratio_digits_url',
        'ratio_digits_host', 'punycode', 'port', 'tld_in_path', 'tld_in_subdomain'
    ];
    
    // Extract features in the correct order
    featureNames.forEach(name => {
        const value = formData.get(name);
        features.push(parseFloat(value) || 0);
    });
    
    // Validate features
    if (features.length !== 31) {
        showAlert('Please fill in all 31 features', 'warning');
        return;
    }
    
    showLoading(true);
    hideResults();
    
    try {
        const response = await fetch('/predict_url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ features: features })
        });
        
        const data = await response.json();
        showResults(data, 'url');
        
    } catch (error) {
        console.error('Error:', error);
        showAlert('Failed to analyze URL. Please try again.', 'danger');
    } finally {
        showLoading(false);
    }
}

async function handleScreenshotSubmission(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('screenshotFile');
    const file = fileInput.files[0];
    
    if (!file) {
        showAlert('Please select a screenshot file', 'warning');
        return;
    }
    
    showLoading(true);
    hideResults();
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/predict_screenshot', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        showResults(data, 'screenshot');
        
    } catch (error) {
        console.error('Error:', error);
        showAlert('Failed to analyze screenshot. Please try again.', 'danger');
    } finally {
        showLoading(false);
    }
}

function showResults(data, type) {
    const resultsDiv = document.getElementById('results');
    const resultsContent = document.getElementById('resultsContent');
    
    if (data.error) {
        resultsContent.innerHTML = `
            <div class="result-warning">
                <i class="fas fa-exclamation-triangle fa-2x mb-3"></i>
                <h4>Analysis Error</h4>
                <p>${data.error}</p>
                <small>Please ensure the model is trained and try again.</small>
            </div>
        `;
    } else {
        const prediction = data.prediction;
        const confidence = data.confidence || 0;
        const probabilities = data.probabilities || { safe: 0, phishing: 0 };
        
        const isPhishing = prediction === 1;
        const resultClass = isPhishing ? 'result-phishing' : 'result-safe';
        const icon = isPhishing ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle';
        const title = isPhishing ? 'Phishing Detected!' : 'Website Appears Safe';
        const description = isPhishing ? 
            'This website shows characteristics of a phishing site. Exercise caution!' :
            'This website appears to be legitimate based on our analysis.';
        
        resultsContent.innerHTML = `
            <div class="${resultClass}">
                <i class="${icon} fa-3x mb-3"></i>
                <h3>${title}</h3>
                <p class="mb-3">${description}</p>
                <div class="confidence-info">
                    <small>Confidence: ${(confidence * 100).toFixed(1)}%</small>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body text-center">
                            <h6 class="text-success">Safe Probability</h6>
                            <h4>${(probabilities.safe * 100).toFixed(1)}%</h4>
                            <div class="progress">
                                <div class="progress-bar bg-success" style="width: ${probabilities.safe * 100}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body text-center">
                            <h6 class="text-danger">Phishing Probability</h6>
                            <h4>${(probabilities.phishing * 100).toFixed(1)}%</h4>
                            <div class="progress">
                                <div class="progress-bar bg-danger" style="width: ${probabilities.phishing * 100}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="mt-4 text-center">
                <small class="text-muted">
                    Analysis performed using ${type === 'url' ? 'Random Forest' : 'CNN'} model
                </small>
            </div>
        `;
    }
    
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = show ? 'block' : 'none';
    }
}

function hideResults() {
    const resultsDiv = document.getElementById('results');
    if (resultsDiv) {
        resultsDiv.style.display = 'none';
    }
}

function showAlert(message, type = 'info') {
    // Remove existing alerts
    const existingAlerts = document.querySelectorAll('.dynamic-alert');
    existingAlerts.forEach(alert => alert.remove());
    
    // Create new alert
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show dynamic-alert mt-3`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to the active tab
    const activeTab = document.querySelector('.tab-pane.active');
    if (activeTab) {
        const firstCard = activeTab.querySelector('.card');
        if (firstCard) {
            firstCard.insertAdjacentElement('beforebegin', alertDiv);
        }
    }
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}


// Utility functions
function validateUrl(string) {
    try {
        new URL(string);
        return true;
    } catch (_) {
        return false;
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Animation utilities
function animateNumber(element, start, end, duration) {
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const currentValue = start + (end - start) * progress;
        element.textContent = Math.round(currentValue);
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}
