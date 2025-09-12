// Main JavaScript for Predictive Maintenance System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize form handling
    initializePredictionForm();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize smooth scrolling
    initializeSmoothScrolling();
});

function initializePredictionForm() {
    const form = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    
    if (!form || !submitBtn) return;
    
    form.addEventListener('submit', function(e) {
        // Show loading state
        showLoadingState(submitBtn);
        
        // Validate form before submission
        if (!validateForm()) {
            e.preventDefault();
            hideLoadingState(submitBtn);
            return false;
        }
    });
    
    // Add real-time validation
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('blur', validateField);
        input.addEventListener('input', clearFieldError);
    });
}

function validateForm() {
    const form = document.getElementById('predictionForm');
    let isValid = true;
    
    // Clear previous errors
    clearFormErrors();
    
    // Validate required fields
    const requiredFields = form.querySelectorAll('[required]');
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            showFieldError(field, 'Bu alan zorunludur');
            isValid = false;
        }
    });
    
    // Validate numeric ranges
    const validationRules = {
        'temperature': { min: -50, max: 150, name: 'Temperature' },
        'vibration': { min: 0, max: 100, name: 'Vibration' },
        'power_consumption': { min: 0, max: 10000, name: 'Power Consumption' },
        'operational_hours': { min: 0, max: 100000, name: 'Operational Hours' },
        'error_codes': { min: 0, max: 1000, name: 'Error Codes' },
        'oil_level': { min: 0, max: 100, name: 'Oil Level' },
        'coolant_level': { min: 0, max: 100, name: 'Coolant Level' },
        'maintenance_count': { min: 0, max: 1000, name: 'Maintenance Count' },
        'failure_count': { min: 0, max: 1000, name: 'Failure Count' }
    };
    
    Object.keys(validationRules).forEach(fieldName => {
        const field = document.getElementById(fieldName);
        if (field && field.value) {
            const value = parseFloat(field.value);
            const rule = validationRules[fieldName];
            
            if (isNaN(value)) {
                showFieldError(field, `${rule.name} geçerli bir sayı olmalıdır`);
                isValid = false;
            } else if (value < rule.min || value > rule.max) {
                showFieldError(field, `${rule.name} ${rule.min} ile ${rule.max} arasında olmalıdır`);
                isValid = false;
            }
        }
    });
    
    return isValid;
}

function validateField(event) {
    const field = event.target;
    const fieldName = field.name;
    
    // Clear previous error
    clearFieldError(field);
    
    // Check if required field is empty
    if (field.hasAttribute('required') && !field.value.trim()) {
        showFieldError(field, 'Bu alan zorunludur');
        return false;
    }
    
    // Validate numeric fields
    if (field.type === 'number' && field.value) {
        const value = parseFloat(field.value);
        const min = parseFloat(field.min);
        const max = parseFloat(field.max);
        
        if (isNaN(value)) {
            showFieldError(field, 'Must be a valid number');
            return false;
        }
        
        if (!isNaN(min) && value < min) {
            showFieldError(field, `Must be at least ${min}`);
            return false;
        }
        
        if (!isNaN(max) && value > max) {
            showFieldError(field, `Must be at most ${max}`);
            return false;
        }
    }
    
    return true;
}

function showFieldError(field, message) {
    field.classList.add('is-invalid');
    
    // Remove existing error message
    const existingError = field.parentNode.querySelector('.invalid-feedback');
    if (existingError) {
        existingError.remove();
    }
    
    // Add new error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    field.parentNode.appendChild(errorDiv);
}

function clearFieldError(field) {
    if (typeof field === 'object' && field.target) {
        field = field.target;
    }
    
    field.classList.remove('is-invalid');
    const errorMessage = field.parentNode.querySelector('.invalid-feedback');
    if (errorMessage) {
        errorMessage.remove();
    }
}

function clearFormErrors() {
    const form = document.getElementById('predictionForm');
    if (!form) return;
    
    const invalidFields = form.querySelectorAll('.is-invalid');
    invalidFields.forEach(field => {
        field.classList.remove('is-invalid');
    });
    
    const errorMessages = form.querySelectorAll('.invalid-feedback');
    errorMessages.forEach(error => {
        error.remove();
    });
}

function showLoadingState(button) {
    button.disabled = true;
    button.classList.add('loading');
    button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>İşleniyor...';
}

function hideLoadingState(button) {
    button.disabled = false;
    button.classList.remove('loading');
    button.innerHTML = '<i class="fas fa-calculator me-1"></i>Arıza Tahmin Et';
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

function initializeSmoothScrolling() {
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Utility functions for API interaction
function callPredictionAPI(data) {
    return fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    });
}

function checkAPIStatus() {
    return fetch('/api_status')
        .then(response => response.json())
        .catch(error => {
            console.error('API status check failed:', error);
            return { status: 'error', models_loaded: false };
        });
}

// Form auto-save functionality (optional)
function enableFormAutoSave() {
    const form = document.getElementById('predictionForm');
    if (!form) return;
    
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        // Load saved value
        const savedValue = localStorage.getItem(`form_${input.name}`);
        if (savedValue && !input.value) {
            input.value = savedValue;
        }
        
        // Save on change
        input.addEventListener('change', function() {
            localStorage.setItem(`form_${this.name}`, this.value);
        });
    });
}

// Export functions for external use
window.PredictiveMaintenance = {
    validateForm,
    callPredictionAPI,
    checkAPIStatus,
    enableFormAutoSave
};
