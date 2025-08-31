# PhishGuard - Phishing Detection System

## Overview

PhishGuard is an AI-powered web application that detects phishing websites using two complementary approaches: URL-based feature analysis and screenshot-based visual analysis. The system employs machine learning models (Random Forest for URL features and CNN for images) to classify websites as legitimate or phishing attempts. Built with Flask, the application provides a user-friendly interface for real-time phishing detection and includes model training capabilities for continuous improvement.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Single-page application** using vanilla HTML, CSS, and JavaScript
- **Bootstrap framework** with dark theme for responsive design
- **Tabbed interface** separating URL analysis, screenshot analysis, and model training
- **Real-time file preview** and form validation on the client side
- **AJAX-based communication** with backend using fetch API

### Backend Architecture
- **Flask web framework** serving as the main application server
- **RESTful API design** with dedicated endpoints for different detection methods
- **File upload handling** with security validations (file type, size limits)
- **Model loading system** that dynamically loads trained ML models at startup
- **Error handling and logging** throughout the application stack

### Machine Learning Pipeline
- **Dual detection approach**: URL feature extraction (31 features) and image-based analysis
- **Random Forest classifier** for URL-based detection using extracted features like URL length, special characters, domain characteristics
- **Convolutional Neural Network (CNN)** for screenshot-based detection using TensorFlow/Keras
- **Feature extraction utilities** for processing URLs and preprocessing images
- **Model persistence** using joblib for scikit-learn models and HDF5 for neural networks

### Data Processing
- **URL feature extraction** including domain analysis, character counting, and pattern detection
- **Image preprocessing** for screenshot analysis with PIL and OpenCV
- **Synthetic data generation** for training when real datasets are unavailable
- **File handling** with secure filename processing and upload directory management

### Security Measures
- **File upload restrictions** limiting file types and sizes
- **Secure filename handling** using Werkzeug utilities
- **Session management** with configurable secret keys
- **Input validation** on both client and server sides

## External Dependencies

### Core Framework Dependencies
- **Flask**: Web application framework and routing
- **Werkzeug**: WSGI utilities for secure file handling
- **Jinja2**: Template engine (included with Flask)

### Machine Learning Stack
- **scikit-learn**: Random Forest classifier and preprocessing tools
- **TensorFlow/Keras**: Deep learning framework for CNN models
- **joblib**: Model serialization and persistence
- **NumPy**: Numerical computing and array operations
- **pandas**: Data manipulation and analysis

### Image Processing
- **PIL (Pillow)**: Image loading, manipulation, and preprocessing
- **OpenCV (cv2)**: Advanced image processing operations

### URL Analysis
- **tldextract**: Top-level domain extraction and URL parsing
- **urllib.parse**: URL parsing and component extraction

### Frontend Libraries
- **Bootstrap**: CSS framework for responsive design
- **Font Awesome**: Icon library for UI elements

### Development and Deployment
- **Logging**: Python's built-in logging for debugging and monitoring
- **OS utilities**: File system operations and environment variable handling

### Optional/Future Dependencies
- **Database integration**: Currently uses file-based model storage, can be extended with SQLAlchemy for data persistence
- **Authentication services**: Can integrate with OAuth providers for user management
- **Cloud storage**: Can integrate with AWS S3 or similar for model and file storage