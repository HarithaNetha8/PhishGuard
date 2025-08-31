

```markdown
# PhishGuard 🚨

**Code Without Limits: Real-Time Phishing Detection**

PhishGuard is an AI-powered web application that detects phishing websites using **URL-based features** and **screenshot analysis**. The system combines traditional machine learning (Random Forest) and deep learning (CNN) to enhance phishing detection accuracy.

---

## Table of Contents
- [Features](#features)  
- [Demo](#demo)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Technologies](#technologies)  
- [Models](#models)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Features
- **URL-Based Detection:** Uses 31 features extracted from URLs to predict phishing (Random Forest).  
- **Screenshot-Based Detection:** Classifies website screenshots as safe or phishing using CNN.  
- **User-Friendly Interface:** Web frontend for entering features or uploading screenshots.  
- **REST API:** Flask backend with `/predict_url` and `/predict_screenshot` endpoints.  
- **Cross-Platform Deployment:** Can be hosted online on Replit, Heroku, or any cloud service.  

---



URL Input → Predict → Output: Phishing
Screenshot Upload → Predict → Output: Safe


---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/HarithaNetha8/PhishGuard.git
cd PhishGuard
````

2. **Create a virtual environment**

```bash
python -m venv venv
```

3. **Activate virtual environment**

* Windows:

```bash
venv\Scripts\activate
```

* Linux / Mac:

```bash
source venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Run the app**

```bash
python app.py
```

---

## Usage

### URL-Based Detection

Send a POST request to `/predict_url` with JSON:

```json
{
  "features": [1,0,1,1,0,1,...]  // 31 features
}
```

Response:

```json
{
  "prediction": 1
}
```

* `1` → Phishing
* `0` → Safe

### Screenshot-Based Detection

Send a POST request to `/predict_screenshot` with form-data containing the screenshot file.
Response:

```json
{
  "prediction": 0
}
```

* `1` → Phishing
* `0` → Safe

---

## Project Structure

```
PhishGuard/
├── app.py                   # Flask backend
├── train_url_model.py       # Train URL-based Random Forest
├── train_screenshot_model.py# Train CNN screenshot classifier
├── models/
│   ├── random_forest_model.pkl
│   └── cnn_screenshot_model.h5
├── templates/
│   └── index.html           # Frontend HTML
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── requirements.txt
└── README.md
```

---

## Technologies

* Python 3.10
* Flask
* Pandas & NumPy
* Scikit-learn (Random Forest)
* TensorFlow/Keras (CNN for screenshots)
* OpenCV & Pillow (image handling)
* HTML, CSS, JavaScript (frontend)

---

## Models

1. **Random Forest:** Predict phishing based on URL features.
2. **CNN:** Predict phishing based on screenshots of websites.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features.

---

## License

This project is licensed under the MIT License.

```



Do you want me to do that next?
```
