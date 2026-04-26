## Real-Time Email Spam Classifier (PyTorch + FastAPI + Gmail Integration)

# Overview
This project is an end-to-end email spam classification system that combines machine learning, API development, and real-world automation.

# What is this project?
A real-time spam detection system that connects to your Gmail inbox, 
classifies incoming emails using a trained ML model, and automatically 
forwards spam emails.

The system:

* Trains a spam classifier using a Kaggle dataset
* Serves the model via FastAPI
* Fetches real emails from Gmail using IMAP
* Classifies emails in real-time
* Automatically forwards high-confidence spam emails to another email address

---

# Features

* Spam classification using PyTorch
* Text preprocessing with TF-IDF (scikit-learn)
* FastAPI REST API for real-time inference
* Gmail integration via IMAP
* Automatic spam forwarding via SMTP
* Modular and extensible project structure

---

# Tech Stack

* Python
* PyTorch
* Scikit-learn
* FastAPI
* IMAP (Gmail)
* SMTP (Email forwarding)
* BeautifulSoup (HTML parsing)

---

# Project Structure

```
MAIL SPAM PROJECT
├── Data/
│   ├── norm_spam.csv
│   └── norm_spam.arff
├── api.py
├── gmail_inference.py
├── model.py
├── train.py
├── dataset.py
├── prepare_data.py
├── inference.py
├── best_model.pth
├── vectorizer.pkl
├── requirements.txt
├── .gitignore
└── .env (not included)
```

---

# Installation

Clone the repository:

```bash
# Clone the repository
git clone https://github.com/eraygenc-eng/real-time-email-spam-classifier.git

# Navigate into project
cd real-time-email-spam-classifier
```

Install dependencies:

```bash

# Install dependencies
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the root directory:

```
EMAIL=your_gmail@gmail.com
APP_PASSWORD=your_app_password
FORWARD_EMAIL=destination_email@outlook.com
```

> Never share your `.env` file publicly.

---

# Model Usage

The trained model (`best_model.pth`) and vectorizer (`vectorizer.pkl`) are already included.

### Run the API

```bash
uvicorn api:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

Test with:

```json
{
  "text": "Free money win now!!!"
}
```

---

# Gmail Integration

Run the email processing script:

```bash
python gmail_inference.py
```

This will:

1. Connect to your Gmail inbox
2. Fetch recent emails
3. Send email content to the API
4. Classify emails as spam or normal
5. Automatically forward spam emails (high confidence only)

---

# Spam Forwarding Logic

Emails are forwarded if:

```
prediction == "spam" AND probability >= 0.90
```

This helps reduce false positives.

---

# Limitations

* The model is trained primarily on a Turkish email dataset.
* As a result, its performance on English emails may be limited or less accurate.
* The model is trained on a public Kaggle dataset, which may not fully represent real-world email distributions.
* Promotional emails may sometimes be classified as spam.
* The system is designed for learning and demonstration purposes, not production deployment.

---

# Future Improvements

* Model fine-tuning with real email data
* Deployment (Render / Docker)
* Web interface or dashboard
* Logging and monitoring system

---

# Conclusion

This project demonstrates how to build a complete machine learning pipeline:

```
Data → Model → API → Real-world integration → Automation
```

It goes beyond model training and focuses on building usable AI systems.
