# Face Detection Attendance System

## Table of Contents

- [Overview](#overview)
- [Features](#features)
  - [User Registration](#user-registration)
  - [User Authentication](#user-authentication)
  - [Dashboard](#dashboard)
  - [Face Recognition](#face-recognition)
  - [Real-Time Attendance Scanner](#real-time-attendance-scanner)
- [System Workflow](#system-workflow)
- [Application Routes](#application-routes)
- [Technologies Used](#technologies-used)
  - [Backend](#backend)
  - [Frontend](#frontend)
  - [Computer Vision](#computer-vision)
  - [Database](#database)
- [Project Structure](#project-structure)
- [Face Recognition Process](#face-recognition-process)
  - [Step 1: User Registration](#step-1-user-registration)
  - [Step 2: Dataset Creation](#step-2-dataset-creation)
  - [Step 3: Embedding Generation](#step-3-embedding-generation)
  - [Step 4: Face Scanning](#step-4-face-scanning)
  - [Step 5: Face Matching](#step-5-face-matching)
  - [Step 6: Attendance Recording](#step-6-attendance-recording)
- [Database Models](#database-models)
  - [User Model](#user-model)
  - [Attendance Model](#attendance-model)
- [Installation Guide](#installation-guide)
  - [Clone Repository](#1-clone-repository)
  - [Create Virtual Environment](#2-create-virtual-environment)
  - [Install Dependencies](#3-install-dependencies)
  - [Apply Migrations](#4-apply-migrations)
  - [Create Superuser](#5-create-superuser-optional)
  - [Run Development Server](#6-run-development-server)
- [Requirements](#requirementstxt)
- [Common Commands](#common-commands)
- [Screenshots](#screenshots)
  - [Registration Page](#registration-page)
  - [Login Page](#login-page)
  - [Dashboard](#dashboard-1)
  - [Face Scanner](#face-scanner)
- [Security Features](#security-features)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)
- 
## 📌 Overview

The Face Detection Attendance System is a Django-based web application that automates attendance management using Face Recognition technology. The system allows users to register, create facial datasets, log in, and mark attendance through real-time face scanning.

Instead of traditional attendance methods, the system identifies users through facial features and records attendance automatically with date and time information.

---

## 🚀 Features

### 👤 User Registration
Users can create an account by providing:

- Full Name
- Phone Number
- Department
- Profile Picture

During registration, the system captures multiple facial images and automatically creates a dataset for the user.

### 🔐 User Authentication

- User Login
- Secure Session Management
- Dashboard Access

### 📊 Dashboard

The dashboard provides:

- User Information
- Attendance Records
- Attendance Date
- Attendance Time
- Attendance History

### 🤖 Face Recognition

The system uses face recognition technology to:

1. Detect faces from camera input.
2. Generate facial embeddings.
3. Store embeddings in the database.
4. Compare scanned faces with registered users.
5. Mark attendance automatically upon successful recognition.

### 📷 Real-Time Attendance Scanner

A dedicated scanning window captures live video and:

- Detects faces
- Matches embeddings
- Identifies users
- Records attendance automatically

---

# 🏗️ System Workflow

```text
Registration
     │
     ▼
Dataset Creation
     │
     ▼
Face Embedding Generation
     │
     ▼
User Login
     │
     ▼
Dashboard
     │
     ▼
Scan Face
     │
     ▼
Face Recognition
     │
     ▼
Attendance Marked
```

---

# 🌐 Application Routes

| Route | Description |
|---------|-------------|
| `/register` | User Registration |
| `/login` | User Login |
| `/dashboard` | User Dashboard |
| `/scan` | Face Scanning & Attendance |

Development URLs:

```text
http://127.0.0.1:8000/register
http://127.0.0.1:8000/login
http://127.0.0.1:8000/dashboard
http://127.0.0.1:8000/scan
```

---

# ⚙️ Technologies Used

## Backend

- Django
- Python

## Frontend

- HTML
- CSS
- Bootstrap
- JavaScript

## Computer Vision

- OpenCV
- Face Recognition
- NumPy

## Database

- SQLite (Default)
- MySQL (Optional)

---

# 📂 Project Structure

```text
FaceAttendanceSystem/
│
├── attendance/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   └── face_recognition.py
│
├── templates/
│   ├── register.html
│   ├── login.html
│   ├── dashboard.html
│   └── scan.html
│
├── media/
│   ├── profile_pics/
│   └── datasets/
│
├── static/
│
├── db.sqlite3
├── manage.py
├── requirements.txt
└── README.md
```

---

# 🧠 Face Recognition Process

## Step 1: User Registration

The user submits:

- Name
- Phone Number
- Department
- Profile Picture

The system creates a facial dataset.

---

## Step 2: Dataset Creation

Multiple facial samples are captured and stored:

```text
datasets/
└── User_Name/
    ├── img1.jpg
    ├── img2.jpg
    ├── img3.jpg
    └── ...
```

---

## Step 3: Embedding Generation

The facial images are converted into numerical embeddings.

Example:

```python
[0.1234, -0.4321, 0.8876, ...]
```

These embeddings represent unique facial characteristics.

---

## Step 4: Face Scanning

When the scanner is opened:

```text
/scan
```

The camera captures live frames.

---

## Step 5: Face Matching

The scanned face embedding is compared with stored embeddings.

If a match is found:

```text
Attendance Marked Successfully
```

---

## Step 6: Attendance Recording

Attendance information is saved:

| Name | Date | Time |
|--------|--------|--------|
| John Doe | 2025-06-10 | 09:00 AM |

---

# 🗄️ Database Models

## User Model

```python
class User:
    name
    phone_number
    department
    profile_picture
```

## Attendance Model

```python
class Attendance:
    user
    date
    time
    status
```

---

# 📦 Installation Guide

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/face-attendance-system.git

cd face-attendance-system
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux/Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Apply Migrations

```bash
python manage.py makemigrations

python manage.py migrate
```

---

## 5. Create Superuser (Optional)

```bash
python manage.py createsuperuser
```

---

## 6. Run Development Server

```bash
python manage.py runserver
```

Server starts at:

```text
http://127.0.0.1:8000/
```

---

# 📋 requirements.txt

Create a file named:

```text
requirements.txt
```

and add:

```txt
Django>=4.2

opencv-python

numpy

face-recognition

face-recognition-models

Pillow

dlib

scipy

cmake
```

Install using:

```bash
pip install -r requirements.txt
```

---

# 🔧 Common Commands

### Run Server

```bash
python manage.py runserver
```

### Make Migrations

```bash
python manage.py makemigrations
```

### Apply Migrations

```bash
python manage.py migrate
```

### Create Admin User

```bash
python manage.py createsuperuser
```

### Collect Static Files

```bash
python manage.py collectstatic
```

---

# 📸 Screenshots

## Registration Page

<img width="1126" height="878" alt="Screenshot 2026-06-03 121354" src="https://github.com/user-attachments/assets/653d9b0a-ab9f-48e5-a369-15eff8b1ad50" />

## Login Page

<img width="498" height="588" alt="Screenshot 2026-06-03 121330" src="https://github.com/user-attachments/assets/a834b842-4532-4d06-98c4-2dccc3661458" />

## Dashboard

<img width="1919" height="910" alt="Screenshot 2026-06-03 121315" src="https://github.com/user-attachments/assets/1b58db67-bfcf-4dd2-a100-c75c01fe1888" />

## Face Scanner

<img width="875" height="776" alt="Screenshot 2026-06-03 121256" src="https://github.com/user-attachments/assets/7b96e625-28cc-4610-b432-2db65a655170" />

---

# 🔒 Security Features

- User Authentication
- Face-Based Identification
- Unique Face Embeddings
- Duplicate Attendance Prevention
- Secure Database Storage

---

# 🎯 Future Improvements

- Multi-Camera Support
- Email Notifications
- Attendance Reports (PDF/Excel)
- Admin Analytics Dashboard
- Cloud Deployment
- Anti-Spoofing Detection
- Mobile Application Integration

---

# 👨‍💻 Author

**Muhammad Abdullah**

Face Detection Attendance System using Django, OpenCV, and Face Recognition.

---

# 📜 License

This project is licensed under the MIT License.

Feel free to use and modify this project for educational and research purposes.
