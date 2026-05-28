# 🤖 AI Face Recognition Attendance System

An AI-powered web-based attendance management system built using Django, MySQL, OpenCV, and DeepFace.

This project automatically detects and recognizes employee faces through a webcam and marks attendance in real time using deep learning-based face embeddings.

---

# 📌 Features

- 👤 User Registration System
- 📸 Automatic Face Dataset Collection
- 🧠 AI Face Recognition using Deep Learning
- 🕒 Automatic Attendance Marking
- 🔐 Django Authentication System
- 📊 Employee Dashboard
- 🗄️ MySQL Database Integration
- 📷 Real-Time Webcam Scanning
- 🧬 Face Embedding Generation
- ⚡ Fast Face Matching
- 🛡️ Duplicate Attendance Prevention

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Django | Backend Framework |
| Python | Programming Language |
| MySQL | Database |
| OpenCV | Image Processing |
| DeepFace | Face Recognition |
| HTML/CSS/JavaScript | Frontend |
| Bootstrap | UI Styling |
| NumPy | Numerical Operations |

---

# 🧠 How Face Recognition Works

The system does NOT compare images directly.

Instead:

```text
Face Image
    ↓
AI Model (FaceNet)
    ↓
Face Embedding Vector
    ↓
Vector Comparison
    ↓
Recognition Result
```

Each face is converted into a mathematical vector called an embedding.

The system compares:
- current webcam face embedding
with
- stored employee embeddings

using Euclidean Distance.

If the distance is below a threshold:
- the face is recognized
- attendance is marked automatically

---

# 📂 Project Structure

```text
face_attendance_system/
│
├── accounts/
├── attendance/
├── recognition/
├── templates/
├── media/
│   ├── datasets/
│   └── encodings/
│
├── static/
├── manage.py
└── requirements.txt
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/face-attendance-system.git

cd face-attendance-system
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

OR manually:

```bash
pip install django mysqlclient opencv-python deepface tensorflow numpy pillow
```

---

## 4️⃣ Configure MySQL Database

Create database:

```sql
CREATE DATABASE face_attendance_db;
```

---

## 5️⃣ Configure Django Database Settings

Inside `settings.py`

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'face_attendance_db',
        'USER': 'root',
        'PASSWORD': 'your_password',
        'HOST': 'localhost',
        'PORT': '3306',
    }
}
```

---

## 6️⃣ Run Migrations

```bash
python manage.py makemigrations

python manage.py migrate
```

---

## 7️⃣ Create Superuser

```bash
python manage.py createsuperuser
```

---

## 8️⃣ Start Server

```bash
python manage.py runserver
```

---

# 🌐 Routes

| Route | Description |
|---|---|
| `/register/` | Register Employee |
| `/login/` | Employee Login |
| `/dashboard/` | Attendance Dashboard |
| `/scan/` | Face Recognition Scanner |
| `/admin/` | Django Admin Panel |

---

# 📸 Registration Workflow

1. Employee registers
2. Webcam captures multiple face images
3. Dataset images are saved
4. DeepFace generates embeddings
5. Encodings stored in filesystem

---

# 🕒 Attendance Workflow

1. User opens scanner
2. Webcam captures live face
3. Face embedding generated
4. Compared with stored embeddings
5. Attendance automatically saved

---

# 🧬 Face Recognition Pipeline

```text
Webcam
   ↓
Capture Face
   ↓
DeepFace / FaceNet
   ↓
Generate Embedding
   ↓
Compare Embeddings
   ↓
Find Match
   ↓
Mark Attendance
```

---

# 📊 Database Models

## Employee Model

```python
class Employee(models.Model):

    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE
    )

    profile_image = models.ImageField(
        upload_to='profiles/'
    )

    phone = models.CharField(max_length=20)

    department = models.CharField(max_length=100)
```

---

## Attendance Model

```python
class Attendance(models.Model):

    employee = models.ForeignKey(
        Employee,
        on_delete=models.CASCADE
    )

    check_in = models.DateTimeField(
        auto_now_add=True
    )

    status = models.CharField(
        max_length=20,
        default='Present'
    )
```

---

# 🔥 AI Model Used

This project uses FaceNet through DeepFace.

FaceNet generates high-dimensional facial embeddings for accurate recognition.

---

# 🛡️ Current Security Features

- Password Authentication
- Login Required Dashboard
- Duplicate Attendance Prevention
- AI Face Verification

---

# 🚧 Future Improvements

- Anti-Spoofing Detection
- Blink Detection
- Real-Time Bounding Boxes
- Multiple Face Detection
- Email Notifications
- Attendance Reports
- CSV/PDF Export
- REST API
- Docker Deployment
- Cloud Deployment
- Mobile Application

---

# 📷 Screenshots

## Registration Page

_Add screenshot here_

---

## Dashboard

_Add screenshot here_

---

## Face Scanner

_Add screenshot here_

---

# 📦 Requirements

Create `requirements.txt`

```txt
django
mysqlclient
opencv-python
deepface
tensorflow
numpy
pillow
```

Install:

```bash
pip install -r requirements.txt
```

---

# 🤝 Contributing

Pull requests are welcome.

For major changes:
- open an issue first
- discuss proposed changes

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

## Muhammad Abdullah

- GitHub: https://github.com/your-github
- LinkedIn: https://linkedin.com/in/your-linkedin

---

# ⭐ Support

If you like this project:

⭐ Star the repository  
🍴 Fork the repository  
🧠 Contribute improvements
