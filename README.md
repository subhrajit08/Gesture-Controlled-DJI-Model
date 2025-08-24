# Gesture-Controlled DJI Model 🚁✋

This project implements a **gesture-controlled system** to interact with a DJI drone using hand signs.  
It includes three main modules:
1. **Data Collection** → `signDataset.py`  
2. **Model Training** → `training.ipynb`  
3. **Live Detection** → `liveDetect.py`

---

## 📂 Project Structure

```
├── Gestures/ # Folder containing collected gesture images
├── pycache/ # Python cache files
├── GestureModel.pt # Saved PyTorch model
├── GestureModel.py # Model architecture definition
├── best_gesture_model.pth # Best trained model weights
├── liveDetect.py # Real-time gesture detection & control
├── signDataset.py # Script for gesture dataset collection
├── training.ipynb # Jupyter Notebook for training the model
├── tempCodeRunnerFile.py # Temporary runner file (can be ignored)
└── README.md # Project documentation
```

---

## 🚀 Features
- **Collect custom gesture datasets** using your webcam.  
- **Train deep learning models** for hand sign recognition.  
- **Perform real-time detection** and control a DJI drone.  

---

## ⚙️ Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/your-username/Gesture-Controlled-DJI-Model.git
cd Gesture-Controlled-DJI-Model
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Connect your DJI Drone**  
Ensure your drone SDK/API is properly configured.  

---

## 📊 Workflow

### 1. Data Collection (`signDataset.py`)
Run the script to capture hand gesture images from your webcam.  
Data is saved in the `Gestures/` folder.  

```bash
python signDataset.py
```

---

### 2. Model Training (`training.ipynb`)
The model was trained using the collected gesture dataset.  
Training ran for ~60 epochs with the following results:

- **Final Training Accuracy:** ~99%  
- **Final Test Accuracy:** ~99%  
- **Loss steadily decreased** with no major signs of overfitting.  

#### Training Curves
Below are the training curves for **Loss** and **Accuracy**:

<img width="1222" height="624" alt="image" src="https://github.com/user-attachments/assets/655b7b2e-e82c-482e-8c59-e878d1fccf5b" />

---

### 3. Live Gesture Detection (`liveDetect.py`)
Run the live detection script to recognize gestures in real-time.  
The drone responds to recognized hand signs.  

```bash
python liveDetect.py
```

✅ Example Gestures  

- 👆 → Up / TakeOff  
- 👎 → Down / Land  
- ✊ → Come Forward  
- ✋ → Stop  
- 👉 → Left  
- 🫲 → Right  
- 🤘 → Turn Backward  

(Modify gesture mappings as needed in `liveDetect.py`)

---

## ⚙️ Steps to Implement Gesture Control on Tello

### 1. Install DJI Tello Python Library
```bash
pip install djitellopy
```

### 2. Connect to the Tello Drone
- Power on your Tello drone.  
- Connect your computer to Tello’s Wi-Fi (SSID usually `TELLO-xxxxxx`).  
- Test connection with the following script:  

```python
from djitellopy import Tello

tello = Tello()
tello.connect()

print(f"Battery: {tello.get_battery()}%")
```

If everything is working, you should see the current battery percentage of the drone.  

---

### 3. Integrate with Your Gesture Detection
In `liveDetect.py`, after detecting a gesture, map it to drone commands.  

---

## 📌 Future Improvements
- Add more robust gesture datasets.  
- Improve accuracy with transfer learning.  
- Extend support for additional drone commands.  

---

## 📝 License
This project is licensed under the MIT License - feel free to use and modify.  

---

## 🤝 Contributing
Pull requests are welcome! For major changes, open an issue first to discuss what you’d like to change.  
