# Gesture-Controlled DJI Model 🚁✋

This project implements a **gesture-controlled system** to interact with a DJI drone using hand signs.  
It includes three main modules:
1. **Data Collection** → `signDataset.py`
2. **Model Training** → `training.ipynb`
3. **Live Detection** → `liveDetect.py`

---

## 📂 Project Structure

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

yaml
Copy
Edit

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
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
(Make sure to include packages like torch, opencv-python, numpy, etc.)

Connect your DJI Drone
Ensure your drone SDK/API is properly configured.

📊 Workflow
1. Data Collection (signDataset.py)
Run the script to capture hand gesture images from your webcam.

Data is saved in the Gestures/ folder.

bash
Copy
Edit
python signDataset.py
2. Model Training (training.ipynb)
Open the Jupyter Notebook.

Train the model with your collected dataset.

The best model weights are saved as best_gesture_model.pth.

3. Live Gesture Detection (liveDetect.py)
Run the live detection script to recognize gestures in real-time.

The drone responds to recognized hand signs.

bash
Copy
Edit
python liveDetect.py
✅ Example Gestures
✊ Fist → Takeoff

✋ Open Palm → Land

👆 Pointing → Move Forward

👉 Right Gesture → Move Right

👈 Left Gesture → Move Left

(Modify gesture mappings as needed in liveDetect.py)

📌 Future Improvements
Add more robust gesture datasets.

Improve accuracy with transfer learning.

Extend support for additional drone commands.

📝 License
This project is licensed under the MIT License - feel free to use and modify.

🤝 Contributing
Pull requests are welcome! For major changes, open an issue first to discuss what you’d like to change.
