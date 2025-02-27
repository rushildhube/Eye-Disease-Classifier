# 👁️ Eye Disease Classifier

## 🚀 Overview
Ever wondered if AI could help detect eye diseases just from an image? That's exactly what **Eye Disease Classifier** does! This web app, built using **Streamlit**, allows you to upload a retinal image and get a quick, AI-powered diagnosis. By combining the strengths of two deep learning models and a meta-classifier, it provides accurate predictions across multiple eye diseases.

## 🏆 Features
- 🔍 **Smart Disease Detection**: Upload an image, and the AI identifies potential eye diseases.
- ⚡ **Dual Deep Learning Models**: Uses two powerful CNN models (`model2.h5` and `model1.keras`).
- 🤖 **Meta-Classifier Magic**: Improves accuracy by analyzing predictions from both models.
- 🎨 **User-Friendly Interface**: A clean and interactive UI built with Streamlit.
- 📊 **Detects Multiple Diseases**: Covers **Cataract, Diabetic Retinopathy, Glaucoma, Macular Degeneration, and Normal** eye conditions.
- 📂 **Runs on Your Device**: No need for a high-end GPU – works efficiently on local machines.

## 🖥️ Demo
![Eye Disease Classifier]([https://via.placeholder.com/800x400?text=Demo+Image](https://eye-disease-classifier.streamlit.app/))  
👉 **[Live Demo](#)** (If deployed, add the link here)

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/rushildhube/Eye-Disease-Classifier.git
cd Eye-Disease-Classifier
```

### 2️⃣ Install Dependencies
Make sure you have Python installed (preferably **Python 3.8+**).
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit App
```sh
streamlit run app.py
```

## 📂 Project Structure
```
📁 Eye-Disease-Classifier
│-- app.py                 # Main Streamlit App
│-- model2.h5              # Deep Learning Model 1 (224x224)
│-- model1.keras           # Deep Learning Model 2 (256x256)
│-- meta_model.pkl         # Meta-classifier (ML Model)
│-- requirements.txt       # Dependencies
│-- .gitattributes         # Git LFS tracking
└── README.md              # Project Documentation
```

## 📖 How It Works
1. **Upload a Retinal Image**
2. **Image Preprocessing**
   - Resized to match the expected input sizes for both models
   - Converted to an array and normalized
3. **Model Predictions**
   - `model2.h5` makes a prediction using a (224, 224) image input
   - `model1.keras` makes a prediction using a (256, 256) image input
4. **Meta-Classifier in Action**
   - Combines predictions from both models
   - Uses `meta_model.pkl` to decide the final diagnosis
5. **Results Are Displayed on the UI** 🎯

## 🏥 Supported Eye Diseases
The classifier can identify the following conditions:

1. **Cataract** 🏥 - A cloudy lens causing blurred vision.
2. **Diabetic Retinopathy** 🩸 - Retinal damage due to high blood sugar levels.
3. **Glaucoma** 👁️ - Increased eye pressure leading to optic nerve damage.
4. **Macular Degeneration** 🔬 - A disease affecting the central part of the retina.
5. **Normal** ✅ - No visible signs of disease detected.

## 📌 Technologies Used
- **Python** 🐍 - Core programming language.
- **TensorFlow/Keras** 🤖 - Deep learning framework for model training.
- **Streamlit** 🖥️ - Web app framework for easy UI development.
- **NumPy & OpenCV** 🔢 - Image processing and numerical operations.
- **Pickle** 🏗️ - Used for storing the meta-classifier.
- **PIL (Pillow)** 📷 - Image handling and preprocessing.

## 🛠️ Troubleshooting
- **CUDA Issues?** If your system has trouble with GPU acceleration, disable it by setting:
  ```sh
  export CUDA_VISIBLE_DEVICES=-1  # Linux/macOS
  set CUDA_VISIBLE_DEVICES=-1      # Windows (CMD)
  ```
- **File Not Found?** Ensure that the models are placed correctly in the project directory.
- **Slow Predictions?** Try reducing the image size before uploading.

## 🤝 Contributing
We’d love for you to contribute! Here’s how you can help:
- 📌 Report bugs or issues
- 🚀 Suggest improvements or optimizations
- 🎨 Enhance the UI
- 📜 Improve documentation

### Steps to Contribute:
1. **Fork the repository** 🍴
2. **Create a new branch** `git checkout -b feature-branch`
3. **Make your changes** and commit `git commit -m "Added a new feature"`
4. **Push to your fork** `git push origin feature-branch`
5. **Submit a Pull Request** 🚀

## 📜 License
This project is open-source and licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 📬 Contact
👤 **Rushil Dhube**  

---

