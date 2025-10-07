# ML Visualization Tool  

A web-based interactive application for visualizing how different **Machine Learning algorithms** perform on various datasets with customizable hyperparameters. The tool allows users to explore decision boundaries, compare models, and experiment with real-world datasets.  

## 🚀 Features  
- Interactive **Flask web app**  
- Supports multiple ML algorithms (classification & regression)  
- Hyperparameter tuning with instant visualization  
- Built-in test datasets and support for custom data  
- Decision boundary plots for classification tasks  
- Performance metrics and error visualization  
- Simple and responsive UI  

## 🛠️ Tech Stack  
- **Backend**: Python, Flask  
- **Machine Learning**: Scikit-learn, XGBoost, CatBoost, LightGBM  
- **Visualization**: Matplotlib, Seaborn  
- **Frontend**: HTML (Jinja2 templates), CSS  
- **Others**: NumPy, Pandas  

## 📂 Project Structure  
ML-Visualization-Tool/  
│── app.py                # Main Flask application  
│── ml_tool.py            # Core ML logic and visualization functions  
│── requirements.txt      # Python dependencies  
│── templates/            # HTML templates (Flask)  
│   └── index.html  
│── static/               # Static assets  
│   └── plots/            # Saved plots  
│── catboost_info/        # CatBoost training logs  

## ⚡ Installation & Usage  

### 1️⃣ Clone the repository  
git clone https://github.com/your-username/ML-Visualization-Tool.git  
cd ML-Visualization-Tool  

### 2️⃣ Create a virtual environment & install dependencies  
python -m venv venv  
source venv/bin/activate   # On Linux/Mac  
venv\Scripts\activate      # On Windows  

pip install -r requirements.txt  

### 3️⃣ Run the Flask app  
python app.py  

The app will start running at **http://127.0.0.1:5000/**  

## 📸 Example Outputs  
- Decision boundary plots for classifiers  
- Regression line fits  
- Accuracy and error visualizations  

## 📌 Future Enhancements  
- Add support for deep learning models (TensorFlow, PyTorch)  
- Upload custom CSV datasets  
- Export trained models and results  

## 🤝 Contributing  
Pull requests are welcome! Please open an issue first to discuss changes.  

## 📜 License  
This project is licensed under the MIT License.  
