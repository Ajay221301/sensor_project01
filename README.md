# 📄✏ Sensor Fault Detection Project
**Brief:** In electronics, a **wafer** (also called a slice or substrate) is a thin slice of semiconductor, such as a crystalline silicon (c-Si), used for the fabrication of integrated circuits and, in photovoltaics, to manufacture solar cells. The wafer serves as the substrate(serves as foundation for contruction of other components) for microelectronic devices built in and upon the wafer. 

It undergoes many microfabrication processes, such as doping, ion implantation, etching, thin-film deposition of various materials, and photolithographic patterning. Finally, the individual microcircuits are separated by wafer dicing and packaged as an integrated circuit.

#### Dataset is taken from Kaggle and stored in mongodb

<img width="2164" height="1356" alt="image" src="https://github.com/user-attachments/assets/e3920c76-3aa0-4e14-bbb9-b75242eb39c6" />


### ⚡ Automation & Deployment
- Fully automated **CI/CD pipeline** using **GitHub Actions** for testing, building, and deployment.  
- Uses **Docker** for containerization to ensure consistent runtime environments.  
- Deployed automatically to **AWS EC2** using images pushed to **Amazon ECR**.  
- Sensitive credentials (e.g., MongoDB URL, AWS keys) are securely managed via **GitHub Secrets**.



💿 Installing
1. Environment setup.
```
conda create --prefix venv python==3.8 -y
```
```
conda activate venv/
````
2. Install Requirements and setup
```
pip install -r requirements.txt
```
5. Run Application
```
python app.py
```

🔧 Built with
- flask
- Python 3.8
- Machine learning
- Scikit learn
- 🏦 Industrial Use Cases

🏭 Industrial Relevance

⚡ Enables real-time wafer fault detection in semiconductor manufacturing processes.

🔧 Reduces manual inspection time and overall production costs, improving efficiency.

🤖 Implements a fully automated ML lifecycle — from data ingestion → preprocessing → model training → evaluation → prediction → deployment.

🌐 Scalable and adaptable for various industrial quality-control systems beyond semiconductors (e.g., automotive sensors, solar cells, electronic components).
