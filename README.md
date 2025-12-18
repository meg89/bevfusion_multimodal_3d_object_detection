# BEVFusion-Based Multimodal 3D Object Detection  
**Camera + LiDAR + Radar Fusion (CPU-Friendly Implementation)**

This repository contains a lightweight, CPU-friendly implementation of a **BEVFusion-inspired multimodal 3D object detection pipeline**, developed as part of an academic project.  
The system fuses **camera, LiDAR, and radar** sensor data into a unified **Bird’s-Eye View (BEV)** representation and performs detection using a **CenterNet-style detection head**.

The project is evaluated on the **nuScenes v1.0 mini dataset** using standard metrics such as **mAP** and **NDS**.

📄 **Project Report (ACM-style):**  
See the accompanying project report for detailed methodology, experiments, and analysis.

🔗 **Reference Repository:**  
https://github.com/meg89/bevfusion_multimodal_3d_object_detection

---

## 🚀 Features

- Multimodal sensor fusion: **Camera, LiDAR, Radar**
- Unified **BEV representation**
- **ResNet-18** camera encoder
- **PointNet-based** LiDAR and radar encoders
- **BEVFusion-style feature projection**
- **CenterNet-based detection head**
- CPU-friendly (no custom CUDA ops)
- Modular design for easy experimentation with sensor combinations

---

## Follow run_instructions.txt for instruction to reproduce the code.
# Follow demo.ipynb to run the code and see the results

## 📁 Repository Structure

```
.
├── configs/
│   └── base.yaml
├── data/
│   └── nuscenes/
├── checkpoints/
│   └── best_model.pth
├── encoders.py
├── fusion.py
├── train_detect.py
├── eval.py
├── inference.py
├── data_converter.py
├── data_validate.py
├── validate_data_with_samples.py
├── requirements.txt
└── README.md
```


---

## ⚙️ Environment Setup

### 1️⃣ Install Requirements

```bash
python >= 3.10
pip install -r requirements.txt
```

---

## 📦 Dataset Preparation (nuScenes-mini)

### 2️⃣ Download Dataset

1. Register at https://www.nuscenes.org/nuscenes#download  
2. Download **nuScenes v1.0 mini**  
3. Extract the dataset into a `data/` directory  

---

### 3️⃣ Configure Dataset Path

Edit `configs/base.yaml` and set:
```yaml
dataset:
  root_path: /absolute/path/to/data/nuscenes
```

---

### 4️⃣ Convert Dataset

```bash
python data_converter.py --config configs/base.yaml --split train
python data_converter.py --config configs/base.yaml --split val
python data_converter.py --config configs/base.yaml --split test
```

---

### 5️⃣ Validate Dataset Conversion

```bash
python data_converter.py --config configs/base.yaml --show-config
python data_validate.py --config configs/base.yaml --show-config
```

---

### 6️⃣ Inspect Dataset Samples (Optional)

```bash
python validate_data_with_samples.py --config configs/base.yaml
python validate_data_with_samples.py --samples 10
python validate_data_with_samples.py --split train --samples 5
```

---

## 🏋️ Training Pipeline

### 7️⃣ Sanity Check Encoders and Fusion

```bash
python encoders.py
python fusion.py
```

---

### 8️⃣ Train the Model

```bash
python train_detect.py train configs/base.yaml
```

---

## 📊 Evaluation

```bash
python eval.py configs/base.yaml
```

---

## 🔍 Inference

```bash
python inference.py --model checkpoints/best_model.pth
```

---

## 🧪 Experiments & Modality Configurations

| Modality | Fusion | Detection Head |
|--------|--------|----------------|
| camera_only | BEVFusion | CenterNet |
| lidar_only | BEVFusion | CenterNet |
| camera + lidar | BEVFusion | CenterNet |
| camera + lidar + radar | BEVFusion | CenterNet |

---

## 👤 Author

**Megha Agrawal**  
Rutgers University  

---

## 📜 License

Academic and research use only.
