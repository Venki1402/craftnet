
# 🚀 CraftNet

**CraftNet** is a deep learning-based image classification app built with **PyTorch**, using a fine-tuned **ResNet-18** model. It identifies vehicles across three modes of transport — **airplanes**, **cars**, and **ships** — through a clean and interactive **Streamlit** web interface.

## 🖼️ Demo

Upload an image and get instant classification results like:
<img width="627" alt="Screenshot 2025-06-15 at 6 58 18 AM" src="https://github.com/user-attachments/assets/7d9a6eac-e4ea-4233-bbd6-f05622ec78a2" />
<img width="615" alt="Screenshot 2025-06-15 at 6 58 40 AM" src="https://github.com/user-attachments/assets/887ade9e-a58d-4920-ae52-995e285b9d61" />
<img width="619" alt="Screenshot 2025-06-15 at 6 59 44 AM" src="https://github.com/user-attachments/assets/57533b06-e15d-4115-a3df-9440d461a41b" />


## 🎥 Video


https://github.com/user-attachments/assets/22789f7c-220e-4589-bb34-605a10d297c1




## 📁 Project Structure

```
CraftNet/
├── app.py                              # Streamlit app for inference
├── train.py                            # Script to train the ResNet18 model
├── dataset/
│   ├── train/
│   │   ├── airplane/
│   │   ├── car/
│   │   └── ship/
│   └── test/
│       ├── airplane/
│       ├── car/
│       └── ship/
├── vehicle\_classification\_model.pth  # Trained model weights
├── class\_names.txt                    # Class names (one per line)
|── requirements.txt
└── README.md

```

## ⚙️ Features

- 🧠 Built on top of **ResNet-18**
- 🖥️ Streamlit-powered Web UI
- 🔁 Easily extendable for other categories
- 🗃️ Simple directory-based dataset structure
- 💾 Trained weights export for reuse

## 🚀 Getting Started

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt
````

### 🏃 Run the App

```bash
streamlit run app.py
```

Make sure `vehicle_classification_model.pth` and `class_names.txt` are in the same directory.

## 🏋️‍♂️ Train Your Own Classifier

### 1. 🗂️ Organize Your Dataset

Structure your dataset like this:

```
dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

Each subfolder must contain images of that class.

### 2. ⚙️ Modify Training Script

In `train.py`, set the `data_dir` path:

```python
data_dir = 'dataset'  # update this if using a different folder
```

The number of classes is automatically determined from the subfolder names.

### 3. 🧠 Run Training

```bash
python train.py
```

This will:

* Fine-tune ResNet-18 on your dataset
* Save the model as `vehicle_classification_model.pth`
* Save class labels to `class_names.txt`

### 4. 🚀 Run App with New Model

Once training is done, simply run the app:

```bash
streamlit run app.py
```

Make sure `vehicle_classification_model.pth` and `class_names.txt` are updated with your new model and class list.

## 📌 Notes

* Model uses **transfer learning** — only the final layer is trained.
* Normalization is done using ImageNet mean & std for compatibility with pretrained ResNet.
* GPU will be used automatically if available.

## 📜 License

MIT License
