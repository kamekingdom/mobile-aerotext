# mobile-aerotext

**mobile-aerotext** is a lightweight character recognition and transformation project using a dataset of handwritten Hiragana character images (256x256 pixels, 1000 images).

## 🔗 Dataset

The Hiragana Train Dataset (256x256, 1000 images) is available via the following link:

[Google Drive Link](https://drive.google.com/file/d/1oKIFJD2T-bTw3Y_d1SU3SvIjHKCjCq6v/view?usp=drive_link)

- Format: PNG
- Size: 256x256 pixels
- Number of images: 1000
- Content: Pre-processed handwritten-style Hiragana character images

## 📁 Directory Structure (Partial)

```
mobile-aerotext/
├── model/
│   └── weights/        # Model weights (included in version control)
├── images/             # Training/auxiliary images (excluded from version control)
├── scripts/            # Scripts for processing and training (optional)
├── README.md
└── .gitignore
```

## 🚫 Notes

- The `images/` folder is excluded from Git tracking via `.gitignore`.
- All contents under `model/` are ignored **except** for `model/weights/`.
- The weight files are provided to facilitate inference and reproducibility.

## 🛠 Development Environment

- Python ≥ 3.8
- PyTorch / TensorFlow (depending on the model implementation)
- NumPy / OpenCV / Matplotlib (for preprocessing and visualization)

A `requirements.txt` or `environment.yml` will be provided (or should be added) to specify dependencies.

## 📄 License & Usage

The dataset and code are intended for academic use only. For commercial applications, please contact the project maintainer.

---

Contributions, feedback, and discussions are welcome.
