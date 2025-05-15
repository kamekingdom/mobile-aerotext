# mobile-aerotext

**mobile-aerotext** is a lightweight character recognition and transformation project using a dataset of handwritten Hiragana character images (256x256 pixels, 1000 images).
[![Watch on YouTube](https://img.youtube.com/vi/zxwabglNMT0/0.jpg)](https://youtu.be/zxwabglNMT0?si=ltFUmZbv-KYWacYm)

## Dataset

The Hiragana Train Dataset (256x256, 46000 images) is available via the following link:

[train_images Google Drive Link](https://drive.google.com/file/d/1oKIFJD2T-bTw3Y_d1SU3SvIjHKCjCq6v/view?usp=drive_link)

- Format: PNG
- Size: 256x256 pixels
- Number of images: 46000
- Content: Pre-processed handwritten-style Hiragana character images

The Hiragana Train Dataset (256x256, 1600 images) is available via the following link:

[test_images Google Drive Link](https://drive.google.com/file/d/1A-qv_GlGpE46OKY9vRzmBpvBK3dWHcFS/view?usp=drive_link)

- Format: PNG
- Size: 256x256 pixels
- Number of images: 1600
- Content: Pre-processed handwritten-style Hiragana character images

## Directory Structure (Partial)

```
mobile-aerotext/
├── collector/
│   ├── how-to-use.txt  # Instruction and Description
│   └── run.bat         # Data Collection Program
├── model/
│   └── weights/best.pt # Best Model weights (included in version control)
├── train/
│   └── train.py        # Training Program
├── infer/
│   ├── result          # Inference Result Data
│   └── infer.py        # Inference Test Program
├── test_images/        # Test images for Inference (excluded from version control)
├── scripts/            # Scripts for processing and training (optional)
├── README.md
└── .gitignore
```

## Development Environment

- Python ≥ 3.8
- PyTorch / TensorFlow (depending on the model implementation)
- NumPy / OpenCV / Matplotlib (for preprocessing and visualization)

## License & Usage

- _CC BY 4.0 license_: The dataset and code are intended for academic use only. For commercial applications, please contact the project maintainer.

---

Contributions, feedback, and discussions are welcome.
