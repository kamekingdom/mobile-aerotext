# mobile-aerotext

**mobile-aerotext** is a lightweight character recognition and transformation project using a dataset of handwritten Hiragana character images (256x256 pixels, 1000 images).

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
├── infer/
│   └── infer.py        # Inference Program
├── model/
│   └── weights/        # Model weights (included in version control)
├── test_images/        # Test images (excluded from version control)
├── scripts/            # Scripts for processing and training (optional)
├── README.md
└── .gitignore
```

## Development Environment

- Python ≥ 3.8
- PyTorch / TensorFlow (depending on the model implementation)
- NumPy / OpenCV / Matplotlib (for preprocessing and visualization)

## License & Usage

The dataset and code are intended for academic use only. For commercial applications, please contact the project maintainer.

---

Contributions, feedback, and discussions are welcome.
