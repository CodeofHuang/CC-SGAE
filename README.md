# CC-SGAE

**Cross-Cycle Structured Graph Autoencoder for Unsupervised Cross-Sensor Image Change Detection.**

## 📋 Environment Requirements

The code has been tested in the following environment. We recommend using these specific versions for reproducibility:

* python==3.12.2
* numpy==1.26.4
* torch==2.3.0
* torch_geometric==2.5.0
* scikit-learn==1.4.1
* scikit-image==0.22.0
* opencv-python==4.9.0
* imageio==2.34.0
* scipy==1.12.0

You can install the dependencies using pip:
```bash
pip install numpy scikit-learn scikit-image opencv-python imageio scipy torch_geometric
# Note: Please ensure PyTorch is installed according to your CUDA version.
```

## 📂 Project Structure

This repository currently contains the core implementation of the proposed method:

* `Networks.py`: Implementation of the Cross-Cycle Structured Graph Autoencoder.
* `utils.py`: Utility functions.
* `data_loader.py`: Data loading and preprocessing logic for cross-sensor datasets.

## 🔗 Related Datasets and Supporting Algorithms

The datasets used in this work are publicly available from the following sources:

* **Dataset #2 & Dataset #4**: Download from Professor Max Mignotte's webpage: http://www-labs.iro.umontreal.ca/~mignotte/
* **Dataset #3**: Download from Dr. Han's GitHub repository: https://github.com/rshante0426/MCD-datasets

The following repositories are related to the graph-based structural consistency and change alignment mechanisms. Great thanks to the authors for their excellent works:

* **SCASC**: https://github.com/yulisun/SCASC
* **SRF**: https://github.com/yulisun/SRF

## 📧 Contact

If you have any queries, please do not hesitate to contact us at: dearhyk@126.com
