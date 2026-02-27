# Computer Vision Research Showcase

This repository showcases a series of computer vision research experiments, culminating in the development of a Residual U-Net (ResUnet) for image processing tasks. The project documents the progression from baseline U-Net architectures to more advanced models, including experiments with different datasets and noise models.

## Project Structure

The repository is organized into the following directories:

- `src/`: Contains all the Python source code.
  - `architectures/`: Defines the neural network models (U-Net, ResUnet).
  - `data/`: Includes the data loading and processing scripts.
  - `training/`: Contains the scripts for training the models on different datasets (MNIST, BSDS).
  - `utils/`: Provides utility functions, including noise simulation and Quanta Image Sensor (QIS) modeling.
- `weights/`: Stores the pre-trained model weights.
- `archive/`: Contains older scripts, initial experiments, and other miscellaneous files.

## Research Progression

My research involved several stages:

### 1. U-Net for Denoising

I began by implementing a standard U-Net architecture to explore its effectiveness for image denoising tasks. The initial models were tested on the MNIST and BSDS datasets.

- **U-Net Architecture**: `src/architectures/UNET.py`
- **Training Scripts**: `src/training/train_mnist.py`, `src/training/train_bsds.py`

### 2. Quanta Image Sensor (QIS) Simulation

To model more realistic sensor noise, I developed a simulation for a Quanta Image Sensor (QIS). This allowed for more challenging and practical denoising experiments.

- **QIS Utilities**: `src/utils/qis_utils.py`
- **Noise Modeling**: `src/utils/Noise.py`

### 3. Residual U-Net (ResUnet)

Based on the initial findings, I moved to a more advanced Residual U-Net (ResUnet) architecture. The residual connections help to improve gradient flow and training stability, leading to better performance on the denoising tasks.

- **ResUnet Architecture**: `src/architectures/ResUnet.py`

## How to Run

1.  **Explore the architectures** in the `src/architectures/` directory to see the U-Net and ResUnet implementations.
2.  **Examine the training scripts** in `src/training/` to understand how the models were trained.
3.  **Load a pre-trained model** from the `weights/` directory and use the scripts in `archive/scripts` as a reference for running inference.

## Research Poster

This poster summarizes the research conducted on denoising images from single-photon detectors, a key aspect of the QIS simulation work. It provides a visual overview of the problem, methodology, and key findings.

[View the Research Poster](https://cvgi-website.vercel.app/2025/research/single-photon-detector/)

This repository reflects my journey through different deep learning architectures for computer vision, demonstrating an iterative and research-driven approach to model development.
