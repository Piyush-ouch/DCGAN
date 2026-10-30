# 👕 DCGAN on Fashion-MNIST | Deep Convolutional Generative Adversarial Network

A Deep Convolutional Generative Adversarial Network (DCGAN) implemented using TensorFlow/Keras to generate realistic synthetic fashion images from random noise.

The project demonstrates how Generative Adversarial Networks (GANs) learn the underlying distribution of real-world images through adversarial training between two neural networks: a **Generator** and a **Discriminator**.

## 📌 Project Overview

This project implements a DCGAN trained on the **Fashion-MNIST dataset**, consisting of 60,000 grayscale training images of fashion products across 10 different categories.

The Generator learns to create synthetic clothing images from random latent vectors, while the Discriminator learns to distinguish between real Fashion-MNIST images and artificially generated images.

The two networks compete against each other, progressively improving the quality of generated images.

## 🎯 Objectives

- Understand the fundamentals of Generative Adversarial Networks.
- Implement Generator and Discriminator architectures using CNNs.
- Train a DCGAN on the Fashion-MNIST dataset.
- Generate synthetic fashion images from random noise.
- Analyze Generator and Discriminator loss curves.
- Evaluate generated images using visual comparisons.
- Explore latent space interpolation and image diversity.

## 🧠 Model Architecture

The DCGAN consists of two competing neural networks.

### 1. Generator

The Generator transforms a random latent vector into a synthetic 28 × 28 grayscale fashion image.

**Architecture:**
- Input: 100-dimensional latent vector.
- Dense layer: 128 × 7 × 7 neurons.
- Reshape: 7 × 7 × 128.
- Transposed Convolution: Upsamples to 14 × 14.
- Transposed Convolution: Upsamples to 28 × 28.
- Output: 28 × 28 × 1 grayscale image.
- Activation: Tanh.

### 2. Discriminator

The Discriminator determines whether an input image is real or generated.

**Architecture:**
- Input: 28 × 28 × 1 grayscale image.
- Convolutional Layer: 64 filters, 3 × 3 kernel.
- LeakyReLU activation.
- Dropout: 0.4.
- Convolutional Layer: 64 filters, 3 × 3 kernel.
- LeakyReLU activation.
- Dropout: 0.4.
- Flatten layer.
- Dense output layer with Sigmoid activation.

### 3. Combined GAN

The Generator and Discriminator are connected to form a combined GAN model.

During Generator training, the Discriminator's weights are frozen. The Generator learns to produce images that can fool the Discriminator into classifying synthetic images as real.

## 📊 Dataset

**Fashion-MNIST**

| Property | Description |
|---|---|
| Dataset | Fashion-MNIST |
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Dimensions | 28 × 28 |
| Color Channels | 1 (Grayscale) |
| Number of Classes | 10 |
| Pixel Range | 0–255 |
| Training Range | -1 to 1 |

### Fashion Categories

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

Dataset is loaded directly using Keras:

```python
from keras.datasets.fashion_mnist import load_data

(trainX, trainy), (testX, testy) = load_data()
```

## ⚙️ Technologies Used

- **Python** – Programming language
- **TensorFlow / Keras** – Deep learning framework
- **NumPy** – Numerical computations
- **Matplotlib** – Image visualization and plotting
- **Pandas** – Loss smoothing and analysis
- **Pydot** – Model architecture visualization
- **Fashion-MNIST** – Training dataset
- **Google Colab / Jupyter Notebook** – Development environment

## 🔄 Training Configuration

| Hyperparameter | Value |
|---|---|
| Latent Dimension | 100 |
| Batch Size | 128 |
| Training Epochs | 5 (current notebook configuration) |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Beta 1 | 0.5 |
| Loss Function | Binary Crossentropy |
| Generator Output Activation | Tanh |
| Discriminator Output Activation | Sigmoid |
| Image Size | 28 × 28 |

The training loop alternates between updating the Discriminator using real and generated images and updating the Generator through the combined GAN model.

## 📈 Evaluation and Visualization

The project includes multiple evaluation techniques:

### 1. Generator and Discriminator Loss Curves
Tracks the adversarial training process and analyzes the relationship between Generator and Discriminator losses.

### 2. Discriminator Accuracy
Evaluates the Discriminator's ability to distinguish between real and generated images.

### 3. Synthetic Image Generation
Generates a 7 × 7 grid of 49 synthetic fashion images using random latent vectors.

### 4. Real vs Generated Comparison
Displays real Fashion-MNIST images alongside synthetic images produced by the trained Generator.

### 5. Latent Space Interpolation
Interpolates between two random latent vectors to visualize how generated images transition through the learned latent space.

## 📂 Project Structure

```text
DCGAN-FashionMNIST/
│
├── DCGAN_FashionMNIST.ipynb
│
├── generated_plot_eXXX.png
├── final_generated_images.png
├── loss_curves.png
├── discriminator_accuracy.png
│
├── generator_model_XXX.h5
│
└── README.md
```

*Note: Generated images and model checkpoints are created when the corresponding notebook cells execute. The checkpoint function saves models every 10 epochs.*

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/DCGAN-FashionMNIST.git
```

### 2. Navigate to the Project Directory

```bash
cd DCGAN-FashionMNIST
```

### 3. Install Dependencies

```bash
pip install tensorflow numpy matplotlib pandas pydot
```

### 4. Run the Notebook

Open the notebook using Jupyter:

```bash
jupyter notebook DCGAN_FashionMNIST.ipynb
```

Alternatively, upload the notebook to Google Colab and execute the cells sequentially.

### 5. Train the Model

The current notebook uses:

```python
train(
    g_model,
    d_model,
    gan_model,
    dataset,
    latent_dim,
    n_epochs=5,
    n_batch=128
)
```

Increase `n_epochs` to train for more iterations and potentially improve image quality.

## 🔍 Key Learnings

- Understanding adversarial training and GAN fundamentals.
- Designing CNN-based Generator and Discriminator networks.
- Working with latent space representations.
- Understanding Binary Crossentropy loss.
- Training deep learning models using TensorFlow/Keras.
- Evaluating GAN performance through loss curves and generated samples.
- Understanding mode collapse and training stability.
- Exploring latent space interpolation.

## 🔮 Future Improvements

- Train the model for more epochs to improve image quality.
- Experiment with different Generator and Discriminator architectures.
- Implement Wasserstein GAN (WGAN).
- Implement Conditional GAN (CGAN) to generate specific fashion categories.
- Experiment with different latent dimensions and batch sizes.
- Explore quantitative evaluation using FID scores.

## 👨‍💻 Author

**Piyush Jangade**

B.Tech – Computer Science and Engineering (AI & ML)

GitHub: [@Piyush-ouch](https://github.com/Piyush-ouch)

Portfolio: [Visit Portfolio](https://piyush-ouch.github.io/portfolio1/)

---

⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, TensorFlow, and Deep Learning.**
