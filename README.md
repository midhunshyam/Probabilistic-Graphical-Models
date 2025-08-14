# Conditional Generation of Kuzushiji Characters using CVAE and CGAN

This project implements **Conditional Variational Autoencoders (CVAE)** and **Conditional Generative Adversarial Networks (CGAN)** to generate images of classical Japanese handwritten characters from the **Kuzushiji-49** dataset. The models are conditioned on both **character class** and **writing style** (thick or thin).

> NOTE: The source code was accidentally deleted. This repository currently serves as documentation with the code for each section and reference for potential reconstruction.

## Project Overview

- **Course**: MATH 7017 – Probabilistic Graphical Models  
- **Author**: Midhun Shyam  
- **Institution**: Western Sydney University  
- **Supervisors**: Prof. Oliver Obst & Stuart Fitzpatrick

This applied project explores the use of CVAEs and CGANs for controlled image generation based on categorical labels. It includes architectural design, training methodology, and comparative evaluation.


## Dataset: Kuzushiji-49 (Kaggle)

- 270,912 grayscale images of 49 different Japanese cursive characters  
- Each image: 28×28 pixels  
- Labels:
  - **Class**: One of 49 characters
  - **Style**: Binary classification (Thick / Thin — derived via pixel density threshold)

**Source**: [Kaggle – Kuzushiji-MNIST Dataset]([https://www.kaggle.com/datasets/rois-codh/kuzushiji](https://www.kaggle.com/datasets/anokas/kuzushiji))



##  Model Architectures

### CVAE

- Inputs: Flattened image (784), class one-hot (49), style one-hot (2)
- Latent Space: 2D
- Loss: Binary Cross-Entropy + KL Divergence
- Output: Reconstructed image conditioned on class and style

### CGAN

- Generator Input: Noise + class one-hot + style one-hot  
- Discriminator Input: Image + class + style  
- Loss: Binary Cross-Entropy (GAN loss)  
- Output: New image conditioned on class and style



## Experiments

- **Preprocessing**:
  - Normalisation to [0, 1] for CVAE
  - Normalisation to [–1, 1] for CGAN (Tanh)
  - Foreground pixel count used to derive "style" labels

- **Model Variants**:
  - CVAE: Alternate input combinations, hidden layer depth
  - CGAN: Tanh vs. Sigmoid outputs, activation variants
  - Style-controlled conditional generation

## Results Summary

| Model | Output Sharpness | Style Control | Class Control | Training Time |
|-------|------------------|---------------|----------------|----------------|
| CVAE  | Moderate (blurry) | ✅ Yes        | ✅ Yes         | ⏱️ Fast        |
| CGAN  | High (sharper)   | ✅ Yes        | ✅ Yes         | ⏱️ Longer      |

- CVAE outputs lacked fine detail
- CGAN significantly improved over epochs (notably after 1000+)
- Best fidelity achieved with CGAN Architecture 2 (Sigmoid activation + [0, 1] inputs)


## Visual Examples

Generated examples (from saved report):

- Class 0 (Thin vs Thick)
- Class 10 (Thin vs Thick)
- CVAE vs CGAN training outputs
- Loss plots and reconstructions

> 📄 See `PGM_Report.pdf` for visuals and extended documentation


## Challenges

- Image clarity in CVAE limited by latent bottleneck  
- Long training time for CGAN to reach quality generation  
- GAN training instability (loss oscillations)  


## Future Work

- Implement Wasserstein or Diffusion models for stability and quality  
- Reconstruct architecture from scratch using this README and report  
- Use higher-dimensional latent spaces in CVAE  
- Apply transfer learning to multi-language or ancient script datasets  


## Code Restoration Plan

To recover this project:

1. Recreate preprocessing pipeline (`.npz` loading, normalisation, label encoding)
2. Rebuild CVAE & CGAN using PyTorch (reference architecture in report)
3. Use saved model weights (`.pth`) if available
4. Retrain and re-plot loss curves + generated examples
