# Reflection Removal using Diffusion Model

This repository implements a diffusion-based deep learning model for reflection removal from images. The model uses a progressive denoising approach to separate reflection artifacts from the main image content.

## Model Architecture

### Overview
The model consists of three main components:
1. **UNet Backbone**
   - Initial hidden dimension: 96
   - Progressive channel expansion: 96 → 192 → 384 → 768
   - Multi-stage processing with skip connections
   
2. **Stage Blocks**
   - **Encoder Path**:
     - Stage 0: 96 channels, spatial downsampling
     - Stage 1: 192 channels, spatial downsampling
     - Stage 2: 384 channels, spatial downsampling
     - Stage 3: 768 channels, no spatial change
   - **Middle Block**: 768 channels
   - **Decoder Path**:
     - Stage 3: 384 channels (1536 → 384), upsampling
     - Stage 2: 192 channels (576 → 192), upsampling
     - Stage 1: 96 channels (288 → 96), upsampling
     - Final spatial resolution matches input

3. **Diffusion Process**
   - Timesteps: 1000
   - Progressive denoising with conditional generation
   - Beta schedule: Linear scheduling from start to end
   - Sampling process includes noise prediction and denoising

### Key Features
- **Spatial Changes**:
  - Encoder: Progressive downsampling (256 → 128 → 64 → 32)
  - Decoder: Progressive upsampling (32 → 64 → 128 → 256)
  
- **Channel Attention**:
  - Multi-scale feature processing
  - Skip connections between encoder and decoder
  - Channel projection in decoder stages

- **Conditioning**:
  - Input image conditioning for guided denoising
  - Timestep embedding for diffusion process

## Setup
- bash
- Create conda environment
- conda create -n reflection python=3.8
- conda activate reflection
- Install requirements
- pip install -r requirements.txt

## Training
- bash
- python train.py --batch-size 4 --epochs 100 --lr 2e-4

## Project Structure
- reflection/
- ├── models/
- │ ├── init.py
- │ ├── diffusion.py
- │ └── transformer.py
- ├── utils/
- │ ├── init.py
- │ └── training.py
- ├── train.py
- ├── requirements.txt
- └── README.md

## Training

- Training details are implemented in `train.py`. 
- The model is trained using a diffusion-based approach with both reflection and reflection-free images.

## Inference

- To run inference on a single image:

bash
- python inference.py \
--input path/to/input/image.jpg \
--output_dir ./outputs/sample1 \
--checkpoint path/to/checkpoint.pth \
--save_interval 50

### Inference Parameters

- `--input`: Path to input image with reflection
- `--output_dir`: Directory to save output images
- `--checkpoint`: Path to model checkpoint
- `--save_interval`: Save intermediate results every N timesteps (default: 50)

### Output

The inference script saves:
1. Input image
2. Initial noise image
3. Intermediate results at specified intervals
4. Final reflection-removed result

## Implementation Details

### Diffusion Model

The diffusion process is implemented in `models/diffusion.py` with two main classes:
- `DiffusionModel`: Base diffusion implementation
- `ConditionedDiffusionModel`: Extended version for conditional generation

### Key Features

- Progressive denoising visualization
- Customizable sampling steps
- Support for both CPU and GPU inference
- Intermediate result visualization

## Repository Information

- Personal GitHub: [VincentChoi33](https://github.com/VincentChoi33)
- Company GitHub: [nsptr](https://github.com/nsptr)

## Notes

- The model supports both CPU and GPU inference
- For GPU usage, ensure CUDA drivers match PyTorch version
- Intermediate results can be visualized to understand the denoising process
