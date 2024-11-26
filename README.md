# Reflection Removal Model

This project implements a deep learning model for reflection removal in images using a diffusion-based approach.

## Setup
bash
Create conda environment
conda create -n reflection python=3.8
conda activate reflection
Install requirements
pip install -r requirements.txt

## Training
bash
python train.py --batch-size 4 --epochs 100 --lr 2e-4

## Project Structure
reflection/
├── models/
│ ├── init.py
│ ├── diffusion.py
│ └── transformer.py
├── utils/
│ ├── init.py
│ └── training.py
├── train.py
├── requirements.txt
└── README.md