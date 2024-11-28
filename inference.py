import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from models.diffusion import ConditionedDiffusionModel
from train import ReflectionRemovalModel, Config
import os
from tqdm import tqdm

def load_image(image_path):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

def save_image(tensor, path):
    """Save tensor as image"""
    transform = transforms.Compose([
        transforms.Normalize(mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5], std=[1/0.5, 1/0.5, 1/0.5]),
        transforms.ToPILImage()
    ])
    image = transform(tensor.squeeze(0).cpu())
    image.save(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to input image with reflection')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--save_interval', type=int, default=50, help='Save image every N timesteps')
    args = parser.parse_args()

    # Load model and config
    config = Config()
    config.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ReflectionRemovalModel(config).to(config.DEVICE)
    diffusion = ConditionedDiffusionModel(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load and process input image
    input_image = load_image(args.input).to(config.DEVICE)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save input image
    save_image(input_image, os.path.join(args.output_dir, 'input.jpg'))

    with torch.no_grad():
        print("Removing reflection...")
        # 초기 노이즈 생성 (GPU에서)
        x = torch.randn(1, 3, 256, 256, device=config.DEVICE)
        
        # Save initial noise (CPU로 이동 후 저장)
        save_image(x.cpu(), os.path.join(args.output_dir, 'timestep_1000_noise.jpg'))
        
        # 역확산 과정
        for i in tqdm(range(config.TIMESTEPS-1, -1, -1), desc='Sampling'):
            t = torch.tensor([i], device=config.DEVICE)  # timestep도 GPU로
            x = diffusion.p_sample(
                model=model,
                x=x,
                condition=input_image,
                t=t,
                t_index=i
            )
            
            # Save intermediate results (CPU로 이동 후 저장)
            if i % args.save_interval == 0 or i == config.TIMESTEPS-1 or i == 0:
                save_image(x.cpu(), os.path.join(args.output_dir, f'timestep_{i:04d}.jpg'))
        
        # Save final result (CPU로 이동 후 저장)
        save_image(x.cpu(), os.path.join(args.output_dir, 'final_result.jpg'))
        print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()