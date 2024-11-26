# check_paths.py
from config import Config

def check_data_paths():
    # 경로 존재 확인
    print(f"Reflection directory exists: {Config.REFLECT_DIR.exists()}")
    print(f"Clean directory exists: {Config.CLEAN_DIR.exists()}")
    
    # 각 디렉토리의 이미지 파일 수 확인
    reflect_images = list(Config.REFLECT_DIR.glob("*.jpg"))
    clean_images = list(Config.CLEAN_DIR.glob("*.jpg"))
    
    print(f"\nNumber of reflection images: {len(reflect_images)}")
    print(f"Number of clean images: {len(clean_images)}")
    
    # 처음 몇 개의 파일 이름 출력
    print("\nFirst few reflection images:")
    for img in reflect_images[:3]:
        print(f"- {img.name}")
        
    print("\nFirst few clean images:")
    for img in clean_images[:3]:
        print(f"- {img.name}")

# 실행
if __name__ == "__main__":
    check_data_paths()