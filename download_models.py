# download_models.py
import os
import urllib.request
import zipfile

def download_models():
    """Download required models if they don't exist"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Model URLs (you may need to update these)
    model_urls = {
        "opencv_face_detector.pbtxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt",
        "opencv_face_detector_uint8.pb": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/opencv_face_detector_uint8.pb",
        "age_deploy.prototxt": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_deploy.prototxt",
        "age_net.caffemodel": "https://drive.google.com/uc?export=download&id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW",
        "gender_deploy.prototxt": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_deploy.prototxt", 
        "gender_net.caffemodel": "https://drive.google.com/uc?export=download&id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ"
    }
    
    print("📥 Downloading AI models...")
    
    for filename, url in model_urls.items():
        filepath = os.path.join(models_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"⬇️ Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
        else:
            print(f"✅ {filename} already exists")
    
    print("🎉 Model download completed!")

if __name__ == "__main__":
    download_models()