# test_models.py
import os
import cv2

def check_models():
    models_dir = "models"
    required_models = [
        "opencv_face_detector.pbtxt",
        "opencv_face_detector_uint8.pb", 
        "age_deploy.prototxt",
        "age_net.caffemodel",
        "gender_deploy.prototxt",
        "gender_net.caffemodel"
    ]
    
    print("🔍 Checking model files...")
    
    for model in required_models:
        path = os.path.join(models_dir, model)
        if os.path.exists(path):
            print(f"✅ {model} - FOUND")
            # Try to load the model
            try:
                if model.endswith('.pb'):
                    net = cv2.dnn.readNet(path)
                    print(f"   ✅ Model loaded successfully")
                elif model.endswith('.caffemodel'):
                    # For caffe models, we need both .caffemodel and .prototxt
                    proto = model.replace('.caffemodel', '.prototxt')
                    proto_path = os.path.join(models_dir, proto)
                    if os.path.exists(proto_path):
                        net = cv2.dnn.readNet(path, proto_path)
                        print(f"   ✅ Model loaded successfully")
                    else:
                        print(f"   ❌ Missing prototxt file: {proto}")
            except Exception as e:
                print(f"   ❌ Error loading model: {e}")
        else:
            print(f"❌ {model} - MISSING")
    
    print("\n💡 If models are missing, run: python download_models.py")

if __name__ == "__main__":
    check_models()