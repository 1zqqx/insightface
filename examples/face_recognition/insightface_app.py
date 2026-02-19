import cv2

# import insightface
import numpy as np
from insightface.app import FaceAnalysis

# Initialize face analysis model
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
    root="/home/good/wkspace/pyremodel/insightface/insightface_model/",
)  # Use 'CUDAExecutionProvider' for GPU; "CPUExecutionProvider" for CPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU

"""
w600k_r50.onnx	人脸特征提取/人脸识别 (Face Feature Extraction / Recognition)
w600k=WebFace600K (约600万张人脸的训练数据集) ; r50=ResNet50 (模型骨干网络)	
将人脸转化为512维特征向量, 通过对比特征向量相似度实现人脸识别 (如人脸解锁、考勤打卡等）
"""


def get_face_embedding(image_path):
    """Extract face embedding from an image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = app.get(img)

    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")

    return faces[0].embedding


def compare_faces(emb1, emb2, threshold=0.55):
    # Adjust this threshold according to your usecase.
    """Compare two embeddings using cosine similarity"""

    similarity = np.dot(emb1, emb2) / (
        np.linalg.norm(emb1) * np.linalg.norm(emb2)
    )  # 余弦相似度
    return similarity, similarity > threshold


# Paths to your Indian face images
# data/huaibei_face/litongxin/litongxin_001.png
# image1_path = "/home/good/wkspace/pyremodel/insightface/data/huaibei_face/litongxin/litongxin_001.png"
# image2_path = "/home/good/wkspace/pyremodel/insightface/data/huaibei_face/litongxin/litongxin_002.jpg"
# image1_path = "data/huaibei_face/genghonhlin/genghonhlin_001.png"
# image2_path = "data/huaibei_face/genghonhlin/genghonhlin_002.jpg"
# image1_path = "data/huaibei_face/mahanhui/mahanhui_003.png"
# image2_path = "data/huaibei_face/mahanhui/mahanhui_002.jpg"
image1_path = "data/huaibei_face/sunpo/sunpo_002.jpg"
image2_path = "data/huaibei_face/sunpo/sunpo_001.png"
# image1_path = "data/huaibei_face/yangxiaofeng/yangxiaofeng_002.jpg"
# image2_path = "data/huaibei_face/yangxiaofeng/yangxiaofeng_001.png"

try:
    # Get embeddings
    emb1 = get_face_embedding(image1_path)  # 512 vector
    emb2 = get_face_embedding(image2_path)

    # print(f"Embedding 1: {emb1.shape}")
    # print(f"Embedding 2: {emb2.shape}")

    # Compare faces
    similarity_score, is_same_person = compare_faces(emb1, emb2)

    print(f"Similarity Score: {similarity_score:.4f}")
    print(f"Same person? {'YES' if is_same_person else 'NO'}")

except Exception as e:
    print(f"Error: {str(e)}")
