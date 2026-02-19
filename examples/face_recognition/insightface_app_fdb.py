import os
import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# print(cv2.getBuildInformation())
# sys.exit(0)

# =========================
# 参数
# =========================
FACE_DB_DIR = "/home/good/wkspace/pyremodel/insightface/data/huaibei_face/"
TEST_IMAGE = "/home/good/wkspace/pyremodel/insightface/data/test.jpg"
SIM_THRESHOLD = 0.45  # 常用范围 0.35 ~ 0.5
USE_GPU = False

# =========================
# initialization InsightFace
# =========================
providers = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if USE_GPU
    else ["CPUExecutionProvider"]
)

app = FaceAnalysis(
    name="buffalo_l",
    providers=providers,
    root="/home/good/wkspace/pyremodel/insightface/insightface_model/",
)
app.prepare(ctx_id=0 if USE_GPU else -1, det_size=(640, 640))


# =========================
# 构建人脸库
# =========================
def build_face_db(db_dir):
    face_db = []

    for person_name in os.listdir(db_dir):
        person_dir = os.path.join(db_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = app.get(img)
            if len(faces) == 0:
                continue

            emb = faces[0].embedding
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        if len(embeddings) == 0:
            continue

        mean_emb = np.mean(embeddings, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        face_db.append({"name": person_name, "embedding": mean_emb})

        print(f"[INFO] Loaded {person_name}, images={len(embeddings)}")

    return face_db


# =========================
# 最相似
# =========================
def search_face(query_emb, face_db):
    best_name = "unknown"
    best_score = -1.0

    for item in face_db:
        score = np.dot(query_emb, item["embedding"])
        if score > best_score:
            best_score = score
            best_name = item["name"]

    if best_score < SIM_THRESHOLD:
        best_name = "unknown"

    return best_name, best_score


# =========================
# main
# =========================
def main():
    # 1
    print("[INFO] Building face database...")
    face_db = build_face_db(FACE_DB_DIR)
    print(f"[INFO] Face DB size: {len(face_db)}")

    # if len(face_db) == 0:
    #     print("[ERROR] Face DB is empty.")
    #     return

    # 2
    img = cv2.imread(TEST_IMAGE)
    if img is None:
        print("[ERROR] Failed to load test image.")
        return

    # 3
    faces = app.get(img)

    for face in faces:
        emb = face.embedding
        emb = emb / np.linalg.norm(emb)

        name, score = search_face(emb, face_db)

        # 人脸框
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{name} {score:.2f}"
        cv2.putText(
            img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    # 4
    cv2.imshow("Face Recognition Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
RetinaFace
ArcFace

训练数据集 WebFace600K 
"""
