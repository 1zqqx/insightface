import os
import sys
from typing import Dict, List

import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
    root="/home/good/wkspace/pyremodel/insightface/insightface_model/",
)  # Use 'CUDAExecutionProvider' for GPU; "CPUExecutionProvider" for CPU
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 for CPU, 0 for GPU


class FaceIdentityChecker:
    def __init__(self, face_root: str):
        """
        :param face_root: 人脸数据根目录，例如 huaibei_face
        """
        self.face_root = face_root
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> Dict[str, List[str]]:
        """
        扫描目录，构建 {person_name: [image_path, ...]}
        """
        dataset = {}

        for person_name in os.listdir(self.face_root):
            person_dir = os.path.join(self.face_root, person_name)
            if not os.path.isdir(person_dir):
                continue

            images = []
            for file in os.listdir(person_dir):
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    images.append(os.path.join(person_dir, file))

            if images:
                dataset[person_name] = images

        return dataset

    def _get_face_embedding(self, image_path):
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

    def _compare_faces(self, emb1, emb2, threshold=0.55):
        # Adjust this threshold according to your usecase.
        """Compare two embeddings using cosine similarity"""

        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )  # 余弦相似度
        return similarity, similarity > threshold

    def _check_same_person(self, image_paths: List[str]):
        """
        判断一组图片是否属于同一人
        """
        if len(image_paths) < 2:
            return None

        similarity_score, is_same_person = None, None
        try:
            emb1 = self._get_face_embedding(image_paths[0])  # 512 vector
            emb2 = self._get_face_embedding(image_paths[1])

            similarity_score, is_same_person = self._compare_faces(emb1, emb2, 0.45)

            print(f"Similarity Score: {similarity_score:.4f}")
            print(f"Same person? {'YES' if is_same_person else 'NO'}")
        except Exception as e:
            print(f"Error: {str(e)}")

        return similarity_score, is_same_person

    def check_all_identities(self):
        """
        对每个身份目录进行一致性检查
        """
        results = {}

        for person, images in self.dataset.items():
            # print(f"Checking identity for {person} with {type(images)}")
            # sys.exit(0)
            try:
                score, is_same = self._check_same_person(images)
                results[person] = dict(similarity_score=score, is_same_person=is_same)
            except NotImplementedError:
                results[person] = None

        return results


def main():
    face_root = "/home/good/wkspace/pyremodel/insightface/data/huaibei_face"
    checker = FaceIdentityChecker(face_root)

    print("数据加载结果：")
    for person, images in checker.dataset.items():
        print(f"{person}: {len(images)} 张图片")

    print("\n身份一致性检查：")
    results = checker.check_all_identities()

    print("\nResult:")
    for person, result in results.items():
        print(f"{person}: {result}")

    # print("\n跨身份冲突检查：")
    # conflicts = checker.cross_identity_check()
    # for p1, p2 in conflicts:
    #     print(f"可能是同一人: {p1} <-> {p2}")


if __name__ == "__main__":
    main()
