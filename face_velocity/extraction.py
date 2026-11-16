"""Fixed SCRFD detection, five-point alignment and CASIA embedding extraction."""

import cv2
import numpy as np
import onnxruntime as ort
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.model_zoo.scrfd import SCRFD
from insightface.utils.face_align import norm_crop2

from face_velocity.inputs import digest as file_hash
from face_velocity.pipeline import select_target


def load_models(detector_path, recognizer_path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    def session(path):
        return ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    detector = SCRFD(str(detector_path), session=session(detector_path))
    detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.5, nms_thresh=0.4)
    recognizer = ArcFaceONNX(str(recognizer_path), session=session(recognizer_path))
    if recognizer.input_size != (112, 112):
        raise ValueError("This audit requires the selected 112x112 checkpoint.")
    return detector, recognizer


def extract(row, detector, recognizer, output):
    result = dict(row, status="not_processed", faces=[], selected_face_index=None)
    if file_hash(row["path"]) != row["sha256"]:
        raise ValueError(f"Input changed after selection: {row['path']}")
    image = cv2.imread(row["path"])
    if image is None:
        result["status"] = "unreadable"
        return result, None
    result["image_shape"] = list(image.shape)
    boxes, landmarks = detector.detect(image, input_size=(640, 640))
    if landmarks is None or len(boxes) != len(landmarks):
        result["status"] = "missing_landmarks"
        return result, None
    for index, (box, points) in enumerate(zip(boxes, landmarks)):
        result["faces"].append(
            dict(
                face_index=index,
                bbox=box[:4].tolist(),
                detector_score=float(box[4]),
                landmarks=points.tolist(),
                status="detected",
            )
        )
    result["face_count"] = len(boxes)
    status, target = select_target(result)
    if target is None:
        result["status"] = status
        return result, None
    result["selected_face_index"] = target["face_index"]
    points = np.asarray(target["landmarks"], dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        result["status"] = "invalid_landmarks"
        return result, None
    crop, transform = norm_crop2(image, points, image_size=112)
    if crop.shape != (112, 112, 3) or not np.isfinite(transform).all():
        result["status"] = "invalid_crop"
        return result, None
    if output is not None:
        crop_path = output / "crops" / f"{row['image_id']}.png"
        if not cv2.imwrite(str(crop_path), crop):
            raise OSError(f"Could not save {crop_path}")
        result.update(crop_path=str(crop_path), crop_sha256=file_hash(crop_path))
    result.update(
        crop_shape=list(crop.shape),
        alignment_matrix=transform.tolist(),
    )
    features = recognizer.get_feat(crop)
    norm = float(np.linalg.norm(features))
    if features.shape != (1, 512) or not np.isfinite(features).all():
        result["status"] = "invalid_embedding"
        return result, None
    if not np.isfinite(norm) or norm <= 0:
        result["status"] = "invalid_embedding_norm"
        return result, None
    unit = features[0] / norm
    result.update(
        status="valid_embedding",
        embedding_shape=list(features[0].shape),
        raw_l2_norm=norm,
        normalized_l2_norm=float(np.linalg.norm(unit)),
        first_eight_raw_features=features[0, :8].tolist(),
    )
    return result, features[0]
