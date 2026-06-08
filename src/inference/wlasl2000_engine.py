
from __future__ import annotations

from pathlib import Path
from collections import Counter, deque
import json
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGRUAttentionDeploy(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.35):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        bi_hidden = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(bi_hidden, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(bi_hidden, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.input_projection(x)
        gru_out, _ = self.gru(x)
        scores = self.attention(gru_out).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = torch.sum(gru_out * weights, dim=1)
        return self.classifier(context)


class WLASL2000Engine:
    def __init__(self, project_root="E:/Be_My_Ear"):
        self.project_root = Path(project_root)
        self.deploy_dir = self.project_root / "app" / "models" / "ASL" / "WLASL2000"
        self.config_file = self.deploy_dir / "wlasl2000_deployment_config.json"

        if not self.config_file.exists():
            raise FileNotFoundError(f"Missing deployment config: {self.config_file}")

        with open(self.config_file, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model_path = Path(self.config["model_path"])
        self.label_map_path = Path(self.config["label_map_path"])
        self.rules = self.config.get("confidence_rules", {})

        with open(self.label_map_path, "r", encoding="utf-8") as f:
            raw_label_map = json.load(f)

        self.id_to_gloss = {
            int(label_id): item["gloss"]
            for label_id, item in raw_label_map.items()
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = int(self.config["sequence_length"])
        self.base_feature_size = int(self.config["base_keypoint_shape"][1])
        self.input_size = int(self.config["input_shape"][1])
        self.num_classes = int(self.config["num_classes"])

        self.left_hand_size = 21 * 3
        self.right_hand_size = 21 * 3
        self.pose_size = 33 * 4
        self.mp_holistic = mp.solutions.holistic

        self.norm_stats_file = self._find_norm_stats_file()
        stats = np.load(self.norm_stats_file)
        self.train_mean = stats["mean"].astype(np.float32)
        self.train_std = stats["std"].astype(np.float32)

        self.model = self._load_model()

    def _find_norm_stats_file(self):
        candidates = []
        if "norm_stats_path" in self.config:
            candidates.append(Path(self.config["norm_stats_path"]))

        model_dir = self.project_root / "models" / "ASL" / "WLASL2000"
        selected_name = self.config.get("selected_model_name", "").lower()

        if "light v3" in selected_name:
            candidates.append(model_dir / "wlasl2000_light_v3_two_stage_finetuned_from_wlasl1000_train_norm_stats.npz")
        if "light v2" in selected_name:
            candidates.append(model_dir / "wlasl2000_light_v2_finetuned_from_wlasl1000_train_norm_stats.npz")

        candidates.extend(sorted(model_dir.glob("*norm_stats*.npz")))

        for p in candidates:
            if p.exists():
                return p

        raise FileNotFoundError(f"Could not find norm_stats .npz in {model_dir}")

    def _load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        hidden_size = int(checkpoint.get("hidden_size", 320))
        num_layers = int(checkpoint.get("num_layers", 2))
        dropout = float(checkpoint.get("dropout", 0.35))

        model = BiGRUAttentionDeploy(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_classes=self.num_classes,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def extract_landmarks_from_results(self, results):
        if results.left_hand_landmarks:
            left = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
        else:
            left = np.zeros(self.left_hand_size, dtype=np.float32)

        if results.right_hand_landmarks:
            right = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
        else:
            right = np.zeros(self.right_hand_size, dtype=np.float32)

        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()
        else:
            pose = np.zeros(self.pose_size, dtype=np.float32)

        return np.concatenate([left, right, pose]).astype(np.float32)

    def process_frame_to_keypoints(self, frame_bgr, holistic):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        return self.extract_landmarks_from_results(results)

    def hand_activity_score(self, sequence):
        sequence = np.asarray(sequence, dtype=np.float32)
        left_hand = sequence[:, :63]
        right_hand = sequence[:, 63:126]
        hands = np.concatenate([left_hand, right_hand], axis=1)
        non_zero_ratio = np.mean(np.abs(hands) > 1e-6)
        movement = np.mean(np.abs(hands[1:] - hands[:-1]))
        active = non_zero_ratio > 0.05 and movement > 0.002
        return {"non_zero_ratio": float(non_zero_ratio), "movement": float(movement), "active": bool(active)}

    def prepare_model_input(self, sequence):
        sequence = np.asarray(sequence, dtype=np.float32)
        normalised = (sequence - self.train_mean.reshape(1, -1)) / (self.train_std.reshape(1, -1) + 1e-6)
        velocity = np.zeros_like(normalised, dtype=np.float32)
        velocity[1:] = normalised[1:] - normalised[:-1]
        features = np.concatenate([normalised, velocity], axis=1).astype(np.float32)
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def predict_keypoint_sequence(self, sequence, top_k=5):
        x = self.prepare_model_input(sequence).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

        top_ids = np.argsort(probs)[-top_k:][::-1]
        top_predictions = [
            {"label_id": int(i), "gloss": self.id_to_gloss.get(int(i), str(i)), "probability": float(probs[i])}
            for i in top_ids
        ]

        top1 = top_predictions[0]
        top2_prob = top_predictions[1]["probability"] if len(top_predictions) > 1 else 0.0

        return {
            "top1_label_id": top1["label_id"],
            "top1_gloss": top1["gloss"],
            "top1_confidence": top1["probability"],
            "top1_top2_margin": float(top1["probability"] - top2_prob),
            "top_k": top_predictions
        }

    def decision_from_prediction(self, prediction, recent_predictions=None):
        confidence = prediction["top1_confidence"]
        margin = prediction["top1_top2_margin"]

        auto_accept = float(self.rules.get("auto_accept_confidence", 0.45))
        uncertain_min = float(self.rules.get("uncertain_min_confidence", 0.25))
        required_margin = float(self.rules.get("top1_top2_margin", 0.05))
        min_repeated = int(self.rules.get("min_repeated_predictions", 2))
        stability_count = int(self.rules.get("stability_window_count", 3))

        stable = True
        if recent_predictions is not None and len(recent_predictions) >= min_repeated:
            last_items = list(recent_predictions)[-stability_count:]
            glosses = [item["top1_gloss"] for item in last_items]
            stable = Counter(glosses)[prediction["top1_gloss"]] >= min_repeated

        if confidence >= auto_accept and margin >= required_margin and stable:
            status = "accepted"
            message = f"Detected sign: {prediction['top1_gloss']}"
        elif confidence >= uncertain_min:
            status = "uncertain"
            message = "I am not fully sure."
        else:
            status = "repeat"
            message = "Please sign again slowly."

        return {"status": status, "stable": stable, "message": message, "confidence": confidence, "margin": margin}

    def resample_sequence(self, sequence, target_length=None):
        target_length = target_length or self.sequence_length
        sequence = np.asarray(sequence, dtype=np.float32)

        if len(sequence) == target_length:
            return sequence
        if len(sequence) == 0:
            return np.zeros((target_length, self.base_feature_size), dtype=np.float32)

        old_x = np.linspace(0, 1, len(sequence))
        new_x = np.linspace(0, 1, target_length)
        resampled = [np.interp(new_x, old_x, sequence[:, i]) for i in range(sequence.shape[1])]
        return np.stack(resampled, axis=1).astype(np.float32)

    def extract_keypoints_from_video(self, video_path, max_frames=None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        keypoints = []
        read_count = 0

        with self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            while True:
                if max_frames is not None and read_count >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                keypoints.append(self.process_frame_to_keypoints(frame, holistic))
                read_count += 1

        cap.release()
        return np.array(keypoints, dtype=np.float32), fps, frame_count

    def create_sliding_windows(self, keypoints, window_size=None, stride=15):
        window_size = window_size or self.sequence_length
        windows, meta = [], []
        n = len(keypoints)

        if n <= window_size:
            windows.append(self.resample_sequence(keypoints, self.sequence_length))
            meta.append({"start_frame": 0, "end_frame": max(n - 1, 0)})
            return windows, meta

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            windows.append(self.resample_sequence(keypoints[start:end], self.sequence_length))
            meta.append({"start_frame": start, "end_frame": end - 1})

        return windows, meta

    def run_video_inference(self, video_path, stride=15, top_k=5, max_frames=None):
        keypoints, fps, frame_count = self.extract_keypoints_from_video(video_path, max_frames=max_frames)
        windows, meta = self.create_sliding_windows(keypoints, stride=stride)

        recent = deque(maxlen=int(self.rules.get("stability_window_count", 3)))
        rows, candidate_events = [], []

        for i, window in enumerate(windows):
            activity = self.hand_activity_score(window)

            if not activity["active"]:
                rows.append({"window_id": i, "status": "waiting", "top1_gloss": "", "confidence": 0.0, "top5_glosses": ""})
                continue

            pred = self.predict_keypoint_sequence(window, top_k=top_k)
            recent.append(pred)
            decision = self.decision_from_prediction(pred, recent)

            rows.append({
                "window_id": i,
                "start_frame": meta[i]["start_frame"],
                "end_frame": meta[i]["end_frame"],
                "status": decision["status"],
                "top1_gloss": pred["top1_gloss"],
                "confidence": pred["top1_confidence"],
                "margin": pred["top1_top2_margin"],
                "top5_glosses": ", ".join([x["gloss"] for x in pred["top_k"]]),
            })

            if decision["status"] in ["accepted", "uncertain"]:
                candidate_events.append({
                    "event_id": len(candidate_events) + 1,
                    "time": float(meta[i]["start_frame"] / fps) if fps else float(i),
                    "status": "locked" if decision["status"] == "accepted" else "uncertain",
                    "top_k": pred["top_k"]
                })

        return {"rows": rows, "candidate_events": candidate_events, "fps": fps, "frame_count": frame_count}
