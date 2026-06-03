from pathlib import Path
import json
from collections import Counter


RAW_DIR = Path("data/raw/ASL")
VIDEOS_DIR = RAW_DIR / "videos"
META_FILE = RAW_DIR / "nslt_100.json"
CLASS_LIST_FILE = RAW_DIR / "wlasl_class_list.txt"


def load_class_names():
    class_names = {}

    if not CLASS_LIST_FILE.exists():
        print("WARNING: wlasl_class_list.txt not found.")
        return class_names

    with open(CLASS_LIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")

            if len(parts) >= 2:
                class_id = int(parts[0])
                gloss = parts[1]
                class_names[class_id] = gloss

    return class_names


def main():
    print("Checking WLASL100 dataset...")
    print(f"Metadata file: {META_FILE}")
    print(f"Videos folder: {VIDEOS_DIR}")

    if not META_FILE.exists():
        print("ERROR: nslt_100.json not found.")
        return

    if not VIDEOS_DIR.exists():
        print("ERROR: videos folder not found.")
        return

    class_names = load_class_names()

    with open(META_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nTotal metadata entries: {len(data)}")

    labels = []
    existing_videos = 0
    missing_videos = 0

    for video_id, item in data.items():
        class_id = item["action"][0]
        labels.append(class_id)

        video_path = VIDEOS_DIR / f"{video_id}.mp4"

        if video_path.exists():
            existing_videos += 1
        else:
            missing_videos += 1

    label_counts = Counter(labels)

    print(f"Total classes: {len(label_counts)}")
    print(f"Existing videos: {existing_videos}")
    print(f"Missing videos: {missing_videos}")

    print("\nTop 10 classes by number of videos:")
    for class_id, count in label_counts.most_common(10):
        gloss = class_names.get(class_id, "unknown")
        print(f"Class {class_id} | {gloss}: {count} videos")


if __name__ == "__main__":
    main()