import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import glob

"""
Script thu thập dữ liệu keypoints phục vụ huấn luyện LSTM phát hiện té ngã.
- Hỗ trợ: Video file hoặc Camera
- Lưu chuỗi keypoints (mặc định 30 frame) vào file .npy
- Gán nhãn thủ công (té ngã/không té ngã)
"""

# ==== CẤU HÌNH ====
MODEL_PATH = "runs/pose/coco_pose_demo/weights/best.pt"  # Đường dẫn model pose
SEQUENCE_LENGTH = 30             # Số frame mỗi chuỗi
SAVE_DIR = "fall_sequences"      # Thư mục lưu dữ liệu
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== HÀM HỖ TRỢ ====
def save_sequence(sequence, label, idx):
    arr = np.array(sequence)  # (SEQUENCE_LENGTH, 17, 2)
    fname = f"{label}_{idx:04d}.npy"
    np.save(os.path.join(SAVE_DIR, fname), arr)
    print(f"Đã lưu: {fname}")

def auto_collect_from_folders(root_folder):
    model = YOLO(MODEL_PATH)
    idx = int(time.time())
    subfolders = [f.path for f in os.scandir(root_folder) if f.is_dir()]
    if not subfolders:
        print(f"Không tìm thấy folder con trong {root_folder}")
        return
    for sub in subfolders:
        label = os.path.basename(sub)
        video_files = glob.glob(os.path.join(sub, '*.mp4')) + glob.glob(os.path.join(sub, '*.avi')) + glob.glob(os.path.join(sub, '*.mkv'))
        print(f"=== Nhãn: {label} | {len(video_files)} video ===")
        for vid in video_files:
            print(f"Đang xử lý video: {vid}")
            cap = cv2.VideoCapture(vid)
            sequence = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)
                keypoints = None
                if results and len(results) > 0:
                    kpts = getattr(results[0], 'keypoints', None)
                    if kpts is not None and hasattr(kpts, 'xy'):
                        kpts_xy = kpts.xy.cpu().numpy()
                        if len(kpts_xy) > 0:
                            keypoints = kpts_xy[0]
                if keypoints is not None:
                    sequence.append(keypoints)
                    if len(sequence) == SEQUENCE_LENGTH:
                        save_sequence(sequence, label, idx)
                        sequence = []
                        idx += 1
            cap.release()
    print("Đã xong auto collect!")

def main():
    print("=== THU THẬP DỮ LIỆU KEYPOINTS PHÁT HIỆN TÉ NGÃ ===")
    print("1. Camera\n2. Video file\n3. Folder chứa nhiều video\n4. Tự động duyệt tất cả folder con (fall, normal...)")
    src = input("Chọn nguồn dữ liệu (1/2/3/4): ").strip()
    video_files = []
    if src == '1':
        cap = cv2.VideoCapture(0)
        video_files = [cap]
    elif src == '2':
        path = input("Nhập đường dẫn video: ").strip()
        video_files = [path]
    elif src == '3':
        folder = r"C:/Data_Fall"
        video_files = glob.glob(os.path.join(folder, '*.mp4')) + glob.glob(os.path.join(folder, '*.avi')) + glob.glob(os.path.join(folder, '*.mkv'))
        print(f"Tìm thấy {len(video_files)} video trong {folder}.")
    elif src == '4':
        folder = r"C:/Data_Fall"
        auto_collect_from_folders(folder)
        return
    else:
        print("Lựa chọn không hợp lệ!")
        return
    model = YOLO(MODEL_PATH)
    idx = int(time.time())
    for vid in video_files:
        if src == '1':
            cap = vid
        else:
            print(f"Đang xử lý video: {vid}")
            cap = cv2.VideoCapture(vid)
        sequence = []
        label = input("Nhãn cho chuỗi này (fall/normal): ").strip()
        print("Nhấn SPACE để bắt đầu lưu chuỗi, ESC để chuyển video hoặc thoát.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Không đọc được frame hoặc hết video!")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            keypoints = None
            if results and len(results) > 0:
                kpts = getattr(results[0], 'keypoints', None)
                if kpts is not None and hasattr(kpts, 'xy'):
                    kpts_xy = kpts.xy.cpu().numpy()
                    if len(kpts_xy) > 0:
                        keypoints = kpts_xy[0]  # Lấy người đầu tiên
            disp = frame.copy()
            if keypoints is not None:
                for x, y in keypoints:
                    cv2.circle(disp, (int(x), int(y)), 5, (0,0,255), -1)
            cv2.imshow("Collect Fall Data", disp)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            if key == 32 and keypoints is not None:  # SPACE
                sequence.append(keypoints)
                print(f"Đã thu thập {len(sequence)}/{SEQUENCE_LENGTH} frame...")
                if len(sequence) == SEQUENCE_LENGTH:
                    save_sequence(sequence, label, idx)
                    sequence = []
                    idx += 1
                    label = input("Nhãn cho chuỗi tiếp theo (fall/normal): ").strip()
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
