Camera / Video / Image
        ↓
Convert → RGB numpy array
        ↓
YOLOv8 model inference
        ↓
Result objects
   ├─ boxes
   └─ keypoints
        ↓
Draw skeleton + bbox
        ↓
Resize
        ↓
Display on Tkinter
# Traffic Sign & Pose Detection GUI

## Mô tả
Ứng dụng GUI sử dụng YOLOv8 để nhận diện biển báo giao thông và ước lượng pose (keypoints) trên ảnh, video, hoặc camera. Hỗ trợ inference bằng PyTorch hoặc OpenVINO (iGPU).

## Tính năng
- Chọn ảnh để detect và vẽ pose.
- Chọn video, phát và detect pose theo thời gian thực.
- Bật camera, detect pose trực tiếp.
- Tự động load model YOLOv8 (OpenVINO hoặc PyTorch).
- Vẽ bounding box, keypoints, skeleton với màu sắc tùy chỉnh.
- Lưu kết quả ảnh hoặc video đã xử lý.

## Hướng dẫn sử dụng
1. Cài đặt các thư viện cần thiết:
    ```
    pip install ultralytics opencv-python pillow pyav
    ```
2. Chạy ứng dụng:
    ```
    python pose_gui.py
    ```
3. Chọn model, ảnh, video hoặc bật camera để bắt đầu detect.

## Giải thích chi tiết code
- **Khai báo thư viện:** tkinter (GUI), PIL (xử lý ảnh), cv2 (OpenCV), numpy, ultralytics (YOLOv8), threading, time, av (PyAV).
- **TrafficSignDetectorGUI:** Lớp chính quản lý GUI và các chức năng.
    - `__init__`: Khởi tạo cửa sổ, biến trạng thái, gọi setup_gui.
    - `setup_gui`: Tạo giao diện, các nút chức năng, vùng hiển thị, thanh trạng thái.
    - `load_model_automatically`: Tìm và load model YOLO (OpenVINO hoặc PyTorch) tự động.
    - **Camera:**
        - `toggle_camera`, `start_camera`, `stop_camera`, `update_camera_frame`: Bật/tắt camera, đọc frame, detect, vẽ kết quả.
    - **Video:**
        - `toggle_video`, `select_and_play_video`, `start_video`, `play_video_frame`, `_run_detection`, `_cleanup_video_container`, `stop_video`: Chọn video, giải mã, detect, vẽ kết quả từng frame, dùng thread tăng tốc.
    - **Ảnh:**
        - `select_image`, `display_image`, `detect_signs`, `process_results`: Chọn ảnh, detect, vẽ kết quả.
    - `save_result`: Lưu ảnh hoặc video đã xử lý.
    - `update_status`: Cập nhật thông báo lên thanh trạng thái.
    - `on_closing`: Đảm bảo tắt camera/video khi đóng app.
- **main:** Khởi tạo app, xử lý sự kiện đóng cửa sổ, căn giữa cửa sổ, chạy vòng lặp chính.
- **Vẽ skeleton và keypoints:** Skeleton được vẽ theo nhóm xương (màu sắc khác nhau), keypoints vẽ bằng hình tròn nhỏ (có thể chỉnh màu).

## Tuỳ chỉnh màu skeleton/keypoints
Bạn có thể chỉnh màu từng nhóm xương và keypoints trong code theo ý muốn.

## Yêu cầu
- Python 3.8+
- Model YOLOv8 đã train (best.pt hoặc OpenVINO IR)

## Liên hệ
Nếu gặp lỗi hoặc cần hỗ trợ, hãy liên hệ qua email hoặc github.
