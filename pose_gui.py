import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
import threading
import time
import av  # PyAV - faster video decode than OpenCV

class TrafficSignDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Sign Detector - YOLOv8")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')

        # Initialize variables
        self.model = None
        self.current_image = None
        self.original_image = None
        self.model_path = None
        self.inference_backend = "PyTorch"
        self.confidence_threshold = 0.3
        self.current_video_path = None
        self.processed_video_path = None
        # For OpenVINO: do NOT pass device to model() — it's configured at load time.
        # For PyTorch: use 'cpu' or 'cuda:0'.
        self.device = None  # Will be set based on backend in load_model_automatically()
        
        # Camera variables
        self.camera = None
        self.camera_running = False
        self.camera_job = None
        
        # Video player variables (PyAV)
        self.video_container = None      # av.open() container
        self.video_stream = None         # video stream from container
        self.video_frame_iter = None     # frame iterator
        self.video_playing = False
        self.video_job = None
        self.video_frame_count = 0
        self.video_total_frames = 0
        self.video_source_fps = 30.0
        self.current_video_path = None
        self.processed_video_path = None
        self.detection_cache = {}
        self.fast_mode = True
        self.frame_skip = 9
        self._pending_detection = None   # result from background thread
        self._detection_thread = None

        # FPS tracking variables
        self.fps_start_time = None
        self.fps_frame_count = 0
        self.current_fps = 0.0
        self.fps_update_interval = 30  # Update FPS every 30 frames

        # Reset FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0

        # Setup GUI
        self.setup_gui()
    
    def setup_gui(self):
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50')
        title_frame.pack(pady=10)
        
        title_label = tk.Label(
            title_frame, 
            text="Traffic Sign Detection System",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Powered by YOLOv8 - Select an image to detect traffic signs",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#bdc3c7'
        )
        subtitle_label.pack()
        
        # Control panel
        control_frame = tk.Frame(self.root, bg='#34495e', relief='raised', bd=2)
        control_frame.pack(pady=10, padx=20, fill='x')
        
        # Image section  
        image_frame = tk.LabelFrame(control_frame, text="Image Processing",
                                   bg='#34495e', fg='#ecf0f1', font=('Arial', 10, 'bold'))
        image_frame.pack(side='left', padx=10, pady=5, fill='both', expand=True)
        
        tk.Button(
            image_frame,
            text="Select Image",
            command=self.select_image,
            bg='#27ae60',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=10
        ).pack(side='left', padx=5, pady=5)
        
        # Video button - combines select and play functionality
        self.video_button = tk.Button(
            image_frame,
            text="Select & Play Video",
            command=self.toggle_video,
            bg='#3498db',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=10
        )
        self.video_button.pack(side='left', padx=5, pady=5)
        
        # Camera button
        self.camera_button = tk.Button(
            image_frame,
            text="Start Camera",
            command=self.toggle_camera,
            bg='#16a085',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=10
        )
        self.camera_button.pack(side='left', padx=5, pady=5)
        
        tk.Button(
            image_frame,
            text="Detect Signs",
            command=self.detect_signs,
            bg='#e67e22',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=10
        ).pack(side='left', padx=5, pady=5)
        

        
        tk.Button(
            image_frame,
            text="Save Result", 
            command=self.save_result,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 10, 'bold'),
            relief='flat',
            padx=10
        ).pack(side='left', padx=5, pady=5)
        
        # Main content area
        content_frame = tk.Frame(self.root, bg='#2c3e50')
        content_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Image display area
        self.image_frame = tk.LabelFrame(content_frame, text="Image Display",
                                        bg='#2c3e50', fg='#ecf0f1', font=('Arial', 12, 'bold'))
        self.image_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        self.image_label = tk.Label(
            self.image_frame,
            text="\n\nNo image selected\n\nClick 'Select Image' to start",
            bg='#34495e',
            fg='#7f8c8d',
            font=('Arial', 14),
            justify='center'
        )
        self.image_label.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready - Load a model and select an image to begin",
            bd=1,
            relief='sunken',
            anchor='w',
            bg='#34495e',
            fg='#ecf0f1',
            font=('Arial', 9)
        )
        self.status_bar.pack(side='bottom', fill='x')
        
        # Auto-load model on startup
        self.load_model_automatically()
    
    def load_model_automatically(self):
        """Automatically load the trained model"""
        try:
            print("Auto-loading model...")
            openvino_patterns = [
                'runs/pose/*/weights/*_openvino_model',
                'openvino_model',  
                '*_openvino_model',
                'weights/*_openvino_model'
            ]
            pt_model_patterns = [
                'runs/pose/*/weights/best.pt',
                'runs/pose/*/weights/last.pt',
                'weights/*.pt',
                '*.pt'
            ]

            openvino_models = []
            for pattern in openvino_patterns:
                for path in glob.glob(pattern):
                    if os.path.isdir(path):
                        # Only count as valid if .xml and .bin files exist inside
                        xml_files = glob.glob(os.path.join(path, '*.xml'))
                        bin_files = glob.glob(os.path.join(path, '*.bin'))
                        if xml_files and bin_files:
                            openvino_models.append(path)
                        else:
                            print(f"Skipping {path}: missing .xml or .bin files")

            pt_models = []
            for pattern in pt_model_patterns:
                for path in glob.glob(pattern):
                    if os.path.isfile(path):
                        pt_models.append(path)

            print(f"OpenVINO model dirs found: {len(openvino_models)}")
            for f in openvino_models:
                print(f"   - {f}")
            print(f"PyTorch model files found: {len(pt_models)}")
            for f in pt_models:
                print(f"   - {f}")

            if openvino_models:
                latest_openvino = max(openvino_models, key=os.path.getctime)
                print(f"Trying OpenVINO model: {latest_openvino}")
                try:
                    self.model = YOLO(latest_openvino)
                    self.model_path = latest_openvino
                    self.inference_backend = "OpenVINO"
                    self.device = None  # default: AUTO
                    
                    # Try to use iGPU via OpenVINO GPU plugin
                    # Ultralytics format: device="intel:gpu" → OpenVINO device "GPU"
                    try:
                        import openvino as ov
                        core = ov.Core()
                        available_devices = core.available_devices
                        print(f"OpenVINO available devices: {available_devices}")
                        if 'GPU' in available_devices:
                            print("iGPU detected! Using intel:gpu device...")
                            # Warm-up: compile model on GPU (first run is slow)
                            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                            self.model.predict(dummy, device='intel:gpu', verbose=False)
                            self.device = 'intel:gpu'  # iGPU via OpenVINO
                            print("iGPU warm-up complete!")
                            self.update_status(f"Auto-loaded OpenVINO model on iGPU: {latest_openvino}")
                        else:
                            print(f"No iGPU found. Using AUTO. Devices: {available_devices}")
                            self.update_status(f"Auto-loaded OpenVINO model (AUTO): {latest_openvino}")
                    except ImportError:
                        print("openvino package not found, using default device")
                        self.update_status(f"Auto-loaded OpenVINO model: {latest_openvino}")
                    except Exception as gpu_err:
                        print(f"OpenVINO iGPU init failed, fallback to AUTO: {gpu_err}")
                        self.device = None
                        self.update_status(f"Auto-loaded OpenVINO model (CPU): {latest_openvino}")
                    return
                except Exception as ov_err:
                    print(f"OpenVINO load failed, fallback to .pt: {ov_err}")

            if pt_models:
                latest_model = max(pt_models, key=os.path.getctime)
                print(f"Latest .pt model: {latest_model}")
                self.model = YOLO(latest_model)
                self.model_path = latest_model
                self.inference_backend = "PyTorch"
                self.device = "cpu"  # PyTorch default; change to 'cuda:0' if NVIDIA GPU
                self.update_status(f"Auto-loaded model ({self.inference_backend}): {latest_model}")
            else:
                print("No model files found!")
                self.update_status("No trained model found. Please train/export a model first.")
                
        except Exception as e:
            print(f"Error auto-loading model: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.update_status(f"Error auto-loading model: {str(e)}")

    def toggle_camera(self):
        """Start or stop camera feed"""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera for real-time detection"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first!")
            return
            
        try:
            # Try to open camera (0 = default camera)
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not open camera!\nPlease check if camera is connected.")
                return
                
            self.camera_running = True
            self.camera_button.config(text="Stop Camera", bg='#e74c3c')
            self.update_status("Camera started - Real-time detection active")
            
            # Start camera update loop
            self.update_camera_frame()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera:\n{str(e)}")
            self.update_status(f"Camera error: {str(e)}")
    
    def stop_camera(self):
        """Stop camera feed"""
        self.camera_running = False
        
        if self.camera_job:
            self.root.after_cancel(self.camera_job)
            self.camera_job = None
            
        if self.camera:
            self.camera.release()
            self.camera = None
            
        self.camera_button.config(text="Start Camera", bg='#16a085')
        self.update_status("Camera stopped")
        
        # Clear display
        self.image_label.config(
            image="",
            text="\n\nCamera stopped\n\nClick 'Start Camera' to begin"
        )
    
    def update_camera_frame(self):
        """Update camera frame with detection"""
        if not self.camera_running or not self.camera:
            return
            
        try:
            ret, frame = self.camera.read()
            # ret return boolen type, if true is successfully read the frame, if false is failed to read the frame
            # frame is the image captured from the camera, it is a numpy array in BGR format (height x width x 3)
            if ret:
                # Convert BGR (camera) to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run detection on RGB frame
                # Only pass device if explicitly set (OpenVINO GPU or PyTorch cuda)
                predict_kwargs = {'conf': self.confidence_threshold}
                if self.device:
                    predict_kwargs['device'] = self.device
                results = self.model(frame_rgb, **predict_kwargs)
                if results and len(results) > 0:
                    from PIL import Image, ImageDraw, ImageFont
                    pil_frame = Image.fromarray(frame_rgb)
                    draw = ImageDraw.Draw(pil_frame)
                    boxes = results[0].boxes
                    keypoints = getattr(results[0], 'keypoints', None)
                    skeleton = [
                        (0,1), (0,2),      # nose -> eyes
                        (1,3), (2,4),     # eyes -> ears
                        (1,2),            # left eye <-> right eye
                        (3,5), (4,6),     # ears <-> shoulders
                        (5,6),            # shoulders
                        (5,7), (7,9),     # left arm
                        (6,8), (8,10),    # right arm
                        (5,11), (6,12),   # shoulders -> hips
                        (11,12),          # hips
                        (11,13), (13,15), # left leg
                        (12,14), (14,16)  # right leg
                    ]
                    # Draw only keypoints and skeleton
                    if keypoints is not None and hasattr(keypoints, 'xy'):
                        kpts_xy = keypoints.xy.cpu().numpy()
                        for person in kpts_xy:
                            for idx, (x, y) in enumerate(person):
                                draw.ellipse((x-8, y-8, x+8, y+8), fill='#ff2222', outline='#ff2222')
                            for a, b in skeleton:
                                if a < len(person) and b < len(person):
                                    x1, y1 = person[a]
                                    x2, y2 = person[b]
                                    draw.line((x1, y1, x2, y2), fill='yellow', width=4)
                        self.update_status("Live Camera - pose drawn")
                    else:
                        self.update_status("Live Camera - No detections")
                    self.display_image(pil_frame, fast=True)
                else:
                    pil_frame = Image.fromarray(frame_rgb)
                    self.display_image(pil_frame, fast=True)
                    self.update_status("Live Camera - No detections")
                
            # Schedule next frame update
            self.camera_job = self.root.after(50, self.update_camera_frame)
            # self.root.after(50, self.update_camera_frame) will return id of schduled job.
        except Exception as e:
            print(f"Camera frame error: {str(e)}")
            self.stop_camera()
    
    def toggle_video(self):
        """Toggle between selecting video and stopping playback"""
        if not self.video_playing:
            self.select_and_play_video()
        else:
            self.stop_video()

    def select_and_play_video(self):
        """Select video and automatically start playback with detection (PyAV)"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first!")
            return

        file_path = filedialog.askopenfilename(
            title="Select Video for Real-time Detection",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                if not os.path.exists(file_path):
                    messagebox.showerror("Error", f"Video file not found:\n{file_path}")
                    return

                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    messagebox.showerror("Error", f"Video file is empty or corrupted:\n{file_path}")
                    return

                # Stop any current video/camera
                if self.video_playing:
                    self.stop_video()
                if self.camera_running:
                    self.stop_camera()

                # Validate video with PyAV
                try:
                    test_container = av.open(file_path)
                    test_stream = test_container.streams.video[0]
                    fps = float(test_stream.average_rate) if test_stream.average_rate else 30.0
                    frame_count = test_stream.frames or 0
                    width = test_stream.codec_context.width
                    height = test_stream.codec_context.height

                    # Estimate frame count from duration if not available
                    if frame_count <= 0 and test_stream.duration:
                        duration_s = float(test_stream.duration * test_stream.time_base)
                        frame_count = int(duration_s * fps)

                    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0
                    test_container.close()

                    if width <= 0 or height <= 0:
                        messagebox.showerror("Error", "Invalid video dimensions.")
                        return

                except Exception as av_err:
                    messagebox.showerror("Error",
                        f"Cannot open video file with PyAV.\n\n"
                        f"File: {os.path.basename(file_path)}\n"
                        f"Error: {av_err}\n\n"
                        f"Try converting to MP4 (H.264) with VLC.")
                    return

                self.current_video_path = file_path
                self.update_status(f"Starting video: {os.path.basename(file_path)} ({width}x{height}, {duration:.1f}s, {fps:.1f} FPS)")
                self.start_video()

            except Exception as e:
                messagebox.showerror("Error", f"Unexpected error loading video:\n{str(e)}")
                self.update_status(f"Error loading video: {str(e)}")


    
    def select_image(self):
        """Select image for detection"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display image
                self.original_image = Image.open(file_path)
                self.current_image = self.original_image.copy()
                
                self.display_image(self.current_image)
                self.update_status(f"Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
                self.update_status(f"Error loading image: {str(e)}")
    
    def display_image(self, pil_image, fast=False):
        """Display image on GUI. Use fast=True for video/camera frames."""
        try:
            display_size = (600, 400)
            if fast:
                # For video: thumbnail modifies in-place, no need to copy
                # since Image.fromarray already created a new object
                pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(pil_image)
            else:
                image_copy = pil_image.copy()
                image_copy.thumbnail(display_size, Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image_copy)
            
            self.image_label.config(image=self.photo, text="")
            
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def detect_signs(self):
        """Run inference with YOLOv8"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first!")
            print("Error: Model not loaded!")
            return
        
        if not self.original_image:
            messagebox.showerror("Error", "Please select an image first!")
            print("Error: No image selected!")
            return
        
        try:
            print(f"Starting inference...")
            print(f"Model path: {self.model_path}")
            print(f"Image size: {self.original_image.size}")
            print(f"Image mode: {self.original_image.mode}")
            
            self.update_status("Running inference...")
            
            # Ensure image is RGB (3 channels) - preserves original colors
            if self.original_image.mode != 'RGB':
                print(f"Converting image from {self.original_image.mode} to RGB")
                rgb_image = self.original_image.convert('RGB')
            else:
                rgb_image = self.original_image
                print("Image already in RGB mode")
            
            # Convert PIL image to numpy array (maintains original colors)
            img_array = np.array(rgb_image)
            print(f"Array shape: {img_array.shape}")
            print(f"Array dtype: {img_array.dtype}")
            print(f"Running model inference with conf={self.confidence_threshold}, device={self.device}...")
            # Run inference
            predict_kwargs = {'conf': self.confidence_threshold, 'verbose': False}
            if self.device:
                predict_kwargs['device'] = self.device
            results = self.model(img_array, **predict_kwargs)
            print(f"Results type: {type(results)}")
            print(f"Results length: {len(results)}")
            
            # Process results
            if results and len(results) > 0:
                self.process_results(results[0])
            else:
                print("No results returned from model")
                self.update_status("No detections found")
            
        except Exception as e:
            print(f"Inference error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            messagebox.showerror("Error", f"Inference failed:\n{str(e)}")
            self.update_status(f"Inference error: {str(e)}")
    
    def process_results(self, result):
        """Process and display detection results"""
        try:
            print(f"Processing result: {type(result)}")
            
            # Check if there are any detections
            boxes = result.boxes
            keypoints = getattr(result, 'keypoints', None)
            draw_image = self.original_image.copy()
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(draw_image)
            skeleton = [
                (0,1), (0,2),      # nose -> eyes
                (1,3), (2,4),     # eyes -> ears
                (1,2),            # left eye <-> right eye
                (3,5), (4,6),     # ears <-> shoulders
                (5,6),            # shoulders
                (5,7), (7,9),     # left arm
                (6,8), (8,10),    # right arm
                (5,11), (6,12),   # shoulders -> hips
                (11,12),          # hips
                (11,13), (13,15), # left leg
                (12,14), (14,16)  # right leg
            ]
            # Draw bounding boxes and labels
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    cls = int(box.cls[0]) if box.cls is not None else -1
                    class_name = result.names[cls] if result.names and cls in result.names else f"Class_{cls}"
                    draw.rectangle([x1, y1, x2, y2], outline='lime', width=3)
                    label = f"{class_name} {conf:.2f}"
                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    draw.rectangle([x1, y1-text_height-4, x1+text_width+8, y1], fill='lime')
                    draw.text((x1+4, y1-text_height-2), label, fill='black', font=font)
                # Draw keypoints and skeleton
                if keypoints is not None and hasattr(keypoints, 'xy'):
                    kpts_xy = keypoints.xy.cpu().numpy() # shape: (num_person, 17, 2)
                    for person in kpts_xy:
                        # Draw keypoints
                        for idx, (x, y) in enumerate(person):
                            draw.ellipse((x-3, y-3, x+3, y+3), fill='red')
                        # Draw skeleton
                        for a, b in skeleton:
                            if a < len(person) and b < len(person):
                                x1, y1 = person[a]
                                x2, y2 = person[b]
                                draw.line((x1, y1, x2, y2), fill='yellow', width=2)
                self.current_image = draw_image
                self.display_image(draw_image)
                self.update_status(f"Detection complete: {len(boxes)} signs found (conf≥{self.confidence_threshold}), pose drawn")
            else:
                print("No object boxes detected")
                self.current_image = self.original_image.copy()
                self.display_image(self.current_image)
                self.update_status(f"No traffic signs detected (conf≥{self.confidence_threshold})")
            
        except Exception as e:
            print(f"Process results error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to process results:\n{str(e)}")
            self.update_status(f"Processing error: {str(e)}")
    
    def save_result(self):
        """Save current image or processed video"""
        if not self.current_image and not self.processed_video_path:
            messagebox.showwarning("Warning", "No image or processed video to save!")
            return
        
        if self.current_image:
            # Save image
            file_path = filedialog.asksaveasfilename(
                title="Save Detection Result",
                defaultextension=".jpg",
                filetypes=[
                    ("JPEG files", "*.jpg"),
                    ("PNG files", "*.png"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                try:
                    self.current_image.save(file_path)
                    messagebox.showinfo("Success", f"Image saved successfully:\n{file_path}")
                    self.update_status(f"Image saved: {os.path.basename(file_path)}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")
        
        elif self.processed_video_path:
            # Show info about already processed video
            messagebox.showinfo("Video Info", f"Processed video already saved at:\n{self.processed_video_path}")
            self.update_status(f"Video location: {os.path.basename(self.processed_video_path)}")
    

    
    def start_video(self):
        """Start video playback with real-time detection using PyAV"""
        try:
            if not self.current_video_path or not os.path.exists(self.current_video_path):
                messagebox.showerror("Error", "Video file not found. Please select a valid video file.")
                return

            # Open video with PyAV
            self.video_container = av.open(self.current_video_path)
            self.video_stream = self.video_container.streams.video[0]

            # Enable multithreaded decoding for speed
            self.video_stream.thread_type = 'AUTO'

            fps = float(self.video_stream.average_rate) if self.video_stream.average_rate else 30.0
            self.video_source_fps = fps
            self.video_total_frames = self.video_stream.frames or 0
            width = self.video_stream.codec_context.width
            height = self.video_stream.codec_context.height

            # Estimate frame count from duration if not available
            if self.video_total_frames <= 0 and self.video_stream.duration:
                duration_s = float(self.video_stream.duration * self.video_stream.time_base)
                self.video_total_frames = int(duration_s * fps)

            # Create frame iterator (decode only video stream)
            self.video_frame_iter = self.video_container.decode(video=0)

            print(f"PyAV opened: ~{self.video_total_frames} frames, {fps:.1f} FPS, {width}x{height}")
            print(f"Codec: {self.video_stream.codec_context.name}, Thread type: {self.video_stream.thread_type}")

            self.video_frame_count = 0
            self.detection_cache = {}
            self.video_playing = True
            self.video_button.config(text="Stop Video", bg='#e74c3c')

            # Reset FPS tracking
            self.fps_start_time = time.time()
            self.fps_frame_count = 0
            self.current_fps = 0.0

            self.play_video_frame()

        except Exception as e:
            self._cleanup_video_container()
            messagebox.showerror("Error", f"Failed to start video playback:\n{str(e)}")
            self.update_status(f"Video error: {str(e)}")

    def _cleanup_video_container(self):
        """Safely close PyAV container and reset state"""
        self.video_frame_iter = None
        self.video_stream = None
        if self.video_container:
            try:
                self.video_container.close()
            except:
                pass
            self.video_container = None
    
    def _run_detection(self, frame, scale_x, scale_y):
        """Run YOLO detection in background thread.
        Pre-convert tensors to numpy so the main thread doesn't pay .cpu() cost each frame."""
        try:
            predict_kwargs = {'conf': self.confidence_threshold, 'verbose': False}
            if self.device:
                predict_kwargs['device'] = self.device
            results = self.model(frame, **predict_kwargs)
            
            # Pre-extract detection data as plain Python/numpy to avoid
            # repeated .cpu().numpy() calls on every display frame.
            cached_boxes = []
            if results and len(results) > 0:
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1
                            class_name = (
                                result.names[cls]
                                if result.names and cls in result.names
                                else f"Class_{cls}"
                            )
                            if conf >= self.confidence_threshold:
                                cached_boxes.append({
                                    'xyxy': xyxy,
                                    'conf': conf,
                                    'class_name': class_name,
                                })
            self.detection_cache = {
                'boxes': cached_boxes,
                'scale_x': scale_x,
                'scale_y': scale_y,
            }
        except Exception as e:
            print(f"Detection error: {e}")

    def play_video_frame(self):
        """Play single video frame with detection (PyAV decode → RGB directly)"""
        if not self.video_playing or not self.video_frame_iter:
            return

        frame_start = time.time()

        try:
            # Decode next frame with PyAV (directly to RGB — no cvtColor needed!)
            try:
                av_frame = next(self.video_frame_iter)
            except StopIteration:
                self.stop_video()
                return

            # PyAV converts directly to RGB numpy array — faster than OpenCV BGR→RGB
            frame_rgb = av_frame.to_ndarray(format='rgb24')

            self.video_frame_count += 1

            # FPS calculation
            self.fps_frame_count += 1
            if self.fps_frame_count >= self.fps_update_interval:
                elapsed = time.time() - self.fps_start_time
                if elapsed > 0:
                    self.current_fps = self.fps_frame_count / elapsed
                self.fps_start_time = time.time()
                self.fps_frame_count = 0

            # Update status (throttled)
            if self.video_frame_count % 15 == 0 and self.video_total_frames > 0:
                progress = (self.video_frame_count / self.video_total_frames) * 100
                self.status_bar.config(
                    text=f"Playing: Frame {self.video_frame_count}/{self.video_total_frames} ({progress:.1f}%) "
                    f"| Playback: {self.current_fps:.1f} FPS (Original: {self.video_source_fps:.1f})"
                )

            # Trigger background detection every frame_skip frames
            if self.video_frame_count % self.frame_skip == 0:
                if self._detection_thread is None or not self._detection_thread.is_alive():
                    if self.fast_mode:
                        h, w = frame_rgb.shape[:2]
                        scale = min(800 / w, 800 / h)
                        new_w, new_h = int(w * scale), int(h * scale)
                        det_frame = cv2.resize(frame_rgb, (new_w, new_h))
                        sx, sy = w / new_w, h / new_h
                    else:
                        det_frame = frame_rgb
                        sx, sy = 1.0, 1.0
                    self._detection_thread = threading.Thread(
                        target=self._run_detection, args=(det_frame, sx, sy), daemon=True)
                    self._detection_thread.start()

            # Draw only keypoints and skeleton
            display_frame = frame_rgb.copy()
            # Draw keypoints and skeleton
            # Lấy keypoints từ detection thread nếu có
            skeleton_groups = [
                ([(0,1),(0,2),(1,2)], (255,215,0)), # head (vàng)
                ([(1,3),(2,4)], (255,165,0)),       # eyes-ears (cam)
                ([(3,5),(4,6)], (255,165,0)),       # ears-shoulders (cam)
                ([(5,6)], (0,255,0)),               # shoulders (xanh lá)
                ([(5,7),(7,9)], (0,191,255)),       # left arm (xanh dương)
                ([(6,8),(8,10)], (255,0,255)),      # right arm (hồng)
                ([(5,11),(6,12),(11,12)], (160,82,45)), # body/hips (nâu)
                ([(11,13),(13,15)], (255,0,0)),     # left leg (đỏ)
                ([(12,14),(14,16)], (0,128,0)),     # right leg (xanh lá đậm)
            ]
            if hasattr(self, '_detection_thread') and self._detection_thread is not None:
                try:
                    results = self.model(frame_rgb, conf=self.confidence_threshold, verbose=False)
                    if results and len(results) > 0:
                        keypoints = getattr(results[0], 'keypoints', None)
                        if keypoints is not None and hasattr(keypoints, 'xy'):
                            kpts_xy = keypoints.xy.cpu().numpy()
                            # Scale keypoints to display frame if resized
                            h_orig, w_orig = frame_rgb.shape[:2]
                            h_disp, w_disp = display_frame.shape[:2]
                            scale_x = w_disp / w_orig
                            scale_y = h_disp / h_orig
                            for person in kpts_xy:
                                for idx, (x, y) in enumerate(person):
                                    x_disp = int(x * scale_x)
                                    y_disp = int(y * scale_y)
                                    cv2.circle(display_frame, (x_disp, y_disp), 8, (255,34,34), -1)
                                for group, color in skeleton_groups:
                                    for a, b in group:
                                        if a < len(person) and b < len(person):
                                            x1, y1 = person[a]
                                            x2, y2 = person[b]
                                            x1_disp = int(x1 * scale_x)
                                            y1_disp = int(y1 * scale_y)
                                            x2_disp = int(x2 * scale_x)
                                            y2_disp = int(y2 * scale_y)
                                            cv2.line(display_frame, (x1_disp, y1_disp), (x2_disp, y2_disp), color, 6)
                except Exception:
                    pass

            # Pre-resize with cv2 (much faster than PIL thumbnail)
            h, w = display_frame.shape[:2]
            disp_w, disp_h = 600, 400
            scale_disp = min(disp_w / w, disp_h / h)
            if scale_disp < 1.0:
                new_dw, new_dh = int(w * scale_disp), int(h * scale_disp)
                display_frame = cv2.resize(display_frame, (new_dw, new_dh), interpolation=cv2.INTER_AREA)

            pil_image = Image.fromarray(display_frame)
            self.display_image(pil_image, fast=True)
            self.current_image = pil_image
            self.original_image = pil_image

            # Adaptive delay: subtract processing time from target interval
            processing_ms = (time.time() - frame_start) * 1000
            target_interval = 1000.0 / min(60, self.video_source_fps)
            delay = max(1, int(target_interval - processing_ms))
            self.video_job = self.root.after(delay, self.play_video_frame)

        except Exception as e:
            print(f"Video frame error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.stop_video()
    
    def stop_video(self):
        """Stop video playback"""
        self.video_playing = False
        
        if self.video_job:
            self.root.after_cancel(self.video_job)
            self.video_job = None
        
        # Close PyAV container
        self._cleanup_video_container()
        
        # Reset FPS tracking
        self.fps_start_time = None
        self.fps_frame_count = 0
        self.current_fps = 0.0
            
        self.video_button.config(text="Select & Play Video", bg='#3498db')
        self.update_status("Video playback stopped")
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def on_closing(self):
        """Handle application closing - cleanup camera and video"""
        if self.camera_running:
            self.stop_camera()
        if self.video_playing:
            self.stop_video()
        self.root.destroy()

def main():
    """Main application entry point"""
    try:
        root = tk.Tk()
        app = TrafficSignDetectorGUI(root)
        
        # Handle window closing properly (cleanup camera)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Center window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        root.mainloop()
        
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Fatal Error", f"Application failed to start:\n{str(e)}")

if __name__ == "__main__":
    main()