import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torchvision import transforms
from PIL import Image
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                            QLabel, QFileDialog, QWidget, QTabWidget, QProgressBar, QMessageBox,
                            QSpinBox, QComboBox, QCheckBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

# Model Definition
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm_1 = nn.BatchNorm2d(256)
        self.norm_2 = nn.BatchNorm2d(256)

    def forward(self, x):
        output = self.norm_2(self.conv_2(F.relu(self.norm_1(self.conv_1(x)))))
        return output + x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.norm_1 = nn.BatchNorm2d(64)

        # down-convolution #
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.norm_2 = nn.BatchNorm2d(128)

        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.norm_3 = nn.BatchNorm2d(256)

        # residual blocks #
        residualBlocks = []
        for l in range(8):
            residualBlocks.append(ResidualBlock())
        self.res = nn.Sequential(*residualBlocks)

        # up-convolution #
        self.conv_6 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_7 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.norm_4 = nn.BatchNorm2d(128)

        self.conv_8 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_9 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm_5 = nn.BatchNorm2d(64)

        self.conv_10 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = F.relu(self.norm_1(self.conv_1(x)))
        x = F.relu(self.norm_2(self.conv_3(self.conv_2(x))))
        x = F.relu(self.norm_3(self.conv_5(self.conv_4(x))))
        x = self.res(x)
        x = F.relu(self.norm_4(self.conv_7(self.conv_6(x))))
        x = F.relu(self.norm_5(self.conv_9(self.conv_8(x))))
        x = self.conv_10(x)
        x = sigmoid(x)
        return x

# Image Processing Worker Thread
class ImageProcessor(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, generator, image_path, size=256):
        super().__init__()
        self.generator = generator
        self.image_path = image_path
        self.size = size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def preprocess_image(self):
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img = Image.open(self.image_path).convert('RGB')
        return transform(img).unsqueeze(0).to(self.device)

    def postprocess_image(self, tensor):
        tensor = (tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy() + 1) / 2.0
        return np.clip(tensor * 255, 0, 255).astype(np.uint8)

    def run(self):
        try:
            self.update_progress.emit(10)
            input_tensor = self.preprocess_image()
            self.update_progress.emit(40)
            
            with torch.no_grad():
                output_tensor = self.generator(input_tensor)
                
            self.update_progress.emit(80)
            output_img = self.postprocess_image(output_tensor)
            self.update_progress.emit(100)
            self.finished.emit(output_img)
        except Exception as e:
            self.error.emit(str(e))

# Video Processing Worker Thread
class VideoProcessor(QThread):
    update_progress = pyqtSignal(int)
    update_status = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, generator, video_path, output_path, temp_dir="temp", size=256, fps=None):
        super().__init__()
        self.generator = generator
        self.video_path = video_path
        self.output_path = output_path
        self.temp_dir = temp_dir
        self.size = size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.frames_dir = os.path.join(temp_dir, "frames")
        self.cartoon_frames_dir = os.path.join(temp_dir, "cartoon_frames")
        
        # Get original video fps
        cap = cv2.VideoCapture(video_path)
        self.original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Use specified fps if provided, otherwise use original fps
        self.fps = fps if fps else self.original_fps

    def preprocess_image(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img = Image.open(image_path).convert('RGB')
        return transform(img).unsqueeze(0).to(self.device)

    def postprocess_image(self, tensor):
        tensor = (tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy() + 1) / 2.0
        return np.clip(tensor * 255, 0, 255).astype(np.uint8)

    def extract_frames(self):
        os.makedirs(self.frames_dir, exist_ok=True)
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (self.size, self.size))
            frame_path = os.path.join(self.frames_dir, f"frame_{count:04d}.png")
            cv2.imwrite(frame_path, frame)
            count += 1
            
            # Update progress
            progress = int(count / total_frames * 30)  # Frame extraction is 30% of total progress
            self.update_progress.emit(progress)
            self.update_status.emit(f"Extracting frames: {count}/{total_frames}")
            
        cap.release()
        return count

    def cartoonize_frames(self, frame_count):
        os.makedirs(self.cartoon_frames_dir, exist_ok=True)
        frame_files = sorted([f for f in os.listdir(self.frames_dir) if f.endswith('.png')])
        
        for i, filename in enumerate(frame_files):
            frame_path = os.path.join(self.frames_dir, filename)
            input_tensor = self.preprocess_image(frame_path)
            
            with torch.no_grad():
                output_tensor = self.generator(input_tensor)
                
            output_img = self.postprocess_image(output_tensor)
            output_img = Image.fromarray(output_img)
            output_img.save(os.path.join(self.cartoon_frames_dir, filename))
            
            # Update progress
            progress = 30 + int((i + 1) / frame_count * 60)  # Cartoonizing frames is 60% of total progress
            self.update_progress.emit(progress)
            self.update_status.emit(f"Cartoonizing: {i+1}/{frame_count}")

    def frames_to_video(self):
        frame_files = sorted([f for f in os.listdir(self.cartoon_frames_dir) if f.endswith('.png')])
        
        if not frame_files:
            raise ValueError("No frame files found.")
            
        sample_frame = cv2.imread(os.path.join(self.cartoon_frames_dir, frame_files[0]))
        height, width, layers = sample_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))

        for i, filename in enumerate(frame_files):
            frame_path = os.path.join(self.cartoon_frames_dir, filename)
            frame = cv2.imread(frame_path)
            out.write(frame)
            
            # Update progress
            progress = 90 + int((i + 1) / len(frame_files) * 10)  # Video creation is 10% of total progress
            self.update_progress.emit(progress)
            self.update_status.emit(f"Creating video: {i+1}/{len(frame_files)}")

        out.release()

    def run(self):
        try:
            # Create temp directories
            os.makedirs(self.temp_dir, exist_ok=True)
            os.makedirs(self.frames_dir, exist_ok=True)
            os.makedirs(self.cartoon_frames_dir, exist_ok=True)
            
            # Extract frames
            self.update_status.emit("Starting frame extraction...")
            frame_count = self.extract_frames()
            
            # Cartoonize frames
            self.update_status.emit("Starting cartoonization...")
            self.cartoonize_frames(frame_count)
            
            # Create video
            self.update_status.emit("Creating output video...")
            self.frames_to_video()
            
            # Complete
            self.update_progress.emit(100)
            self.update_status.emit("Processing complete!")
            self.finished.emit(self.output_path)
            
        except Exception as e:
            self.error.emit(str(e))

# Main Window Class
class CartoonApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image/Video Cartoonizer")
        self.setMinimumSize(1000, 600)
        
        # Detect device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_label = QLabel(f"Current device: {self.device}")
        
        # Initialize model
        self.generator = None
        self.model_loaded = False
        
        # Create UI
        self.init_ui()
        
    def init_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Add tabs
        tabs = QTabWidget()
        image_tab = QWidget()
        video_tab = QWidget()
        tabs.addTab(image_tab, "Image Cartoonizer")
        tabs.addTab(video_tab, "Video Cartoonizer")
        
        # Image tab layout
        image_layout = QVBoxLayout(image_tab)
        img_control_layout = QHBoxLayout()
        
        # Image controls
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        self.select_image_btn = QPushButton("Select Image")
        self.select_image_btn.clicked.connect(self.select_image)
        self.process_image_btn = QPushButton("Cartoonize")
        self.process_image_btn.clicked.connect(self.process_image)
        self.save_image_btn = QPushButton("Save Image")
        self.save_image_btn.clicked.connect(self.save_image)
        
        # Image processing parameters
        img_size_label = QLabel("Image Size:")
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(128, 1024)
        self.img_size_spin.setValue(256)
        self.img_size_spin.setSingleStep(32)
        
        # Add widgets to image control layout
        img_control_layout.addWidget(self.load_model_btn)
        img_control_layout.addWidget(self.select_image_btn)
        img_control_layout.addWidget(self.process_image_btn)
        img_control_layout.addWidget(self.save_image_btn)
        img_control_layout.addWidget(img_size_label)
        img_control_layout.addWidget(self.img_size_spin)
        
        # Image display area
        img_display_layout = QHBoxLayout()
        self.original_img_label = QLabel("Original Image")
        self.original_img_label.setAlignment(Qt.AlignCenter)
        self.original_img_label.setMinimumSize(400, 400)
        self.original_img_label.setStyleSheet("border: 1px solid #cccccc;")
        
        self.cartoon_img_label = QLabel("Cartoon Image")
        self.cartoon_img_label.setAlignment(Qt.AlignCenter)
        self.cartoon_img_label.setMinimumSize(400, 400)
        self.cartoon_img_label.setStyleSheet("border: 1px solid #cccccc;")
        
        img_display_layout.addWidget(self.original_img_label)
        img_display_layout.addWidget(self.cartoon_img_label)
        
        # Image processing progress bar
        self.img_progress_bar = QProgressBar()
        self.img_progress_bar.setValue(0)
        
        # Add all widgets to image tab
        image_layout.addLayout(img_control_layout)
        image_layout.addLayout(img_display_layout)
        image_layout.addWidget(self.img_progress_bar)
        
        # Video tab layout
        video_layout = QVBoxLayout(video_tab)
        video_control_layout = QHBoxLayout()
        
        # Video controls
        self.select_video_btn = QPushButton("Select Video")
        self.select_video_btn.clicked.connect(self.select_video)
        self.process_video_btn = QPushButton("Cartoonize")
        self.process_video_btn.clicked.connect(self.process_video)
        
        # Video processing parameters
        video_size_label = QLabel("Video Resolution:")
        self.video_size_spin = QSpinBox()
        self.video_size_spin.setRange(128, 1024)
        self.video_size_spin.setValue(256)
        self.video_size_spin.setSingleStep(32)
        
        fps_label = QLabel("Output FPS:")
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(24)
        self.use_original_fps = QCheckBox("Use Original FPS")
        self.use_original_fps.setChecked(True)
        self.fps_spin.setEnabled(False)
        self.use_original_fps.stateChanged.connect(self.toggle_fps)
        
        # Add widgets to video control layout
        video_control_layout.addWidget(self.select_video_btn)
        video_control_layout.addWidget(self.process_video_btn)
        video_control_layout.addWidget(video_size_label)
        video_control_layout.addWidget(self.video_size_spin)
        video_control_layout.addWidget(fps_label)
        video_control_layout.addWidget(self.fps_spin)
        video_control_layout.addWidget(self.use_original_fps)
        
        # Video processing status and progress
        self.video_path_label = QLabel("No video selected")
        self.video_status_label = QLabel("Status: Not started")
        self.video_progress_bar = QProgressBar()
        self.video_progress_bar.setValue(0)
        
        # Add all widgets to video tab
        video_layout.addLayout(video_control_layout)
        video_layout.addWidget(self.video_path_label)
        video_layout.addWidget(self.video_status_label)
        video_layout.addWidget(self.video_progress_bar)
        
        # Add tabs to main layout
        main_layout.addWidget(self.device_label)
        main_layout.addWidget(tabs)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Initial state
        self.image_path = None
        self.video_path = None
        self.cartoon_result = None
        self.process_image_btn.setEnabled(False)
        self.save_image_btn.setEnabled(False)
        self.process_video_btn.setEnabled(False)
    
    def toggle_fps(self, state):
        """Enable or disable FPS spinbox"""
        self.fps_spin.setEnabled(not state)
    
    def load_model(self):
        """Load model file"""
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.pth)")
        if model_path:
            try:
                # Show loading progress
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self.load_model_btn.setText("Loading...")
                self.load_model_btn.setEnabled(False)
                
                # Initialize model
                self.generator = Generator().to(self.device)
                checkpoint = torch.load(model_path, map_location=self.device)
                self.generator.load_state_dict(checkpoint)
                self.generator.eval()
                
                self.model_loaded = True
                self.load_model_btn.setText("Model Loaded")
                
                # Update button states
                if self.image_path:
                    self.process_image_btn.setEnabled(True)
                if self.video_path:
                    self.process_video_btn.setEnabled(True)
                    
                QApplication.restoreOverrideCursor()
            except Exception as e:
                QApplication.restoreOverrideCursor()
                self.load_model_btn.setText("Load Model")
                self.load_model_btn.setEnabled(True)
                QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
    
    def select_image(self):
        """Select image file"""
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if image_path:
            self.image_path = image_path
            pixmap = QPixmap(image_path)
            
            if not pixmap.isNull():
                # Resize image to fit the label
                scaled_pixmap = pixmap.scaled(
                    self.original_img_label.width(), 
                    self.original_img_label.height(),
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.original_img_label.setPixmap(scaled_pixmap)
                
                # Update button state
                if self.model_loaded:
                    self.process_image_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Warning", "Could not load the selected image.")
    
    def process_image(self):
        """Process image"""
        if not self.model_loaded:
            QMessageBox.warning(self, "Warning", "Please load a model first.")
            return
        
        if not self.image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return
        
        # Create and start processing thread
        size = self.img_size_spin.value()
        self.img_processor = ImageProcessor(self.generator, self.image_path, size)
        self.img_processor.update_progress.connect(self.img_progress_bar.setValue)
        self.img_processor.finished.connect(self.display_cartoon_image)
        self.img_processor.error.connect(self.show_error)
        
        # Update UI state
        self.process_image_btn.setEnabled(False)
        self.img_progress_bar.setValue(0)
        
        # Start processing thread
        self.img_processor.start()
    
    def display_cartoon_image(self, img_array):
        """Display processed cartoon image"""
        # Directly use OpenCV to process the image
        # Ensure image data is contiguous
        if not img_array.flags['C_CONTIGUOUS']:
            img_array = np.ascontiguousarray(img_array)
        
        # Convert RGB to BGR (format expected by Qt)
        bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Create QImage
        height, width, channel = bgr_img.shape
        bytes_per_line = channel * width
        
        q_img = QImage(bgr_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Resize image to fit the label
        scaled_pixmap = pixmap.scaled(
            self.cartoon_img_label.width(), 
            self.cartoon_img_label.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.cartoon_img_label.setPixmap(scaled_pixmap)
        
        # Save result for later use
        self.cartoon_result = img_array
        
        # Update button states
        self.process_image_btn.setEnabled(True)
        self.save_image_btn.setEnabled(True)
        
    def save_image(self):
        """Save cartoon image"""
        if self.cartoon_result is None:
            QMessageBox.warning(self, "Warning", "No cartoon image to save.")
            return
        
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Cartoon Image", "", "Image Files (*.png *.jpg)")
        if save_path:
            try:
                Image.fromarray(self.cartoon_result).save(save_path)
                QMessageBox.information(self, "Success", f"Image saved to: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
    
    def select_video(self):
        """Select video file"""
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if video_path:
            self.video_path = video_path
            self.video_path_label.setText(f"Selected video: {os.path.basename(video_path)}")
            
            # Update button state
            if self.model_loaded:
                self.process_video_btn.setEnabled(True)
    
    def process_video(self):
        """Process video"""
        if not self.model_loaded:
            QMessageBox.warning(self, "Warning", "Please load a model first.")
            return
        
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "Please select a video first.")
            return
        
        # Select save path
        save_dir = os.path.dirname(self.video_path)
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        suggested_name = f"{base_name}_cartoon.mp4"
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Cartoon Video", 
            os.path.join(save_dir, suggested_name),
            "Video Files (*.mp4)"
        )
        
        if not output_path:
            return
        
        # Create temp directory
        temp_dir = "temp_video_process"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Create and start processing thread
        size = self.video_size_spin.value()
        fps = None if self.use_original_fps.isChecked() else self.fps_spin.value()
        
        self.video_processor = VideoProcessor(
            self.generator, 
            self.video_path, 
            output_path, 
            temp_dir=temp_dir,
            size=size,
            fps=fps
        )
        
        self.video_processor.update_progress.connect(self.video_progress_bar.setValue)
        self.video_processor.update_status.connect(self.video_status_label.setText)
        self.video_processor.finished.connect(self.video_processing_done)
        self.video_processor.error.connect(self.show_error)
        
        # Update UI state
        self.process_video_btn.setEnabled(False)
        self.video_progress_bar.setValue(0)
        self.video_status_label.setText("Status: Ready to process...")
        
        # Start processing thread
        self.video_processor.start()
    
    def video_processing_done(self, output_path):
        """Callback when video processing is complete"""
        self.process_video_btn.setEnabled(True)
        QMessageBox.information(self, "Success", f"Video processing complete, saved to:\n{output_path}")
        
        # Open output folder
        if os.path.exists(output_path):
            if sys.platform == 'win32':
                os.startfile(os.path.dirname(output_path))
            elif sys.platform == 'darwin':  # macOS
                import subprocess
                subprocess.Popen(['open', os.path.dirname(output_path)])
            else:  # Linux
                import subprocess
                subprocess.Popen(['xdg-open', os.path.dirname(output_path)])
    
    def show_error(self, error_msg):
        """Display error message"""
        QMessageBox.critical(self, "Error", error_msg)
        self.process_image_btn.setEnabled(self.image_path is not None and self.model_loaded)
        self.process_video_btn.setEnabled(self.video_path is not None and self.model_loaded)
    
    def closeEvent(self, event):
        """Clean up temporary files when closing the window"""
        temp_dirs = ["temp_video_process"]
        for dir_path in temp_dirs:
            if os.path.exists(dir_path):
                try:
                    import shutil
                    shutil.rmtree(dir_path)
                except:
                    pass
        event.accept()

# Main Function
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for consistency across all platforms
    
    # Set application stylesheet
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QPushButton {
            background-color: #4a86e8;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #3a76d8;
        }
        QPushButton:pressed {
            background-color: #2a66c8;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #888888;
        }
        QProgressBar {
            border: 1px solid #cccccc;
            border-radius: 4px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4a86e8;
            width: 10px;
        }
        QLabel {
            color: #333333;
        }
    """)
    
    window = CartoonApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()