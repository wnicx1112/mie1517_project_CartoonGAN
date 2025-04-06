# Cartoonizer App - User Guide

This is a PyTorch-based application that transforms regular photos and videos into cartoon style.

## Features

- Image cartoonization
- Video cartoonization
- Adjustable processing resolution
- Customizable video frame rate
- User-friendly graphical interface

## Installation Requirements

Before running this software, please ensure you have installed the following dependencies:

```bash
pip install -r requirements.txt
```

## How to Use

### Launch the Application

```bash
python cartoonizer_app.py
```

### Load the Model

1. Click the "Load Model" button
2. Select the cartoonization model file (.pth file)

### Image Cartoonization

1. Switch to the "Image Cartoonizer" tab
2. Click "Select Image" to choose an image to process
3. Adjust the "Image Size" parameter (default is 256)
4. Click "Cartoonize" to begin processing
5. After processing completes, click "Save Image" to save the result

### Video Cartoonization

1. Switch to the "Video Cartoonizer" tab
2. Click "Select Video" to choose a video to process
3. Adjust "Video Resolution" parameter (default is 256)
4. Choose whether to use the original frame rate or set a custom one
5. Click "Cartoonize" to begin processing
6. Select output video save location
7. Wait for processing to complete

## Important Notes

- Video processing may take a long time, please be patient
- Higher image/video resolutions will require more memory and processing time
- Using CUDA acceleration is recommended (requires an NVIDIA GPU)
- Temporary files will be created during video processing and automatically cleaned up when the application closes

## Troubleshooting

**Q: Why is my program running slow?**  
A: If your computer doesn't have a CUDA-compatible GPU, the program will use CPU for processing, which is significantly slower. Try reducing the image/video resolution, or use a computer with an NVIDIA GPU.

**Q: Why does my processed video have no sound?**  
A: The current version does not support preserving the audio track from the original video. This feature will be added in future versions.

**Q: My video processing failed, what should I do?**  
A: Check if your video format is supported. The current version supports MP4, AVI, MOV, and MKV formats. If it still fails, try converting your video to MP4 format using another tool before processing.

## Model Information

The cartoonization model used in this software is based on a Generative Adversarial Network (GAN) with the following structure:

- Generator network: U-Net architecture with residual blocks
- Downsampling and upsampling layers
- Batch normalization and ReLU activation functions
- Sigmoid output layer

The model was trained on a large dataset of real photos and cartoon images.

## Version History

- Version 1.0.0: Initial release
  - Image cartoonization support
  - Video cartoonization support
  - Basic configuration options

## License

This software is provided under the MIT License.