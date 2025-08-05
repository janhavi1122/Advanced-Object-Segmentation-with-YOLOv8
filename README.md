# Advanced Object Segmentation with YOLOv8 and Streamlit

🎨 This application utilizes a pre-trained **YOLOv8 Segmentation model** (`yolov8s-seg.pt`) from the Ultralytics library to perform instance segmentation on uploaded images.

## Features

- Upload JPG, PNG, BMP, and WEBP image files.
- Automatically detects various objects including fruits, vegetables, people, animals, and household items based on the COCO dataset.
- Detects fruits and vegetables and highlights them with green and orange polygons respectively.
- Draws pixel-perfect segmentation polygons with confidence scores around each detected object.
- Adjustable confidence threshold for fine-tuning detection results.
- Clean and intuitive UI built with Streamlit, enhanced with custom CSS styling.

## How It Works

1. Upload an image.
2. The YOLOv8 segmentation model processes the image and detects objects.
3. Polygons are drawn around detected objects with labels and confidence scores.
4. Results are displayed interactively with detailed categorized lists of detected fruits, vegetables, and other objects.

## Installation

1. Clone this repository:


Open the provided localhost URL in your browser, upload an image, and enjoy real-time instance segmentation with object classification.

## Dependencies

- [Streamlit](https://streamlit.io/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [OpenCV](https://opencv.org/)
- [Pillow (PIL)](https://python-pillow.org/)

## Notes

- The model is trained on the COCO dataset and can detect common object classes.
- The app classifies fruits and vegetables with special color coding for easier visualization.
- Adjust confidence threshold slider to filter detections as needed.

## Developed By

**Janhavi** ❤️

---

Feel free to contribute or open issues for bugs and feature requests!

