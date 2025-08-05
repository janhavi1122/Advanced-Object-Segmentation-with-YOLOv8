import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Added ImageDraw, ImageFont for potentially richer text if needed
from ultralytics import YOLO
import random # For random colors for polygons
import os # For checking model existence

# --- Configuration Constants ---
# Using the nano segmentation model for efficiency, good for demos
YOLO_MODEL_PATH = "yolov8n-seg.pt" 

# --- Custom CSS for a more attractive interface ---
st.markdown(
    """
    <style>
    /* General page styling */
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        color: #333333; /* Darker text */
        font-family: 'Segoe UI', sans-serif;
    }

    /* Header styling */
    h1 {
        color: #2F80ED; /* Blue for main title */
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        margin-bottom: 0.5em;
        padding-top: 1rem;
    }

    /* Section titles */
    h2, h3, h4 {
        color: #333333; /* Darker gray for section titles */
        font-family: 'Segoe UI', sans-serif;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Sidebar styling */
    .st-emotion-cache-1ldf0s6 { /* Target the sidebar container, might change with Streamlit versions */
        background-color: #e0e2e6; /* Slightly darker gray sidebar */
        padding-top: 2rem;
    }
    .st-emotion-cache-1ldf0s6 h2 { /* Sidebar header */
        color: #333333;
        font-size: 1.4em;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #2F80ED; /* Blue button */
        color: white;
        border-radius: 8px; /* Slightly more rounded */
        border: none;
        padding: 0.6em 1.5em; /* Slightly larger padding */
        font-size: 1.05rem;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }
    .stButton>button:hover {
        background-color: #56CCF2; /* Lighter blue on hover */
        transform: translateY(-2px); /* Slight lift effect */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:active {
        background-color: #256bbd; /* Even darker on click */
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    /* File uploader styling */
    .stFileUploader label {
        color: #333333;
        font-size: 1.1em;
        font-weight: bold;
    }

    /* Image display styling */
    .stImage {
        border-radius: 8px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); /* More prominent shadow for images */
        transition: transform 0.3s ease;
    }
    .stImage:hover {
        transform: scale(1.01); /* Slight zoom on hover */
    }

    /* Info/Warning/Success messages */
    .stAlert {
        border-radius: 8px;
        font-size: 1.05em;
    }

    /* Markdown styling for horizontal rule */
    hr {
        border-top: 2px solid #ccc; /* Thicker, lighter gray rule */
        margin: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load YOLOv8 Segmentation Model ---
# Use st.cache_resource to load the model only once across reruns
@st.cache_resource
def load_segmentation_model(model_path):
    """
    Loads the YOLOv8 segmentation model.
    """
    try:
        if not os.path.exists(model_path):
            st.warning(f"Model weights '{model_path}' not found locally. Attempting to download...")
        model = YOLO(model_path)
        st.success(f"YOLOv8 segmentation model '{model_path}' loaded successfully! 🎉")
        return model
    except Exception as e:
        st.error(f"Failed to load YOLOv8 model: {e}")
        st.warning("Please ensure you have 'ultralytics' installed (`pip install ultralytics`) "
                   "and an active internet connection if downloading weights for the first time.")
        st.stop() # Stop the app if model loading fails
        return None

# Function to perform segmentation and draw polygons
def segment_image(image: Image.Image, model: YOLO, confidence_threshold: float):
    """
    Performs instance segmentation on an image using the YOLOv8 segmentation model
    and draws polygons around detected objects.

    Args:
        image (PIL.Image.Image): The input image.
        model (YOLO): The loaded YOLOv8 segmentation model.
        confidence_threshold (float): Minimum confidence score for object detections.

    Returns:
        tuple: A tuple containing:
            - PIL.Image.Image: The annotated image with polygons.
            - int: Number of objects detected.
            - list: List of strings describing each detected object.
    """
    img_cv = np.array(image.convert('RGB')) # Ensure RGB for consistent processing
    img_cv = img_cv[:, :, ::-1].copy() # Convert RGB to BGR for OpenCV operations

    # Perform inference with specified confidence and without verbose output
    results = model(img_cv, conf=confidence_threshold, verbose=False)

    img_annotated = img_cv.copy()
    overlay = img_annotated.copy() # Create an overlay for translucent polygons
    alpha = 0.4 # Transparency factor for the overlay (0.0 - 1.0)

    detected_objects_info = []
    num_objects = 0

    if results and results[0].masks is not None:
        for r in results: # Iterate through detection results for the image
            if r.masks is None: # Skip if no masks are found for this result object
                continue
            for mask, box in zip(r.masks.xy, r.boxes):
                # Get class name and confidence
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                # Generate a unique, random color for each object's polygon
                object_color = [random.randint(0, 255) for _ in range(3)] # BGR format for OpenCV

                # Convert mask points to integers and reshape for cv2.fillPoly and cv2.polylines
                # The mask.xy contains [x, y] pairs of the polygon vertices
                mask_polygon = np.int32([mask]) # Needs to be a list of arrays for fillPoly/polylines

                # Draw filled polygon (segmentation mask) on the overlay
                cv2.fillPoly(overlay, [mask_polygon], object_color)

                # Draw polygon outline on the main annotated image
                cv2.polylines(img_annotated, [mask_polygon], isClosed=True, color=object_color, thickness=2)

                # Get bounding box coordinates for text placement (not drawing the box itself)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Prepare text label
                label = f"{class_name} {confidence:.2f}"
                
                # Determine text color (white or black for contrast)
                text_color = (255, 255, 255) if sum(object_color) < 300 else (0, 0, 0) # Simple contrast heuristic

                # Add text to the image
                # Position text slightly above the top-left corner of the object
                text_x = x1
                text_y = y1 - 10 if y1 > 20 else y1 + 20 # Adjust if too close to top edge
                
                # Draw a text background rectangle for better readability (optional but good)
                # (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                # cv2.rectangle(img_annotated, (text_x, text_y - label_height - baseline), 
                #               (text_x + label_width, text_y + baseline), object_color, cv2.FILLED)
                # cv2.putText(img_annotated, label, (text_x, text_y),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
                
                detected_objects_info.append(
                    f"- **{class_name}** (Confidence: {confidence:.2f})"
                )
                num_objects += 1

        # Combine the original image with the transparent polygon overlay
        cv2.addWeighted(overlay, alpha, img_annotated, 1 - alpha, 0, img_annotated)
    
    # Convert the annotated image back to PIL Image for Streamlit display
    img_annotated_pil = Image.fromarray(img_annotated[:, :, ::-1]) # Convert BGR back to RGB

    return img_annotated_pil, num_objects, detected_objects_info

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Advanced Object Segmentation Demo",
    layout="wide", # Use wide layout for better display of images
    initial_sidebar_state="expanded"
)

st.title("🎨 Advanced Object Segmentation with YOLOv8")
st.markdown("### Upload an image to detect various objects and see them highlighted with precise polygons!")

# --- Sidebar for controls ---
with st.sidebar:
    st.header("📸 Image Upload")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Select an image for object segmentation:",
        type=["jpg", "jpeg", "png", "bmp", "webp"], # Expanded supported types
        help="Upload an image file (JPG, PNG, etc.) to analyze. Max file size 200MB."
    )
    st.markdown("---")
    
    st.header("⚙️ Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.30, # Slightly increased default for potentially clearer detections
        step=0.05,
        help="Adjust the minimum confidence score for object detections. Higher values mean fewer, "
             "but typically more accurate, detections. Lower values might pick up more objects, "
             "but also more false positives."
    )
    st.markdown("---")
    
    st.header("💡 About This App")
    st.info("""
        This application utilizes a pre-trained **YOLOv8 Segmentation model** (`yolov8n-seg.pt`)
        from the Ultralytics library to perform **instance segmentation**.

        **Instance segmentation** is a cutting-edge computer vision task that not only
        identifies objects (like object detection) but also generates a pixel-perfect
        mask (a polygon) for each individual instance of an object within an image.

        **Key Features:**
        - Upload an image (JPG, PNG, etc.).
        - YOLOv8 automatically detects various objects (e.g., people, cars, animals,
          household items - based on the COCO dataset the model is trained on).
        - Polygons are drawn around each detected object with a confidence score.
        - Adjustable confidence threshold for fine-tuning detections.

        This app is built with:
        - [Streamlit](https://streamlit.io/) for the interactive web interface.
        - [Ultralytics YOLOv8](https://docs.ultralytics.com/) for the deep learning model.
        - [OpenCV](https://opencv.org/) for image processing.
        - [Pillow (PIL)](https://python-pillow.org/) for image handling.
        
        *Developed with ❤️ in Python*
    """)
    st.markdown("---")
import datetime

st.caption(f"App version: 1.1.0 | Last updated: {datetime.date.today().strftime('%B %d, %Y')}")

# --- Main Content Area ---
if uploaded_file is not None:
    try:
        # Read the image from the uploaded file
        image = Image.open(uploaded_file).convert('RGB') # Ensure consistent RGB
        
        st.subheader("🖼️ Original Image")
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
        st.markdown("---")

        st.subheader("💡 Processing Image for Segmentation...")
        
        # Load the segmentation model
        # The spinner ensures the user knows something is happening, especially on first load
        with st.spinner("⏳ Loading YOLOv8 segmentation model... (This might take a moment on first run as weights download)"):
            yolo_model = load_segmentation_model(YOLO_MODEL_PATH)

        if yolo_model:
            # Perform segmentation
            with st.spinner("🚀 Running inference and drawing polygons..."):
                annotated_image, num_objects, objects_info = segment_image(image, yolo_model, confidence_threshold)

            if annotated_image:
                st.subheader("✅ Annotated Image with Polygons")
                st.image(annotated_image, caption=f"Annotated Image ({num_objects} objects detected)", use_container_width=True)
                st.markdown("---")

                st.subheader("📊 Detected Objects Summary")
                
                if num_objects > 0:
                    st.success(f"Successfully detected **{num_objects}** object(s)! 🎉")
                    
                    with st.expander("Click to view detailed list of detected objects"):
                        for i, info in enumerate(objects_info):
                            st.markdown(f"{i+1}. {info}")
                        if num_objects > 20: # Limit display for very crowded images
                            st.info(f"Only showing the first {len(objects_info)} objects. More may be present.")
                else:
                    st.warning("😕 No objects detected in this image with the current confidence threshold. "
                               "Try adjusting the 'Confidence Threshold' slider in the sidebar to a lower value!")
                    st.write("This could happen if objects are too small, blurry, or if the model's confidence "
                             "for detections falls below the set threshold.")
            else:
                st.error("An error occurred during image annotation. Please try again or with a different image.")
    
    except Exception as e:
        st.error(f"An unexpected error occurred while processing the image: {e}")
        st.info("Please ensure the uploaded file is a valid image format.")
else:
    st.info("Upload an image from the sidebar to begin object segmentation! ⬆️")
    # Display a placeholder image when no file is uploaded
    st.image("https://via.placeholder.com/800x500?text=Upload+Your+Image+Here", 
             caption="Waiting for an image...", 
             use_container_width=True,
             channels="RGB")

st.markdown("""
---
### How it Works: Behind the Scenes

This application leverages a powerful deep learning model called **YOLOv8** (You Only Look Once, version 8).
Specifically, it uses a **segmentation variant** of YOLOv8.

1.  **Image Upload:** You upload an image to the Streamlit application.
2.  **Model Loading:** A pre-trained YOLOv8 segmentation model (`yolov8n-seg.pt`) is loaded. This model has been trained on a vast dataset (like COCO) to recognize a wide variety of objects.
3.  **Inference:** The uploaded image is passed through the YOLOv8 model. The model processes the image to identify objects and, crucially, to generate a pixel-level mask (polygon) for each object.
4.  **Polygon Drawing:**
    * For each detected object, the model provides a set of coordinates that define its segmented boundary.
    * OpenCV's `cv2.fillPoly()` function is used to draw a semi-transparent, colored polygon over the detected object's area. This creates the filled-in "mask."
    * OpenCV's `cv2.polylines()` function is then used to draw a clear outline (border) around the same polygon, making the boundaries more distinct.
    * The object's class name and confidence score are also added as text labels.
5.  **Display:** The original image, now adorned with these colorful polygons and labels, is displayed back to you in the Streamlit interface.

This seamless process allows for efficient and accurate instance segmentation, providing more detailed object localization than simple bounding boxes.
""")