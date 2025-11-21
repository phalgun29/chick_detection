import gradio as gr
from ultralytics import YOLO
from PIL import Image
import os

# --- SETUP ---
MODEL_PATH = 'yolov8m_production_ready.pt'

# Check if the model file exists before loading
if not os.path.exists(MODEL_PATH):
    print(f"FATAL ERROR: Model file not found at '{MODEL_PATH}'")
    print("Please make sure your trained model file is in the same folder as this script.")
    exit()

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- BACKEND LOGIC ---
def analyze_chick_image(input_image: Image.Image):
    """
    Analyzes an input image of chicks using two confidence thresholds and provides clean visualization.
    """
    if input_image is None:
        return None, "Please upload an image first."

    # Run prediction with the lower threshold (30%)
    results_low_conf = model(input_image, conf=0.30)
    count_low_conf = len(results_low_conf[0].boxes)
    
    annotated_image = results_low_conf[0].plot(labels=False, conf=False, line_width=2)
    
    # Convert the annotated image from NumPy array (BGR) to PIL Image (RGB) for display
    annotated_image_pil = Image.fromarray(annotated_image[..., ::-1])

    # Run prediction with the higher threshold (80%)
    results_high_conf = model(input_image, conf=0.80)
    count_high_conf = len(results_high_conf[0].boxes)
    
    # Create the result string to display a range
    result_text = (
        f"Confidence Range: There are {count_high_conf} - {count_low_conf} Chicks in the above image.\n\n"
        f"This means the model is highly confident (>80%) about {count_high_conf} chicks, "
        f"but detects up to {count_low_conf} chicks if we include detections it is less certain about (>30%)."
    )

    return annotated_image_pil, result_text

# --- FRONTEND UI ---
iface = gr.Interface(
    fn=analyze_chick_image,
    inputs=gr.Image(type="pil", label="Upload Chick Image"),
    outputs=[
        gr.Image(type="pil", label="Annotated Image "),
        gr.Textbox(label="Analysis Results", lines=5)
    ],
    title="AI Chick Counter ",
    description="",
    allow_flagging="never"
)

# Launch the application
if __name__ == "__main__":
    iface.launch()