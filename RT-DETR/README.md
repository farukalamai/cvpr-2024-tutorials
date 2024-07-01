# Object Detection using RT-DETR

This script provides a tool for detecting and annotating objects in video files using a pre-trained RT-DETR model from Hugging Face's `transformers` library. The output video displays detected objects with bounding boxes, masks, and labels.

## Installation

First, ensure you have Python installed on your system. Then, install the necessary Python libraries using pip:

```bash
pip install torch numpy Pillow tqdm transformers supervision
```

## Usage

1. **Prepare your environment:**
   Make sure all dependencies are installed as mentioned in the installation section.

2. **Set the video path:**
   Open the script and edit the `input_video_path` variable to point to the path of your video file.

3. **Run the script:**
   Execute the script in your Python environment:

   ```bash
   python path_to_script.py

## Acknowledgments

This project utilizes resources and inspiration from the RT-DETR model demo available on Hugging Face Spaces, specifically from [RT-DETR-tracking-coco](https://huggingface.co/spaces/merve/RT-DETR-tracking-coco).
