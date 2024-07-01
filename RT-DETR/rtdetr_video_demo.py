import os
import uuid
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import supervision as sv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365").to(device)

# Initialize annotators
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
MASK_ANNOTATOR = sv.MaskAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator()

def calculate_end_frame_index(source_video_path):
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    return video_info.total_frames

def annotate_image(input_image, detections, labels) -> np.ndarray:
    output_image = MASK_ANNOTATOR.annotate(input_image, detections)
    output_image = BOUNDING_BOX_ANNOTATOR.annotate(output_image, detections)
    output_image = LABEL_ANNOTATOR.annotate(output_image, detections, labels=labels)
    return output_image

def process_video(input_video_path, confidence_threshold=0.6):
    video_info = sv.VideoInfo.from_video_path(input_video_path)
    total = calculate_end_frame_index(input_video_path)
    frame_generator = sv.get_video_frames_generator(
        source_path=input_video_path,
        end=total
    )

    result_file_name = f"{uuid.uuid4()}.mp4"
    result_file_path = os.path.join("./", result_file_name)
    
    with sv.VideoSink(result_file_path, video_info=video_info) as sink:
        for _ in tqdm(range(total), desc="Processing video..."):
            frame = next(frame_generator)
            results = query(Image.fromarray(frame), confidence_threshold)
            final_labels = []
            detections = sv.Detections.from_transformers(results[0])

            for label in results[0]["labels"]:
                final_labels.append(model.config.id2label[label.item()])
            frame = annotate_image(
                input_image=frame,
                detections=detections,
                labels=final_labels,
            )
            sink.write_frame(frame)

    return result_file_path

def query(image, confidence_threshold):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, threshold=confidence_threshold, target_sizes=target_sizes)
    return results

# Example usage
input_video_path = '/home/user/Documents/large-vision-models/test.mp4'  # Provide your video path here
output_video_path = process_video(input_video_path, confidence_threshold=0.5)
print(f"Processed video saved at: {output_video_path}")
