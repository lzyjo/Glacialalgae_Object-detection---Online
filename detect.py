from torchvision import transforms
from dataset import *
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from PIL import Image
from utils import *
import argparse
import numpy as np
from label_map import label_map_Classifier, label_map_OD
import matplotlib.pyplot as plt
import torch
import tifffile  # Use tifffile directly, not imread
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.models.detection import fasterrcnn_resnet50_fpn


device = torch.device("cpu")


########################### ARGUMENTS ################################

parser = argparse.ArgumentParser(description='Detection') # Parsing command-line arguments

## arguments
parser.add_argument('--model_type', type=str, choices=['object_detector', 'object_classifier'], required=True,
                    help='type of model to train: "object_detector" for object detection, "object_classifier" for classification')
parser.add_argument('--checkpoint', required=True, type=str, help='date of the dataset used for training')   
parser.add_argument('--img_path', required=True, type=str, help='path to the image')
parser

args = parser.parse_args() # Parse arguments


# object detector or classifier?
if args.model_type == 'object_detector':
    label_map = label_map_OD  # use object detector label map
if args.model_type == 'object_classifier':
    label_map = label_map_Classifier  # use classifier label map

# Color map for labels   
label_color_map = {label: color for label, color in zip(label_map.keys(), plt.cm.hsv(np.linspace(0, 1, len(label_map))))}

# Load model checkpoint
checkpoint = args.checkpoint
img_path = args.img_path
checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
# resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

"""
def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image

    # Transform
    image = normalize(to_tensor(original_image))  #image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                            max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Plot
    fig, ax = plt.subplots(1)
    ax.imshow(original_image)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        rect = patches.Rectangle((box_location[0], box_location[1]), box_location[2] - box_location[0],
                                box_location[3] - box_location[1], linewidth=2, edgecolor=label_color_map[det_labels[i]],
                                facecolor='none')
        ax.add_patch(rect)

        # Text
        plt.text(box_location[0], box_location[1] - 10, det_labels[i].upper(), color=label_color_map[det_labels[i]],
                fontsize=12, weight='bold', bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.show()


if __name__ == '__main__':
    img_path = img_path
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200)
"""



def visualize_detection_on_image(
    image_path,
    model_path,
    device="cpu",
    label_map=None,
    mask_threshold=0.7,
    threshold=0.5,
    figsize=(12, 12),
    debugging=False):

    """
    Visualizes object detection and segmentation results on an input image using a trained PyTorch model.
    This function loads an image and a trained object detection model, runs inference to detect objects,
    and displays the original image alongside the image with detected bounding boxes and (optionally) segmentation masks.
    It also prints out model and prediction statistics.
    Args:
        image_path (str): Path to the input image file (supports TIFF format).
        model_path (str): Path to the trained model checkpoint file.
        device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to "cpu".
        label_map (dict, optional): Optional mapping from class indices to class names. Not used in current implementation.
        mask_threshold (float, optional): Threshold for mask binarization. Defaults to 0.7.
        threshold (float, optional): Confidence threshold for filtering detections. Defaults to 0.5.
        figsize (tuple, optional): Figure size for matplotlib visualization. Defaults to (12, 12).
    Displays:
        - Side-by-side matplotlib figures of the original image and the image with detections.
        - Bounding boxes and class scores for detected objects.
        - Segmentation masks if available.
    Prints:
        - Image and model loading information.
        - Model training metrics and loss (if available in checkpoint).
        - Number of detections and masks.
        - Statistics of detection scores (min, mean, max, percentiles) for debugging (optional).
    Note:
        - Only the first three channels of the image are used for visualization.
        - The function expects the model checkpoint to contain either a model object or sufficient information to reconstruct the model.
    """

    device = torch.device(device)

    print(f"Reading image from: {image_path}")
    # Read image using tifffile
    image_np = tifffile.imread(image_path)
    print(f"Original image shape: {image_np.shape}")

    # Convert to torch tensor and preserve original shape
    image = torch.from_numpy(image_np)
    # If image is HWC, convert to CHW
    if image.ndim == 3 and image.shape[-1] <= 4:
        image = image.permute(2, 0, 1)
    elif image.ndim == 2:
        image = image.unsqueeze(0)
    print(f"Image tensor shape (CHW): {image.shape}")

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get('num_classes', 2)
    # Load the model architecture and weights from model_path
    if 'model' in checkpoint:
        model = checkpoint['model']
        model.to(device)
        print("Model loaded from provided model_path and moved to device.")
    else:
        model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
        model.to(device)
        print("Model architecture created and moved to device.")

    # Print available metrics and loss if present in checkpoint
    if 'metrics' in checkpoint:
        print("Training metrics:")
        for k, v in checkpoint['metrics'].items():
            print(f"  {k}: {v}")
    if 'loss' in checkpoint:
        print(f"Final training loss: {checkpoint['loss']}")
    elif 'train_loss' in checkpoint:
        print(f"Final training loss: {checkpoint['train_loss']}")
    else:
        print("No loss value found in checkpoint.")

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Use the original image shape for prediction
        x = image[:3, ...].float() / 255.0  # Normalize the image
        x = x.to(device)
        print("Running model inference...")
        predictions = model([x])
        pred = predictions[0]  # Get the first prediction
    print("Inference complete.")

    keep = pred["scores"] > threshold
    pred_boxes = pred["boxes"][keep]

    # Filter labels and scores using keep mask
    filtered_labels = pred["labels"][keep]
    filtered_scores = pred["scores"][keep]

    # For visualization, scale to uint8 and keep original shape
    vis_image = image.clone()
    if vis_image.dtype != torch.uint8:
        vis_image = (255.0 * (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min())).to(torch.uint8)
    vis_image = vis_image[:3, ...]  # Only use first 3 channels for visualization
    print("Image normalized for visualization.")

    pred_labels = [f"cell: {score:.3f}" for score in filtered_scores]
    pred_boxes = pred_boxes.long()
    print(f"Number of predicted boxes: {len(pred_boxes)}")

    if "masks" in pred:
        masks = (pred["masks"] > mask_threshold).squeeze(1)
        print(f"Number of predicted masks: {masks.shape[0]}")
        output_image = draw_bounding_boxes(vis_image, pred_boxes, pred_labels, colors="red")
        output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")
        # Show original and detected images side by side
        fig, axs = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
        axs[0].imshow(image_np if image_np.ndim == 2 else image_np[..., :3])
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(output_image.permute(1, 2, 0).cpu().numpy())
        axs[1].set_title("Detections")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()
    else:
        output_image = draw_bounding_boxes(vis_image, pred_boxes, pred_labels, colors="red")
        print("No masks found in prediction.")
        # Display original and detected images side by side
        fig, axs = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
        axs[0].imshow(image_np if image_np.ndim == 2 else image_np[..., :3])
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(output_image.permute(1, 2, 0).cpu().numpy())
        axs[1].set_title("Detections")
        axs[1].axis("off")
        print(f"Number of detections: {len(pred_boxes)}") #print(f"Number of detections: {len(pred['boxes'])}") returns the number of boxes in the prediction
                                                            # which is by default 100
        plt.tight_layout()
        plt.show()
    
    if debugging:
        print("Debugging information:")
        print(f"Image shape: {image.shape}")
        print(f"Model type: {type(model)}")
        print(f"Number of classes: {num_classes}")
        print("Predicted boxes:", pred_boxes.cpu().numpy())
        print("Predicted labels:", filtered_labels.cpu().numpy())
        print("Predicted scores:", filtered_scores.cpu().numpy())
        print("All scores:", pred["scores"].cpu().numpy())
        print(
            "Min score:", pred["scores"].min().item(),
            "Mean score:", pred["scores"].mean().item(),
            "Max score:", pred["scores"].max().item(),
            "25th percentile score:", torch.quantile(pred["scores"], 0.25).item(),
            "50th percentile score:", torch.quantile(pred["scores"], 0.5).item(),
            "75th percentile score:", torch.quantile(pred["scores"], 0.75).item()
        )
          
    
if __name__ == "__main__":
    visualize_detection_on_image(
        image_path=r'3_TrainingData\20250318_Augmented\Split\test\images\1.tif',
        model_path=r'5_Results/5_Results/training_results_20250513_Augmented.txt_model.pth',
        device="cpu",
        label_map=None,
        mask_threshold=0.7,
        threshold=0.5,
        figsize=(12, 12)
    )