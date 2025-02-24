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
from label_map import *

device = torch.device("cpu")
label_color_map = {label: color for label, color in zip(label_map.keys(), plt.cm.hsv(np.linspace(0, 1, len(label_map))))}


# Parsing command-line arguments
parser = argparse.ArgumentParser(description='Detection')

## checkpoint argument
parser.add_argument('--checkpoint', required=True, type=str, help='date of the dataset used for training')

### img_path argument   
parser.add_argument('--img_path', required=True, type=str, help='path to the image')

# Parse arguments
args = parser.parse_args()


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
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

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
