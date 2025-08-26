import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Prediction function with threshold
def predict_image(model, image, class_names, threshold=0.7):
    """
    image: PIL Image
    Returns predicted class or "Not from known classes"
    """
    model.eval()
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        max_conf, pred_class = torch.max(probs, 1)
        if max_conf.item() < threshold:
            return f"Not from known classes (confidence={max_conf.item():.2f})"
        else:
            return f"{class_names[pred_class.item()]} (confidence={max_conf.item():.2f})"
