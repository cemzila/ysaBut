import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder

MODEL_PATH = "model2.pth"
DATASET_ROOT = "./dataset"
IMAGE_SIZE = 256
NUM_CLASSES = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model 
class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))



# Modelin Yüklenmesi

model = ConvNeuralNet(NUM_CLASSES).to(DEVICE)
model = torch.load("model2.pth", map_location=DEVICE ,weights_only=False)
model.eval()

#Sınıf isimlerini direk fotoğraf dosyalarından alıyor
class_names = ImageFolder(DATASET_ROOT).classes


# Transforms

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# Tkinter 

root = tk.Tk()
root.title("Karar Destek Sistemi")
root.geometry("600x600")
root.configure(bg='#331F8E')

image_label = Label(root)
image_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)


def upload_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not file_path:
        return

    pil_image = Image.open(file_path).convert("RGB")

    display_image = pil_image.resize((300, 300))
    tk_image = ImageTk.PhotoImage(display_image)
    image_label.configure(image=tk_image)
    image_label.image = tk_image

    input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    class_name = class_names[predicted.item()]
    conf = confidence.item() * 100

    result_label.config(
        text=f"Tahmin: {class_name}\nGüven: {conf:.2f}%",
    
    )


upload_button = Button(
    root,
    text="Fotoğraf Seç",
    command=upload_and_predict,
    font=("Arial", 12),
    bg='blue',
    activebackground='red',
    fg='white'
)
upload_button.pack(pady=20)

root.mainloop()
