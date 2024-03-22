import streamlit as st
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

# Load the saved model architecture
model.load_state_dict(torch.load("model.pth"))
model.eval()

def preprocess_input(input_image):
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')
    face = face.unsqueeze(0)  # Add the batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)
    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0
    return face

def predict(input_image: Image.Image):
    # Preprocess the input image for the loaded model
    face = preprocess_input(input_image)

    # Get the visualization using GradCAM
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    face_image_to_plot = (face_image_to_plot * 255).astype(np.uint8)  # Convert to uint8

    target_layers = [model.block8.branch1[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0, :]

    # Resize visualization to match the dimensions of face_image_to_plot
    visualization = cv2.resize(grayscale_cam, (face_image_to_plot.shape[1], face_image_to_plot.shape[0]))

    # Convert visualization to uint8
    visualization = (visualization * 255).astype(np.uint8)

    # Blend the images using addWeighted
    face_with_mask = cv2.addWeighted(
        cv2.cvtColor(face_image_to_plot, cv2.COLOR_RGB2BGR),
        1,
        cv2.cvtColor(visualization, cv2.COLOR_GRAY2BGR),
        0.5,
        0
    )

    # Make predictions using the loaded model
    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"
        real_prediction = 1 - output.item()
        fake_prediction = output.item()

        confidences = {
            'real': real_prediction,
            'fake': fake_prediction
        }

    return confidences, Image.fromarray(cv2.cvtColor(face_with_mask, cv2.COLOR_BGR2RGB))

def main():
    st.title("DeepFake Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Perform prediction
        result, visualization = predict(image)

        # Print or use the result as needed
        st.write("Confidence Scores:")
        st.write(result)

        # Display the visualization
        st.image(visualization, caption='Visualization', use_column_width=True)

if __name__ == "__main__":
    main()