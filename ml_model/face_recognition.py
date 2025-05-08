from facenet_pytorch import InceptionResnetV1
import torch

# Load the pre-trained FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(face_tensor):
    """
    Takes a cropped and preprocessed face tensor and returns its embedding.
    """
    with torch.no_grad():
        embedding = facenet(face_tensor.to(device)).cpu()
    return embedding





