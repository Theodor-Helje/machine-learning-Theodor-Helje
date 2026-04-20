from torchvision.io import decode_image
from torchvision.models import get_model, get_model_weights
from torchcam.methods import LayerCAM
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt


class ResnetVisualizer():
    def __init__(self):
        self.weights = get_model_weights("resnet18").DEFAULT
        self.model = get_model("resnet18", weights=self.weights).eval()
        self.preprocess = self.weights.transforms()
        self.labels = self.weights.meta["categories"]

        self.img = None
        self.img_tensor = None
        self.logits = None
        self.activation_map = None


    def fit_image(self, img_path):
        """fits an image, saves img, img_tensor, logits and activation map as properties"""

        self.img = decode_image(img_path)
        self.img_tensor = self.preprocess(self.img)

        with LayerCAM(self.model, target_layer=self.model.layer4) as cam_extractor:
            self.logits = self.model(self.img_tensor.unsqueeze(0))
            self.activation_map = cam_extractor(self.logits.squeeze(0).argmax().item(), self.logits)


    def visualize_prediction(self):
        """shows fitted original image with heatmap overlay showing activation"""

        if self.logits is None:
            raise ValueError("Image must be fitted to visualize")

        result = overlay_mask(to_pil_image(self.img), to_pil_image(self.activation_map[0].squeeze(0), mode='F'), alpha=0.5)

        plt.imshow(result)
        plt.axis('off')
        plt.tight_layout()
        plt.title(f"predicted class: {self.labels[self.logits.squeeze(0).argmax().item()]}, id: {self.logits.squeeze(0).argmax().item()}")
        plt.show()


    def fit_visualize_image(self, img_path):
        self.fit_image(img_path)
        self.visualize_prediction()



if __name__ == "__main__":
    visualizer = ResnetVisualizer()

    visualizer.fit_visualize_image("data/skivspelare.jpeg")