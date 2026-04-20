from torchvision.io import decode_image
from torchvision.models import get_model, get_model_weights
from torchcam.methods import LayerCAM
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.utils import overlay_mask
from torch.nn.functional import softmax
from torch import topk
import matplotlib.pyplot as plt


class ResnetVisualizer():
    def __init__(self):
        self.weights = get_model_weights("resnet18").DEFAULT
        self.model = get_model("resnet18", weights=self.weights).eval()
        self.preprocess = self.weights.transforms()
        self.labels = self.weights.meta["categories"]

        self.layers = {
            "1": self.model.layer1,
            "2": self.model.layer2,
            "3": self.model.layer3,
            "4": self.model.layer4
        }

        self.img = None
        self.img_tensor = None
        self.logits = None
        self.layer = None
        self.top_k_index = None
        self.top_k_probs = None
        self.activation_maps = None

    def fit_image(self, img_path, top_k=1, layer=None):
        """fits an image, saves img, img_tensor, logits and activation maps as properties"""

        if layer == None:
            self.layer = "4"
        else:
            self.layer = str(layer)
        layer = self.layers[self.layer]

        self.img = decode_image(img_path)
        self.img_tensor = self.preprocess(self.img)

        with LayerCAM(self.model, target_layer=layer) as cam_extractor:
            self.logits = self.model(self.img_tensor.unsqueeze(0))

            probs = softmax(self.logits, dim=1)
            top_probs, top_ids = topk(probs, top_k)

            self.top_k_index = top_ids[0]
            self.top_k_probs = top_probs[0]
            self.activation_maps = []

            for id in self.top_k_index:
                self.logits = self.model(self.img_tensor.unsqueeze(0))

                cam = cam_extractor(id.item(), self.logits)
                self.activation_maps.append(cam[0].squeeze(0))


    def visualize_prediction(self, only_top_1=False):
        """plots top k predictions as heatmaps on the original fitted img"""

        if only_top_1:
            k = 1
        else:
            k = len(self.top_k_index)

        fig, axes = plt.subplots(1, k, figsize=(6 * k, 6))

        if k == 1:
            axes = [axes]

        for i in range(k):
            class_id = self.top_k_index[i].item()
            prob = self.top_k_probs[i].item()
            cam_map = self.activation_maps[i]
            result = overlay_mask(to_pil_image(self.img), to_pil_image(cam_map, mode='F'), alpha=0.5)

            axes[i].imshow(result)
            axes[i].axis('off')
            axes[i].set_title(f"predicted class: {self.labels[class_id]}\nid: {class_id}\nprob: {prob: .3f}\nlayer: {self.layer}")

        plt.tight_layout()
        plt.show()


    def fit_visualize_image(self, img_path, top_k=1, layer=None):
        self.fit_image(img_path, top_k, layer)
        self.visualize_prediction(only_top_1=True if top_k == 1 else False)



if __name__ == "__main__":
    visualizer = ResnetVisualizer()

    visualizer.fit_visualize_image("data/1990-mercedes-benz-190e-evo-ii.jpg", top_k=3, layer=4)

    for i in [1, 2, 3, 4]:
        visualizer.fit_visualize_image("data/1990-mercedes-benz-190e-evo-ii.jpg", top_k=3, layer=i)