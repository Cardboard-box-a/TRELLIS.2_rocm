from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        # transformers 5.x calls all_tied_weights_keys.keys() during model loading,
        # but BiRefNet (trust_remote_code) was written for older transformers and doesn't
        # define this attribute. Patch the base class before loading.
        from transformers import PreTrainedModel
        if not hasattr(PreTrainedModel, '_trellis2_patched'):
            _method_name = '_move_missing_keys_from_meta_to_device' if hasattr(PreTrainedModel, '_move_missing_keys_from_meta_to_device') else '_move_missing_keys_from_meta_to_cpu'
            _orig = getattr(PreTrainedModel, _method_name)
            def _patched(self_model, *args, **kwargs):
                if not hasattr(self_model, 'all_tied_weights_keys'):
                    self_model.all_tied_weights_keys = {}
                return _orig(self_model, *args, **kwargs)
            setattr(PreTrainedModel, _method_name, _patched)
            PreTrainedModel._trellis2_patched = True

        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
