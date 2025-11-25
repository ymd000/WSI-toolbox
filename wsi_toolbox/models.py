import timm
import torch
from timm.layers import SwiGLUPacked

MODEL_LABELS = {
    "uni": "UNI",
    "gigapath": "Prov-Gigapath",
    "virchow2": "Virchow2",
}
_MODEL_NAMES_BY_LABEL = {v: k for k, v in MODEL_LABELS.items()}  # Private
MODEL_NAMES = list(MODEL_LABELS.keys())


def get_model_label(model_name) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def create_foundation_model(model_name: str):
    """
    Create a foundation model instance by preset name.

    Args:
        model_name: One of 'uni', 'gigapath', 'virchow2'

    Returns:
        torch.nn.Module: Model instance (not moved to device, not in eval mode)
    """
    if model_name == "uni":
        return timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, dynamic_img_size=True, init_values=1e-5)

    if model_name == "gigapath":
        return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, dynamic_img_size=True)

    if model_name == "virchow2":
        return timm.create_model(
            "hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
        )

    raise ValueError(f"Invalid model_name: {model_name}. Must be one of {MODEL_NAMES}")
