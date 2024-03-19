from torchvision.models.vision_transformer import vit_b_16
import pathlib
import torch
import importlib


def Vit(
    num_classes,
    pretrain: pathlib.Path = None,
    model_type="torchvision.models.vision_transformer.vit_b_16",
):
    module_path = model_type.rsplit(".", 1)[0]
    module_name = model_type.rsplit(".", 1)[1]
    module = importlib.import_module(module_path)
    model_obj = getattr(module, module_name)
    model = model_obj()
    if pretrain:
        para_dict = torch.load(pretrain)
        model.load_state_dict(para_dict)
    assert hasattr(model.heads, "head") and isinstance(
        model.heads.head, torch.nn.Linear
    )
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    return model


features = []


def feature_extract_hook(module, input, output):
    features.append(input)


if __name__ == "__main__":
    image = torch.rand([64, 3, 224, 224]).cuda()
    model = vit_b_16().cuda()
    model.heads.head.register_forward_hook(feature_extract_hook)
    logits = model(image)
    print(features[-1][0].size())
