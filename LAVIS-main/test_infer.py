import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

raw_image = Image.open("/vepfs-cnbja62d5d769987/liushaokun/sys_work/LAVIS-main/vqav2.png").convert("RGB")

model, vis_processors, _ = load_model_and_preprocess(
    name="mocle",
    model_type="default",
    is_eval=True,
    device=device
)

image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

response = model.generate({
    "image": image,
    "prompt": ["Describe this image."]
})

print(response)