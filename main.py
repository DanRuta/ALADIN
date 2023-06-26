
# pip install ml-collections

import argparse

import torchvision.transforms as transforms
import torch
import torch.nn as nn
from PIL import Image

from vit_networks import VisionTransformer


class AViT (nn.Module):
    def __init__ (self):
        super(AViT, self).__init__()
        self.enc_style = VisionTransformer()

    def encode (self, image):
        style_code_proj, style_code = self.enc_style(image, just_style_code=True)
        return style_code_proj



def load_img (file):
    img_file = Image.open(file)
    img_file.load()

    layers = img_file.split()
    if len(layers)==1 or len(layers)==2:
        img_file = img_file.convert("RGB")
    elif len(layers)==4:
        img_new = Image.new("RGB", img_file.size, (255, 255, 255))
        img_new.paste(img_file, mask=layers[3])
        return img_new

    return img_file


def main (ckpt_file, img_file, gpu_index):

    checkpoint = torch.load(ckpt_file, map_location="cpu")
    model = AViT()
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(f'cuda:{gpu_index}')


    image = load_img(img_file)
    transform_fn = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform_fn(image)
    image = image.view((1, image.shape[0], image.shape[1], image.shape[2])) # Simulate batch
    image = image.to(f'cuda:{gpu_index}')
    print(f'image, {image.shape}')

    embedding = model.encode(image)
    embedding = embedding / embedding.norm(dim=1)[:, None]
    embedding = embedding.view((1, embedding.shape[1])) # Simulate batch
    print("embedding", embedding.shape)
    print(embedding)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="aladinVIT_bamfg_64.47.pt", help="ckpt file path")
    parser.add_argument("--im", default="test.jpg", help="image to run embedding on")
    parser.add_argument("--gpu", default=0, help="GPU index")
    args = parser.parse_args()


    main(args.ckpt, args.im, args.gpu)