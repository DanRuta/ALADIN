import os
import shutil
import random

import faiss
import pickle as pkl
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

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





def extract_embeddings(images_folders, model, transform_fn):

    if os.path.exists("final.npy") and os.path.exists("filenames.pkl"):
        all_embeddings = np.load("final.npy")
        with open("filenames.pkl", "rb") as f:
            all_filepaths = pkl.load(f)
        print(f'Loaded embeddings data from file')
        return all_embeddings, all_filepaths

    all_filepaths = []
    all_embeddings = []
    sub_folders = sorted(os.listdir(images_folders))

    for sfi,sub_folder in enumerate(sub_folders):
        print(f'\r Extracting image embeddings... {sfi+1}/{len(sub_folders)} ({round((sfi+1)/len(sub_folders)*100,2)}%)  ', end="", flush=True)
        images = sorted(os.listdir(f'{images_folders}/{sub_folder}'))

        for image_name in images:
            filepath = f'{images_folders}/{sub_folder}/{image_name}'
            image = load_img(filepath)
            image = transform_fn(image)
            image = image.view((1, image.shape[0], image.shape[1], image.shape[2])) # Simulate batch
            image = image.to(f'cuda:{gpu_index}')

            embedding = model.encode(image)
            embedding = embedding / embedding.norm(dim=1)[:, None]
            embedding = embedding.view((1, embedding.shape[1]))

            all_filepaths.append(filepath)
            all_embeddings.append(embedding.cpu().detach().numpy().squeeze())

    print("")
    np.save("final.npy", all_embeddings)
    with open("filenames.pkl", "wb") as f:
        pkl.dump(all_filepaths, f)

    return np.array(all_embeddings), all_filepaths

def train_cluster(d=256, k = 50000, use_gpu=False):
    if os.path.exists("centroids.npy"):
        kmeans_centroids = np.load("centroids.npy")
        print(f'Loaded kmeans index from file')
        return kmeans_centroids

    print("Computing faiss kmeans...")
    np.random.seed(123)
    x = np.load("final.npy")
    if use_gpu:
        ngpus = faiss.get_num_gpus()
        print(f'Number of GPUs: {ngpus}')
        res = [faiss.StandardGpuResources() for _ in range(ngpus)]
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0

    index = faiss.IndexFlatL2(d)
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)  # use all GPUs

    kmeans = faiss.Kmeans(d, k, niter=20, verbose=True, gpu=True)
    kmeans.train(x)
    kmeans_index = kmeans.index
    if use_gpu:
        kmeans_index = faiss.index_gpu_to_cpu(kmeans_index)

    faiss.write_index(kmeans_index, 'trained_kmeans.index')
    np.save('centroids.npy', kmeans.centroids)

    return kmeans.centroids


def run_image_retrieval(all_embeddings, all_filepaths, kmeans_centroids, k):
    shutil.rmtree("output", ignore_errors=True)
    os.makedirs("output", exist_ok=True)

    centroids_queries = [int(random.random()*k) for _ in range(10)]
    print(f'Querying centroids: {", ".join([str(v) for v in centroids_queries])}')

    index = faiss.IndexFlatL2(256)
    index.add(all_embeddings)

    for ci, centroid_i in enumerate(centroids_queries):
        print(f'\r Running image retrieval... {ci+1}/{len(centroids_queries)} ', end="", flush=True)
        os.makedirs(f'./output/{ci}', exist_ok=True)

        centroid = kmeans_centroids[centroid_i]

        D, I = index.search(np.array([centroid]), 10)
        I = I[0]

        for ri,result_index in enumerate(I):
            base_filename = all_filepaths[result_index].split("/")[-1]
            shutil.copyfile(all_filepaths[result_index], f'./output/{ci}/{ri}___{base_filename}')



if __name__ == '__main__':

    # Set up the ALADIN-ViT model
    model = AViT()
    gpu_index = 0
    checkpoint = torch.load(f'aladinVIT_bamfg_64.47.pt', map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(f'cuda:{gpu_index}')
    transform_fn = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Extract embeddings for a folder of images, into a final.npy file
    images_folders = "<data path>/BAM-FG/images" # a folder in which there are sub-folders of images; Style-coherent images in each sub-folder
    all_embeddings, all_filepaths = extract_embeddings(images_folders, model, transform_fn)

    # Create the faiss kmeans index for all the embeddings, saved to centroids.npy
    k = 100
    kmeans_centroids = train_cluster(k=k, use_gpu=False)

    # Pick a few centroids and run image retrieval over the corpus
    run_image_retrieval(all_embeddings, all_filepaths, kmeans_centroids, k)



