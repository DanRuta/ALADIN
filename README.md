# ALADIN

ALADIN style representation model from the ALADIN/StyleBabel papers

This repo contains the inference code and model weights (releases) for the ViT variant of the ALADIN style representation model, first described in the *ALADIN: All Layer Adaptive Instance Normalization for Fine-grained Style Similarity* paper [1], and evolved with the ViT backbone in the *StyleBabel: Artistic Style Tagging and Captioning* paper [2].

The model weights can be downloaded from the releases tab of this repo. Download and extract that `aladinVIT_bamfg_64.47.pt` file next to the scripts, to use it. These are weights trained over the BAM-FG dataset [1], with an accuracy of 64.47% over its test set, as described in [2].

# In this repo

The `main.py` file contains a minimal runnable example for inference using ALADIN-ViT. It loads the `test.jpg` image and prints its style embedding.

The `cluster_search.py` file contains a more complex example, where ALADIN style codes are extracted for a dataset of images, clustered (k=100) with faiss, and then image retrieval is executed with each centroid as a query, for the top 10 most stylistically similar images.


# Citation(s)

If you use this model in your work/research, please consider citing our work:
```
@InProceedings{Ruta_2021_ICCV,
    author    = {Ruta, Dan and Motiian, Saeid and Faieta, Baldo and Lin, Zhe and Jin, Hailin and Filipkowski, Alex and Gilbert, Andrew and Collomosse, John},
    title     = {ALADIN: All Layer Adaptive Instance Normalization for Fine-Grained Style Similarity},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {11926-11935}
}
@InProceedings{10.1007/978-3-031-20074-8_13,
    author="Ruta, Dan
    and Gilbert, Andrew
    and Aggarwal, Pranav
    and Marri, Naveen
    and Kale, Ajinkya
    and Briggs, Jo
    and Speed, Chris
    and Jin, Hailin
    and Faieta, Baldo
    and Filipkowski, Alex
    and Lin, Zhe
    and Collomosse, John",
    editor="Avidan, Shai
    and Brostow, Gabriel
    and Ciss{\'e}, Moustapha
    and Farinella, Giovanni Maria
    and Hassner, Tal",
    title="StyleBabel: Artistic Style Tagging and Captioning",
    booktitle="Computer Vision -- ECCV 2022",
    year="2022",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="219--236",
    isbn="978-3-031-20074-8"
}
```