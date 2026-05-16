
# Plotting overall accuracy as a function of blur and noise levels. 

# gets the necessary model weights from results folder, gets the data from the path, 
# computes new pictures based on level of distortion, runs inference on the models, 
# generates figure 3



import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mobileNet import B_MobileNet


class DistortedDataset(Dataset):
    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, target = self.base_dataset[self.indices[idx]]
        img = self.transform(img)
        return img, target


class ApplyDistortion:
    def __init__(self, blur_level, noise_level):
        self.blur_level = blur_level
        self.noise_level = noise_level

    def __call__(self, img):
        image = np.array(img)

        if self.blur_level > 0:
            sigma = int(self.blur_level)
            kernel_size = 4 * sigma + 1
            image = cv2.GaussianBlur(
                image,
                (kernel_size, kernel_size),
                sigma,
                None,
                sigma,
                cv2.BORDER_CONSTANT
            )

        if self.noise_level > 0:
            sigma = float(self.noise_level)
            image = image + np.random.normal(0, sigma, image.shape)
            image = np.clip(image, 0, 255)

        return Image.fromarray(np.uint8(image))


def load_indices(index_path, dataset_size, split_train=0.8):
    if os.path.exists(index_path):
        data = np.load(index_path, allow_pickle=True)
        return data[1]

    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(split_train * dataset_size))
    train_idx = indices[:split]
    val_idx = indices[split:]
    np.save(index_path, np.array([train_idx, val_idx], dtype=object))
    return val_idx


def make_loader(dataset_path, index_path, blur_level, noise_level, batch_size):    
    mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    base_dataset = datasets.ImageFolder(dataset_path)
    val_indices = load_indices(index_path, len(base_dataset))

    transform = transforms.Compose([
        transforms.Resize(330),
        transforms.CenterCrop(300),
        ApplyDistortion(blur_level, noise_level),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DistortedDataset(base_dataset, val_indices, transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return loader


def load_model(model_path, device):
    n_classes = 258
    n_branches = 3
    img_dim = 300
    exit_type = None

    model = B_MobileNet(n_classes, True, n_branches, img_dim, exit_type, device)

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model


def select_predictions_and_edge_count(conf_list, class_list, p_tar):
    conf_tensor = torch.stack([c.detach() for c in conf_list], dim=1)
    class_tensor = torch.stack([c.detach() for c in class_list], dim=1)

    selected = []
    edge_count = 0

    for i in range(conf_tensor.shape[0]):
        chosen = None

        for j in range(conf_tensor.shape[1]):
            if conf_tensor[i, j].item() >= p_tar:
                chosen = class_tensor[i, j]

                if j < 3:
                    edge_count += 1

                break

        if chosen is None:
            best_branch = torch.argmax(conf_tensor[i])
            chosen = class_tensor[i, best_branch]


        selected.append(chosen)

    return torch.stack(selected), edge_count


def evaluate_model(model, loader, device, p_tar):
    correct = 0
    total = 0
    edge_exits = 0

    with torch.no_grad():
        for data, target in tqdm(loader, leave=False):
            data = data.to(device)
            target = target.long().to(device)

            output_list, conf_list, class_list = model(data)
            pred, batch_edge_exits = select_predictions_and_edge_count(
                conf_list,
                class_list,
                p_tar
            )

            correct += pred.eq(target).sum().item()
            edge_exits += batch_edge_exits
            total += target.size(0)

    accuracy = 100.0 * correct / total
    edge_probability = edge_exits / total

    return accuracy, edge_probability


def evaluate_all():
    root_dir = "."
    results_dir = os.path.join(root_dir, "results")
    dataset_path = os.path.join(root_dir, "dataset", "256_ObjectCategories")
    index_path = os.path.join(root_dir, "save_idx_b_mobilenet_caltech_21.npy")

    p_tar = 0.8
    batch_size = 32
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_files = {
    "pristine": "Mix_Dist_pristine_model_mobilenet_caltech_21.pth",
    "gaussian_blur": "Mix_Dist_gaussian_blur_model_mobilenet_caltech_21.pth",
    "gaussian_noise": "Mix_Dist_gaussian_noise_model_mobilenet_caltech_21.pth",
    "blur_noise": "Mix_Dist_blur_noise_model_mobilenet_caltech_21.pth"
    }

    labels = {
    "pristine": r"$E_{pristine}$",
    "gaussian_blur": r"$E_{blur}$",
    "gaussian_noise": r"$E_{noise}$",
    "blur_noise": r"$E_{blur+noise}$"
    }

    blur_levels = [0, 1, 2, 3, 4, 5]
    noise_levels = [0, 1, 2, 3, 4, 5]

    models = {}

    for model_name, file_name in model_files.items():
        model_path = os.path.join(results_dir, file_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}")

        models[model_name] = load_model(model_path, device)

    rows = []

    for blur_level in blur_levels:
        for noise_level in noise_levels:
            loader = make_loader(
                dataset_path=dataset_path,
                index_path=index_path,
                blur_level=blur_level,
                noise_level=noise_level,
                batch_size=batch_size
            )

            for model_name, model in models.items():
                print(f"Evaluating model={model_name}, pair=({blur_level},{noise_level})")
                acc, prob_edge = evaluate_model(model, loader, device, p_tar)

                rows.append({
                    "model": model_name,
                    "sigma_blur": blur_level,
                    "sigma_noise": noise_level,
                    "pair": f"({blur_level},{noise_level})",
                    "avg_acc": acc,
                    "p_inference_on_edge": prob_edge
                })

    df = pd.DataFrame(rows)
    df.to_csv(
      os.path.join("experiments", "figure_5_style_results.csv"),
      index=False
    )

    plt.figure(figsize=(16, 5))

    pairs = [f"({b},{n})" for b in blur_levels for n in noise_levels]

    for model_name in ["pristine", "gaussian_blur", "gaussian_noise", "blur_noise"]:
        data = df[df["model"] == model_name].copy()
        data["pair"] = pd.Categorical(data["pair"], categories=pairs, ordered=True)
        data = data.sort_values("pair")

        plt.plot(
            data["pair"],
            data["p_inference_on_edge"],
            marker="o",
            linewidth=2,
            label=labels[model_name]
        )

    plt.xlabel(r"Mixed distortion level $(\sigma_B, \sigma_N)$")
    plt.ylabel("P[Inference on Edge]")
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join("experiments", "figure_5_style.png"),
        dpi=300
    )
    plt.show()


if __name__ == "__main__":
    evaluate_all()