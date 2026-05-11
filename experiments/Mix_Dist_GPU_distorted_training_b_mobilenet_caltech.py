
# Changes from the original model: 

# Used MobileNetV2 trained on ImageNet (another similar dataset), since the authors did not provide 
# access to their trained backbone, and training it would take a long time, without
# significant difference in comparison fairness in testing the two models.  
# Might slightly change the expert branch architectures, if we see fit later. 
# Added torch.no_grad() to the validation, since validation does not require gradients or any 
# backpropagation, so the GPU memory usage becomes high unnecessarily and validating becomes slower.
# Here we trained the mobilenetV2 on the actual caltech256 images rather than directly using the 
# pretrained V2 on the imagenet, to get better accuracies, like the original paper.  


import torch
import torch.nn as nn
import numpy as np
import sys, time, math, os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import entropy
import pandas as pd
import torchvision.transforms as transforms
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from scipy import stats
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.datasets.voc as voc
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from pthflops import count_ops

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mobileNet import B_MobileNet
import cv2
from PIL import Image
import argparse
from tqdm import tqdm

class AddGaussianNoise(object):
  def __init__(self, distortion_list, mean=0.):
    self.distortion_list = distortion_list

  def __call__(self, img):
    image = np.array(img)
    self.std = self.distortion_list[np.random.choice(len(self.distortion_list), 1)[0]]
    noise_img = image + np.random.normal(0, self.std, (image.shape[0], image.shape[1], image.shape[2]))
    noise_img = np.clip(noise_img, 0, 255)
    return Image.fromarray(np.uint8(noise_img))
    
  def __repr__(self):
    return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddGaussianBlur(object):
	def __init__(self, distortion_list, mean=0.):
		self.distortion_list = distortion_list

	def __call__(self, img):
		image = np.array(img)
		self.std = self.distortion_list[np.random.choice(len(self.distortion_list), 1)[0]]
		blur = cv2.GaussianBlur(image, (4*self.std+1, 4*self.std+1), self.std, None, self.std, cv2.BORDER_CONSTANT)
		return Image.fromarray(blur) 

	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddBlurNoise(object):
  def __init__(self, blur_list, noise_list, mean=0.):
    self.blur_list = blur_list
    self.noise_list = noise_list

  def __call__(self, img):
    image = np.array(img)

    blur_std = self.blur_list[np.random.choice(len(self.blur_list))]
    image = cv2.GaussianBlur(
        image,
        (4*blur_std+1, 4*blur_std+1),
        blur_std,
        None,
        blur_std,
        cv2.BORDER_CONSTANT
    )

    noise_std = self.noise_list[np.random.choice(len(self.noise_list))]
    image = image + np.random.normal(0, noise_std, image.shape)
    image = np.clip(image, 0, 255)

    return Image.fromarray(np.uint8(image))


def save_idx(train_idx, val_idx, savePath):
  data = np.array([train_idx, val_idx])
  np.save(savePath, data)

class MapDataset(torch.utils.data.Dataset):
  def __init__(self, dataset, transformation):
    self.dataset = dataset
    self.transformation = transformation

  def __getitem__(self, index):
    x = self.transformation(self.dataset[index][0])
    y = self.dataset[index][1]
    return x, y

  def __len__(self):
    return len(self.dataset)


def load_caltech(root_path, transf_train, transf_valid, batch_size,savePath_idx_dataset, split_train=0.8):

  dataset = datasets.ImageFolder(root_path)

  train_dataset = MapDataset(dataset, transf_train)
  val_dataset = MapDataset(dataset, transf_valid)

  if (os.path.exists(savePath_idx_dataset)):
    data = np.load(savePath_idx_dataset, allow_pickle=True)
    train_idx, valid_idx = data[0], data[1]

  else:
    nr_samples = len(dataset)
    indices = list(range(nr_samples))

    split = int(np.floor(split_train * nr_samples))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[:split], indices[split:]
    save_idx(train_idx, valid_idx, savePath_idx_dataset)


  train_data = torch.utils.data.Subset(train_dataset, indices=train_idx)
  val_data = torch.utils.data.Subset(val_dataset, indices=valid_idx)

  trainLoader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                          shuffle=True, num_workers=4)
  valLoader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, 
                                            num_workers=4)

  return trainLoader, valLoader


def trainBranches(model, train_loader, optimizer, criterion, n_branches, epoch, device, loss_weights):
  running_loss = []
  train_acc_dict = {i: [] for i in range(1, (n_branches+1)+1)}
  model.train()

  for i, (data, target) in enumerate(tqdm(train_loader), 1):
    #print("Batch: %s/%s"%(i, len(train_loader)))
    data, target = data.to(device), target.long().to(device)

    output_list, conf_list, class_list = model(data)

    optimizer.zero_grad()
    loss = 0
    for j, (output, inf_class, weight) in enumerate(zip(output_list, class_list, loss_weights), 1):
      loss += weight*criterion(output, target)
      train_acc_dict[j].append(100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0))


    running_loss.append(float(loss.item()))
    loss.backward()
    optimizer.step()
    

    # clear variables
    del data, target, output_list, conf_list, class_list

  loss = round(np.average(running_loss), 4)
  print("Epoch: %s"%(epoch))
  print("Train Loss: %s"%(loss))

  result_dict = {"epoch":epoch, "train_loss": loss}
  for key, value in train_acc_dict.items():
    result_dict.update({"train_acc_branch_%s"%(key): round(np.average(train_acc_dict[key]), 4)})    
    print("Train Acc Branch %s: %s"%(key, result_dict["train_acc_branch_%s"%(key)]))
  
  return result_dict

def evalBranches(model, val_loader, criterion, n_branches, epoch, device):
  running_loss = []
  val_acc_dict = {i: [] for i in range(1, (n_branches+1)+1)}
  model.eval()

  with torch.no_grad():
    for i, (data, target) in enumerate(val_loader, 1):
      data, target = data.to(device), target.long().to(device)

      output_list, conf_list, class_list = model(data)

      loss = 0
      for j, (output, inf_class, weight) in enumerate(zip(output_list, class_list, loss_weights), 1):
        loss += weight*criterion(output, target)
        val_acc_dict[j].append(100*inf_class.eq(target.view_as(inf_class)).sum().item()/target.size(0))


      running_loss.append(float(loss.item()))    

      # clear variables
      del data, target, output_list, conf_list, class_list

    loss = round(np.average(running_loss), 4)
    print("Epoch: %s"%(epoch))
    print("Val Loss: %s"%(loss))

    result_dict = {"epoch":epoch, "val_loss": loss}
    for key, value in val_acc_dict.items():
      result_dict.update({"val_acc_branch_%s"%(key): round(np.average(val_acc_dict[key]), 4)})    
      print("Val Acc Branch %s: %s"%(key, result_dict["val_acc_branch_%s"%(key)]))
    
    return result_dict


parser = argparse.ArgumentParser(description='Evaluating DNNs perfomance using distorted image: blur ou gaussian noise')
parser.add_argument('--distortion_type', type=str, default="pristine", 
  choices=['pristine', 'gaussian_blur','gaussian_noise', "blur_noise"], help='Distortion Type (default: pristine)')
parser.add_argument('--root_path', type=str, help='Path to the pristine Caltech256-dataset')
#parser.add_argument('--dataset_path', type=str, help='Path to the Caltech-256 dataset')

args = parser.parse_args()



root_dir = args.root_path
seed = 42
distortion_type = args.distortion_type
dataset_path = os.path.join(".", "dataset", "256_ObjectCategories")
batch_size = 32
model_name = "mobilenet"
pretrained = True
dataset_name = "caltech"
model_id = 21

mean, std = [0.457342265910642, 0.4387686270106377, 0.4073427106250871],[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]


#if (distortion_type == "gaussian_blur"):
#  distortion_list = [1, 2, 3, 4, 5]
#  distortion_app = AddGaussianBlur
#else:
#  distortion_list = [5, 10, 20, 30, 40]
#  distortion_app = AddGaussianNoise

blur_list = [1, 2, 3, 4, 5]
noise_list = [5, 10, 20, 30, 40]

if distortion_type == "gaussian_blur":
  distortion_transform = AddGaussianBlur(blur_list, 0)

elif distortion_type == "gaussian_noise":
  distortion_transform = AddGaussianNoise(noise_list, 0)

elif distortion_type == "blur_noise":
  distortion_transform = AddBlurNoise(blur_list, noise_list, 0)

else:
  distortion_transform = None

root_dir = os.path.join(root_dir, model_name, dataset_name)
results_dir = os.path.join(".", "results")

model_save_path = os.path.join(
    results_dir,
    f"Mix_Dist_{distortion_type}_model_{model_name}_{dataset_name}_{model_id}.pth"
)
#savePath_idx_dataset = os.path.join(root_dir, "save_idx_b_%s_%s_%s.npy"%(model_name, dataset_name, model_id))
savePath_idx_dataset = os.path.join(".", "save_idx_b_%s_%s_%s.npy"%(model_name, dataset_name, model_id))
#pristine_model_path = os.path.join(root_dir, "pristine_model_b_mobilenet_caltech_21.pth")
df_history_save_path = os.path.join(
    results_dir,
    f"Mix_Dist_history_distorted_{distortion_type}_{model_name}_{dataset_name}_{model_id}.csv"
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(seed)
torch.manual_seed(seed)

if distortion_type == "pristine":
  distorted_transf_train = transforms.Compose([
      transforms.Resize((300, 300)),
      transforms.RandomChoice([
          transforms.ColorJitter(brightness=(0.80, 1.20)),
          transforms.RandomGrayscale(p=0.25)
      ]),
      transforms.RandomHorizontalFlip(p=0.25),
      transforms.RandomRotation(25),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std),
  ])

  distorted_transf_valid = transforms.Compose([
      transforms.Resize(330),
      transforms.CenterCrop(300),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std),
  ])

else:
  distorted_transf_train = transforms.Compose([
      transforms.Resize((300, 300)),
      transforms.RandomApply([distortion_transform], p=0.5),
      transforms.RandomChoice([
          transforms.ColorJitter(brightness=(0.80, 1.20)),
          transforms.RandomGrayscale(p=0.25)
      ]),
      transforms.RandomHorizontalFlip(p=0.25),
      transforms.RandomRotation(25),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std),
  ])

  distorted_transf_valid = transforms.Compose([
      transforms.Resize(330),
      transforms.CenterCrop(300),
      transforms.RandomApply([distortion_transform], p=0.5),
      transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std),
  ])


train_loader, val_loader = load_caltech(dataset_path, distorted_transf_train, distorted_transf_valid, 
		batch_size, savePath_idx_dataset)

n_classes = 258
pretrained = True
calibration = True
n_branches = 3
img_dim = 300
exit_type = None
optimizer_name = "Adam"
lr = [1.5e-4, 1e-2]
weight_decay = 0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

branchynet = B_MobileNet(n_classes, True, n_branches, img_dim, exit_type, device)

if distortion_type != "pristine":
  checkpoint = torch.load(
    os.path.join(
      results_dir,
      f"Mix_Dist_pristine_model_{model_name}_{dataset_name}_{model_id}.pth"
    ),
    map_location=device,
    weights_only=False
  )
  branchynet.load_state_dict(checkpoint["model_state_dict"])

branchynet = branchynet.to(device)

criterion = nn.CrossEntropyLoss()

if distortion_type == "pristine":
  for param in branchynet.stages.parameters():
    param.requires_grad = True
else:
  for param in branchynet.stages.parameters():
    param.requires_grad = False

if distortion_type == "pristine":
  optimizer = optim.Adam([
      {'params': branchynet.stages.parameters(), 'lr': lr[0]},
      {'params': branchynet.exits.parameters(), 'lr': lr[1]},
      {'params': branchynet.fully_connected.parameters(), 'lr': lr[1]}
  ], weight_decay=weight_decay)
else:
  optimizer = optim.Adam([
      {'params': branchynet.exits.parameters(), 'lr': lr[1]},
      {'params': branchynet.fully_connected.parameters(), 'lr': lr[1]}
  ], weight_decay=weight_decay)


scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=0, last_epoch=-1)

loss_weights = [1, 1, 1, 1]

epoch = 0
count = 0
best_val_loss = np.inf
patience = 5

df = pd.DataFrame()
while 1:
  epoch+=1
  print("Epoch: %s"%(epoch))
  result = {}
  result.update(trainBranches(branchynet, train_loader, optimizer, criterion, n_branches, epoch, device, loss_weights))
  scheduler.step()
  result.update(evalBranches(branchynet, val_loader, criterion, n_branches, epoch, device))

  df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
  df.to_csv(df_history_save_path, index=False)

  if (result["val_loss"] < best_val_loss):
    best_val_loss = result["val_loss"]
    count = 0
    save_dict = {"model_state_dict": branchynet.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "epoch": epoch, "val_loss": result["val_loss"]}
    
    for i in range(1, n_branches+1+1):
      save_dict.update({"val_acc_branch_%s"%(i): result["val_acc_branch_%s"%(i)]})

    torch.save(save_dict, model_save_path)

  else:
    count += 1
    print("Count: %s"%(count))
    if (count > patience):
      print("Stop! Patience is finished")
      break
