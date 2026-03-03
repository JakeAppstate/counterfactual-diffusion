# pylint: disable=import-error
from typing import List, Tuple, Union, Optional
import os
import re
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
from torchvision.transforms import v2

class BaseDataset(Dataset):
    def __init__(self, df, data_path, resize_size):
        self.data_path = data_path
        self.resize_size = resize_size
        self.df = self._get_df(df)

    def _get_df(self, df):
        columns = ["Eye ID", "Final Label"]
        df = df[columns].copy() # drop all unspecified columns
        df["path"] = df["Eye ID"] \
            .map(lambda id: os.path.join(self.data_path, str(self._get_folder(id)), f"{id}.JPG"))
        df = df[df["path"].map(os.path.exists)] # filter if image exists

        df["label"] = (df["Final Label"] != "NRG").astype(dtype = np.int32)
        df = df.drop(columns, axis=1)
        return df

    def _get_folder(self, id: str) -> int:
        pattern = r"\d+"
        n = int(re.findall(pattern, id)[0])
        if n <= 17407:
            return 0
        if n <= 34815:
            return 1
        if n <= 52223:
            return 2
        if n <= 69631:
            return 3
        if n <= 87039:
            return 4
        else:
            return 5
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df["path"].iloc[idx]
        img = self._preprocess(img_path)
        label = torch.tensor(self.df["label"].iloc[idx])

        return img, label
    
    def _preprocess(self, img_path: str):
        """Preprocesses the image by
            Images must be located in subdirectories numbered 0-5 as per _get_folder logic in BaseDataset. applying CLAHE Contrast Enhancement
        
        Cropping occurs in the collate_fn due to GPU Usage
        """
        # Load Image
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.resize_size)
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        new_img = cv2.merge((l, a, b))
        img = cv2.cvtColor(new_img, cv2.COLOR_Lab2RGB)
        # To pytorch float tensor
        img = v2.functional.to_image(img)
        img = v2.functional.to_dtype(img, torch.float32, scale=True)
        return img
    
    def get_sample_weights(self):
        neg_weight = 1 / len(self.df[self.df["label"] == 0])
        pos_weight = 1 / len(self.df[self.df["label"] == 1])
        weights = np.where(self.df["label"] == 1, pos_weight, neg_weight)
        return torch.tensor(weights).double()
    
    def get_labels(self):
        return self.df["label"].to_numpy()
    
class GlaucomaDataset(BaseDataset):
    def __init__(self, df, bbox_df, data_path, resize_size, transform):
        # Merge predicted bouinding boxes for optic disk
        super().__init__(df, data_path, resize_size)
        for c in ["y", "x", "h", "w"]:
            assert c in bbox_df, \
                "Columns 'y', 'x', 'w', and 'h' should be in bbox_df"
        self.df = pd.merge(self.df, bbox_df, on="path")
        self.transform = transform

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        # top, left, height, width = self.df["y"], self.df["x"], self.df["h"], self.df["w"]
        img = v2.functional.crop(img, self.df["y"].iloc[idx], self.df["x"].iloc[idx],
                                 self.df["h"].iloc[idx], self.df["w"].iloc[idx])

        if self.transform is not None:
            img = self.transform(img)
        return img, label

class CropROITransform(torch.nn.Module):
    def __init__(self, yolo_path: str, yolo_size: Union[int, Tuple[int, int]],
                 target_size: Union[int, Tuple[int, int]]):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = torch.jit.load(yolo_path, map_location=self.device)
        self.yolo.eval()
        self.yolo_size = yolo_size if isinstance(yolo_size, tuple) else (yolo_size, yolo_size)
        self.target_size = target_size if isinstance(target_size, tuple) else (target_size, target_size)
    
    def get_bounding_boxes(self, batch):
        with torch.no_grad():
            output = self.yolo(batch)

        max_conf_indicies = torch.argmax(output[:, 4, :], dim = 1)
        batch_indices = torch.arange(output.size(0), device = output.device)
        x = output[batch_indices, 1, max_conf_indicies]
        y = output[batch_indices, 0, max_conf_indicies]
        return x, y
    
    def convert_boxes(self, x, y, orig_size):
        # map cordinates from yolo_size to original size
        x = x * orig_size[1] / self.yolo_size[0]
        y = y * orig_size[0] / self.yolo_size[1]
        target_w, target_h = self.target_size
        # convert center x y to top left
        left = x - 0.5 * target_w
        left = torch.clamp(left, min=0)
        top = y - 0.5 * target_h
        top = torch.clamp(top, min=0)

        return top, left

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = v2.functional.resize(images, self.yolo_size)

        x, y = self.get_bounding_boxes(batch)
        top, left = self.convert_boxes(x, y, images.shape[2:])
        target_w, target_h = self.target_size

        batch_indicies = torch.arange(batch.size(0), device=batch.device)
        batch = torchvision.ops.roi_align(input=batch,
                                          boxes=torch.stack([batch_indicies,
                                                             left, top, left + target_w,
                                                             top + target_h], dim=1),
                                          output_size=self.target_size)
        return batch

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.yolo.to(*args, **kwargs)
        return self

# typedef
DatasetTuple = Tuple[GlaucomaDataset, GlaucomaDataset, GlaucomaDataset, Optional[GlaucomaDataset]]

# TODO can probably refactor some of the code to be cleaner
# Could use torch.utils.data.Subset and sklearn.model_selection.train_test_split
class DataModule:
    def __init__(self, csv_path: str, data_path: str,
                 resize_size:  Union[int, Tuple[int, int]],
                 val_ratio: float,
                 test_ratio: float,
                 include_real: bool,
                 rec_ratio: float = 0,
                 seed: int = 7,
                 bbox_csv: Optional[str] = None,
                 yolo: CropROITransform = None,
                 n_sample: Union[int, Tuple[int, int, int, int]] = None,
                 rec_split: bool = False,
                 split_index: int = 1,
                 train_transform: torch.nn.Module = None,
                 val_transform: torch.nn.Module = None):
        self.csv_path = csv_path
        self.data_path = data_path
        self.resize_size = resize_size if isinstance(resize_size, tuple) else resize_size, resize_size
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.include_real = include_real
        self.rec_ratio = rec_ratio
        self.seed = seed
        self.bbox_csv = bbox_csv
        self.yolo = yolo
        self.n_sample = (n_sample,) * 4 if isinstance(n_sample, int) else n_sample
        self.rec_split = rec_split
        self.split_index = split_index
        self.train_transform = train_transform
        self.val_transform = val_transform

    def load_datasets(self) -> DatasetTuple:
        # Load in cdv and split
        df = pd.read_csv(self.csv_path, sep=';')
        dataframes = self._split_dataframes(df, self.val_ratio, self.test_ratio, self.include_real)
        # Get subset of datasets id needed
        if self.n_sample is not None:
            dataframes = [self._get_subset(df, n) for df, n in zip(dataframes, self.n_sample)]
        train_df, val_df, test_df, real_df = dataframes
        # Recursively split dataset if needed
        # Heatmap classifier only trains on validation data
        # Need train subset and validation subset
        if self.rec_split:
            df = dataframes[self.split_index]
            train_df, val_df, _, _ = self._split_dataframes(df, self.rec_ratio, 0, False)

        # Get precomputed bounding boxes if available
        # If they aren't, then compute them before training
        bbox_df = None
        if os.path.exists(self.bbox_csv):
            bbox_df = pd.read_csv(self.bbox_csv)
        else:
            bbox_df = self._compute_boxes_df(df)
            bbox_df.to_csv(self.bbox_csv, index=False)

        train = GlaucomaDataset(train_df, bbox_df, self.data_path,
                                self.resize_size, self.train_transform)
        val = GlaucomaDataset(val_df, bbox_df, self.data_path,
                              self.resize_size, self.val_transform)
        test = GlaucomaDataset(test_df, bbox_df, self.data_path,
                               self.resize_size, transform=self.val_transform)
        real = None
        if self.include_real:
            real = GlaucomaDataset(real_df, bbox_df, self.data_path,
                                   self.resize_size, self.val_transform)

        return train, val, test, real

    def _split_dataframes(self, df: pd.DataFrame, val_ratio: float, test_ratio: float, include_real: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        pos_df = df[df['Final Label'] == "RG"]
        neg_df = df[df['Final Label'] != "RG"]
        pos_df = pos_df.sample(frac=1, random_state=self.seed)
        neg_df = neg_df.sample(frac=1, random_state=self.seed)

        n_pos, n_neg = len(pos_df), len(neg_df)
        n_pos_val = int(len(pos_df) * val_ratio)
        n_pos_test = int(len(pos_df) * test_ratio)
        n_neg_real = int(n_pos_test / n_pos * n_neg)
        if not include_real:
            n_neg_real = 0

        # ds = [*test, *val, *train]
        pos_test = pos_df.iloc[:n_pos_test]
        pos_val = pos_df.iloc[n_pos_test : n_pos_test + n_pos_val]
        pos_train = pos_df.iloc[n_pos_test + n_pos_val:]
        neg_val = neg_df.iloc[:n_pos_val]
        neg_test = neg_df.iloc[n_pos_val : n_pos_val + n_pos_test]
        neg_real = neg_df.iloc[n_pos_val : n_pos_val + n_neg_real]
        neg_train = neg_df.iloc[n_pos_val + n_neg_real : ]

        train = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=self.seed)
        val = pd.concat([pos_val, neg_val]).sample(frac=1, random_state=self.seed)
        test = pd.concat([pos_test, neg_test]).sample(frac=1, random_state=self.seed)
        if include_real:
            real = pd.concat([pos_test, neg_real]).sample(frac=1, random_state=self.seed)
        else:
            real = None

        return train, val, test, real
    
    # Maybe get a weighted subset?
    # Maybe use Sampler to get subset?
    def _get_subset(self, df, n):
        if df is None or n is None:
            return df
        # Have 50/50 split
        # If n is odd then give the extra to negative
        n_neg = n // 2 + n % 2
        n_pos = n // 2
        df_neg = df[df['Final Label'] != "RG"]
        df_pos = df[df['Final Label'] == "RG"]
        if len(df_pos) < n_pos:
            n_neg += n_pos - len(df_pos)
            n_pos = len(df_pos)
        df_neg = df_neg.sample(n=n_neg, random_state=self.seed)
        df_pos = df_pos.sample(n=n_pos, random_state=self.seed)
        return pd.concat([df_neg, df_pos]).sample(frac=1.0, random_state=self.seed)

    def _compute_boxes_df(self, df):
        # Helper Class
        # pylint: disable-next=missing-class-docstring
        class PathDataset(Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self):
                return len(self.ds)
            def __getitem__(self, idx):
                # path = self.ds.df["path"].iloc[idx]
                img, _ = self.ds[idx]
                return idx, img

        assert not os.path.exists(self.bbox_csv), f"{self.bbox_csv} already exists"
        raw_ds = BaseDataset(resize_size=self.yolo.yolo_size, df=df, data_path=self.data_path)
        path_ds = PathDataset(raw_ds)
        loader = DataLoader(path_ds, batch_size = 16, num_workers = 8,
                            shuffle = False, pin_memory = True,
                            prefetch_factor = 2)
        print("Computing bounding boxes for ROI")
        columns = ["path", "x", "y", "w", "h"]
        all_paths = raw_ds.df["path"].tolist()
        path_list = []
        x_list = []
        y_list = []
        with torch.no_grad():
            for idx, images in tqdm(loader):
                images = images.to(self.yolo.device)
                x, y = self.yolo.get_bounding_boxes(images)
                x, y = self.yolo.convert_boxes(x, y, orig_size = self.resize_size)
                x, y = x.round().int(), y.round().int()
                paths = [all_paths[i] for i in idx]
                # x = x.cpu().tolist()
                # y = y.cpu().tolist()
                assert len(paths) == len(x) == len(y)
                path_list += paths
                x_list.append(x)
                y_list.append(y)
        x_list = torch.cat(x_list).cpu().tolist()
        y_list = torch.cat(y_list).cpu().tolist()
        n = len(x_list)
        w = [self.yolo.target_size[0]] * n
        h = [self.yolo.target_size[1]] * n
        rows = list(zip(path_list, x_list, y_list, w, h))
        new_df = pd.DataFrame(data = rows, columns = columns)
        return new_df

    def get_sampler(self, ds: BaseDataset, oversample: bool, num_samples: int = None,
                    replacement: bool = True, use_generator: bool = False):
        num_samples = num_samples if num_samples is not None else len(ds)
        generator = torch.Generator().manual_seed(self.seed) if use_generator else None
        if oversample:
            weights = ds.get_sample_weights()
            return WeightedRandomSampler(weights = weights, num_samples = num_samples,
                                         replacement = replacement, generator = generator)
        else:
            return RandomSampler(ds, replacement = replacement,
                                 num_samples = num_samples, generator = generator)
