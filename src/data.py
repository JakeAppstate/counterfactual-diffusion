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
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

class BaseDataset(Dataset):
    """
    BaseDataset is an abstract class that serves as a foundation for
    specific dataset implementations. It handles common functionalities such as
    loading data from a DataFrame, filtering valid image paths, and preparing labels.
    Subclasses must implement the __getitem__ method to define how individual
    data samples are retrieved.
    Attributes:
        df (pd.DataFrame): DataFrame containing image paths and labels.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Abstract method to retrieve a sample by index.
        collate_fn(batch): Custom collate function to stack images and labels.
    """
    def __init__(self, df: pd.DataFrame, data_dir: str, n: int = None):
        """
        Initializes the BaseDataset with a DataFrame and data directory.

        Args:
            df (pd.DataFrame): DataFrame containing image paths and labels.
            data_dir (str): Directory where the images are stored.
            n (int): Number of total items to include in the dataset.
                Defaults to None which includes the entire dataset.
        """
        columns = ["Eye ID", "Final Label"]
        df = df[columns].copy() # drop all unspecified columns
        df["path"] = df["Eye ID"] \
            .map(lambda id: os.path.join(data_dir, str(self._get_folder(id)), f"{id}.JPG"))
        df = df[df["path"].map(os.path.exists)] # filter if image exists

        df["label"] = (df["Final Label"] != "NRG").astype(dtype = np.int32)
        df = df.drop(columns, axis=1)

        if n is not None:
            df = df.sample(n)

        self.df = df

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract method")

    # def collate_fn(self, batch):
        """
        Collate a batch of (image, label) pairs into batched tensors and move them to the instance device.

        This method expects `batch` to be an iterable of two-element tuples (image_tensor, label_tensor).
        Each image tensor and each label tensor must have the same shape and dtype across the batch so that
        they can be stacked into a single tensor along a new batch dimension.

        Args:
            batch (Sequence[Tuple[torch.Tensor, torch.Tensor]]): Sequence of (image, label) pairs to collate.
                - image tensors should have shape (C, H, W) or similar, consistent across items.
                - label tensors should have a shape compatible for stacking (e.g., scalar, 1-D, or matching dims).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple (images, labels) where:
                - images is a tensor of shape (batch_size, C, H, W, ...) containing stacked image tensors,
                  moved to `self.device`.
                - labels is a tensor of shape (batch_size, ...) containing stacked label tensors,
                  moved to `self.device`.

        Raises:
            TypeError: If an item in `batch` is not a tuple of two torch.Tensor objects.
            RuntimeError: If tensors in the batch cannot be stacked due to mismatched shapes or dtypes.
        """
        # img = torch.stack([item[0] for item in batch]).to(self.device)
        # label = torch.stack([item[1] for item in batch]).to(self.device)
        # return img, label

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

class RawDataset(BaseDataset):
    """ Dataset that computes all preprocessing steps on the fly.

    RawDataset provides dataset handling for images that are preprocessed with CLAHE contrast
    enhancement and cropped to a region-of-interest (ROI) determined by a pretrained YOLO
    model. It is intended to be used with a DataLoader that calls the custom collate_fn to
    perform YOLO-based cropping on a GPU, while file-level reading and CPU-based preprocessing
    (however minimal) happen per sample.
    Args:
        resize_size (int | Tuple[int, int]):
            Size to which each image is first resized after loading. If an int is provided,
            it is treated as a square (resize_size, resize_size).
        target_size (int | Tuple[int, int]):
            Final crop size (width, height) in pixels that will be extracted around the
            YOLO-predicted center. If an int is provided, it is treated as a square.
        df (pandas.DataFrame):
            DataFrame describing the dataset. Must contain at least the columns:
                - "path": filesystem path to the image file.
                - "label": the target label for the image (any type used by training code).
        data_dir (str):
            Root directory for dataset files (delegated to BaseDataset for any folder logic).
        yolo_path (str):
            Filesystem path to a serialized YOLO model compatible with torch.jit.load. The
            model is loaded onto the selected device (CUDA if available, otherwise CPU).
        yolo_size (Tuple[int, int] | int, optional):
            Size (width, height) used as input to the YOLO model. If an int is provided,
            it is treated as (yolo_size, yolo_size). Defaults to 640.
    Attributes:
        resize_size (Tuple[int, int]):
            Normalized tuple form of the provided resize_size.
        target_size (Tuple[int, int]):
            Normalized tuple form of the provided target_size.
        device (torch.device):
            Device used for model inference and cropping (CUDA if available, else CPU).
        yolo (torch.jit.ScriptModule):
            ScriptModule loaded from yolo_path and set to eval() for inference.
        yolo_size (Tuple[int, int]):
            Normalized tuple form of the provided yolo_size.
        df (pandas.DataFrame):
            Reference to the provided DataFrame (inherited/initialized via BaseDataset).
    Methods:
        __getitem__(idx) -> (image, label):
            - Loads and preprocesses a single sample indicated by idx using _preprocess.
            - Returns a tuple (img, label). The preprocessing step performs OpenCV image
              read, resizing to resize_size, CLAHE contrast enhancement on the L channel,
              and a conversion to an image object compatible with torchvision/functional
              transforms (the implementation uses v2.functional.to_image).
        collate_fn(batch) -> (cropped_images_tensor, labels_tensor):
            - Receives a list of samples (img, label) and uses BaseDataset.collate_fn to
              stack/batch them.
            - Moves batched tensors to self.device, runs YOLO-based ROI selection via
              _get_croped_roi, and returns the cropped image tensor and the label tensor.
            - Note: cropping/inference occurs on the device (GPU if available) for
              efficiency; this may affect multi-GPU setups.
        _preprocess(img_path: str) -> PIL.Image | ImageLike:
            - CPU-side preprocessing for a single image path. Steps:
                1. Read image from disk with OpenCV (BGR).
                2. Resize to self.resize_size.
                3. Convert to Lab color space and apply CLAHE on the L channel.
                4. Merge channels back and convert to RGB.
                5. Convert to an image object expected by torchvision functional transforms.
            - Returns the processed image ready for batching.
        _get_croped_roi(images: torch.Tensor) -> torch.Tensor:
            - Performs batched YOLO inference and crops each image to the configured
              target_size centered on the YOLO-predicted bounding box center.
            - Expected YOLO output behavior (as used by this method):
                * Model output is indexed such that:
                    - x center is at index 0
                    - y center is at index 1
                    - object confidence is available at index 4 across anchors/classes
                * The method selects, per batch item, the anchor/index with maximum
                  confidence (argmax over output[:, 4, :]) and reads the corresponding
                  center coordinates.
            - Steps:
                1. Convert input images to float32 and scale to [0,1], resize to yolo_size.
                2. Run the YOLO model in no-grad mode to obtain output tensors.
                3. For each sample, pick the coordinate pair (x,y) corresponding to the
                   maximum confidence prediction, map those coordinates from yolo_size
                   space back to resize_size space, and compute the top-left corner for
                   cropping using target_size.
                4. Clamp top/left to be >= 0 to avoid negative crop indices.
                5. Crop each original (resized) image at the computed top/left with size
                   target_size using torchvision functional crop and return a batched
                   tensor of stacked cropped images.
            - Returns:
                A torch.Tensor of shape (batch_size, C, target_h, target_w) containing the
                stacked cropped image tensors.
    Notes:
        - This class assumes the DataFrame's "path" entries point to images organized and
          accessible via the dataset's data_dir / BaseDataset logic.
        - Cropping occurs in the collate function on the device to leverage GPU inference.
        - The implementation relies on specific YOLO output indexing; changes to the
          model's output format will require corresponding updates to _get_croped_roi.
        - The class depends on BaseDataset for core dataset behaviors (e.g., folder logic),
          and on torchvision (v2.functional) for conversion, resizing, and cropping ops.
    """
    def __init__(self, resize_size: Tuple[int, int], df: pd.DataFrame, data_dir: str, n = None):
        super().__init__(df, data_dir, n)
        self.resize_size = resize_size

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses a single sample from the dataset.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple (img, label) where:
                - img is the preprocessed image tensor.
                - label is the corresponding label tensor.
        """
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
        x = output[batch_indices, 0, max_conf_indicies]
        y = output[batch_indices, 1, max_conf_indicies]
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

class PrecomputedDataset(RawDataset):
    def __init__(self, new_df, **kwargs):
        # TODO Best to refactor at some point due to code reuse in BaseDataset
        super().__init__(**kwargs)
        self.df = pd.merge(self.df, new_df, on="path")

    def __getitem__(self, idx):
        for c in ["y", "x", "h", "w"]:
            assert c in self.df.columns, \
                f"{c} column not in dataframe. Need to call add_box_df method"
        img, label = super().__getitem__(idx)
        # top, left, height, width = self.df["y"], self.df["x"], self.df["h"], self.df["w"]
        img = v2.functional.crop(img, self.df["y"].iloc[idx], self.df["x"].iloc[idx],
                                 self.df["h"].iloc[idx], self.df["w"].iloc[idx])
        return img, label

class DataModule:
    def __init__(self, csv_path: str, data_path: str,
                 resize_size:  Union[int, Tuple[int, int]] = 2_000,
                 seed: int = 7, precompute: bool = False,
                 precomputed_filepath: Optional[str] = None,
                 yolo: CropROITransform = None,
                 n_sample: Union[int, Tuple[int, int, int, int]] = None):
        self.csv_path = csv_path
        self.data_path = data_path
        self.resize_size = resize_size if isinstance(resize_size, tuple) else resize_size, resize_size
        self.seed = seed
        self.precompute = precompute
        self.precomputed_file = precomputed_filepath
        self.yolo = yolo
        self.n_sample = n_sample if isinstance(n_sample, tuple) or n_sample is None else (n_sample,) * 4

    def load_datasets(self, val_ratio: float = 0.2, test_ratio: float = 0.2,include_real: bool = True, n_sample = None) -> Tuple[BaseDataset, BaseDataset, BaseDataset, Optional[BaseDataset]]:
        df = pd.read_csv(self.csv_path, sep=';')
        train_df, val_df, test_df, real_df = self._split_dataframes(df, val_ratio,
                                                                    test_ratio, include_real)
        if self.n_sample is not None:
            n1 , n2 , n3 , n4  = self.n_sample
            train_df = train_df.sample(n=n1, random_state=self.seed)
            val_df = val_df.sample(n=n2, random_state=self.seed)
            test_df = test_df.sample(n=n3, random_state=self.seed)
            if include_real:
                real_df = real_df.sample(n=n4, random_state=self.seed)
        
        new_df = None
        if self.precompute:
            if os.path.exists(self.precomputed_file):
                new_df = pd.read_csv(self.precomputed_file)
            else:
                new_df = self._compute_boxes_df(df)
                new_df.to_csv(self.precomputed_file, index=False)
    
        return self._create_datasets(df, include_real, new_df, resize_size = self.resize_size,
                                     data_dir = self.data_path, n = n_sample)

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
    
    def _create_datasets(self, df, include_real, new_df = None, **kwargs):
        ds_class = RawDataset
        if new_df is not None:
            ds_class = PrecomputedDataset
            kwargs["new_df"] = new_df
        train = ds_class(df = df, **kwargs)
        val = ds_class(df = df, **kwargs)
        test = ds_class(df = df, **kwargs)
        real = None
        if include_real:
            real = ds_class(df = df, **kwargs)
        return train, val, test, real

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

        assert not os.path.exists(self.precomputed_file), f"{self.precomputed_file} already exists"
        raw_ds = RawDataset(resize_size=self.yolo.yolo_size, df=df, data_dir=self.data_path)
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
