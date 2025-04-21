import os
from torch.utils.data import Dataset
import torch

from dataset.hme_ink import *

class HMEDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Initializes the HMEDataset class.
        
        Args:
        - root_dir (str):                 The root directory where the dataset is located.
        - split (str, optional):          The split of the dataset to be used, defaults to 'train'.
        - transform (callable, optional): An optional transform to be applied on a sample, defaults to None.
        """
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.split = split

        # array of inkml file paths
        # self.ink_files = sorted(os.path.join(root_dir, split, "*.inkml"))     
        self.ink_files = [f for f in os.listdir(self.split_dir)]
        # print(self.ink_files)

    def __len__(self):
        return len(self.ink_files)

    def __getitem__(self, idx):
        ink_file = os.path.join(self.split_dir, self.ink_files[idx])
        ink = read_inkml_file(ink_file)                 # returns file w/ strokes and annotations (contains the label)
        ink_feature_vector = self.extract_features(ink) # returns feature tensor
        
        if self.transform: ink_feature_vector = self.transform(ink_feature_vector)
                
        return ink_feature_vector, ink.annotations['normalizedLabel']
    
    def extract_features(self, ink: Ink):
        """
        Convert Ink object into stroke features
        
        Args:
        - stroke_data (np.ndarray): Array of shape (N, 3) containing (x, y, t) coordinates
            
        Returns:
            torch.Tensor: Tensor of shape (N-1, feature_dim) containing extracted features
        """

        # [[[x], [y], [t]], [[x], [y], [t]]]
        ink_features = []

        # find global min/max for normalization
        all_x = []
        all_y = []
        for stroke in ink.strokes:
            all_x.extend(stroke[0])
            all_y.extend(stroke[1])
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        # process each stroke
        for stroke_idx, stroke in enumerate(ink.strokes):
            x_values = stroke[0]
            y_values = stroke[1]
            t_values = stroke[2]
            
            # normalize coordinates
            x_normalized = [(x - x_min) / (x_max - x_min + 1e-8) for x in x_values]
            y_normalized = [(y - y_min) / (y_max - y_min + 1e-8) for y in y_values]
            
            # process points within this stroke
            for i in range(1, len(x_values)):
                # compute derivatives (rate of change)
                dx = x_normalized[i] - x_normalized[i-1]
                dy = y_normalized[i] - y_normalized[i-1]
                dt = t_values[i] - t_values[i-1]
                
                # compute direction (angle in radians, converted to sin and cos components)
                direction = math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0
                dir_sin = math.sin(direction)
                dir_cos = math.cos(direction)
                
                # compute speed
                speed = math.sqrt(dx**2 + dy**2) / (dt + 1e-8)
                
                # first point of stroke flag (not the first point of the first stroke)
                is_first_point = 1.0 if i == 1 and stroke_idx > 0 else 0.0
                
                # normalized coordinates
                x_norm = x_normalized[i]
                y_norm = y_normalized[i]
                
                # curve information
                curvature = 0.0
                if i >= 2:
                    prev_direction = math.atan2(
                        y_normalized[i-1] - y_normalized[i-2], 
                        x_normalized[i-1] - x_normalized[i-2]
                    ) if (x_normalized[i-1] != x_normalized[i-2] or y_normalized[i-1] != y_normalized[i-2]) else 0
                    
                    # angle difference
                    angle_diff = direction - prev_direction
                    while angle_diff > math.pi:
                        angle_diff -= 2 * math.pi
                    while angle_diff < -math.pi:
                        angle_diff += 2 * math.pi
                        
                    curvature = angle_diff / (dt + 1e-8)
                
                # stroke id feature to help the model distinguish between strokes
                stroke_id = float(stroke_idx)
                
                # combine all features
                feature_vector = [
                    dx, dy, dt,            # deltas
                    dir_sin, dir_cos,      # direction (as sin/cos to avoid discontinuity)
                    speed,                 # speed
                    curvature,             # curvature
                    is_first_point,        # new stroke indicator
                    x_norm, y_norm,        # absolute position (normalized)
                    stroke_id              # stroke identifier
                ]
                
                ink_features.append(feature_vector)
            
            # if not the last stroke, add a "pen-up" feature vector between strokes
            if stroke_idx < len(ink.strokes) - 1:
                # get the last point of current stroke and first point of next stroke
                last_x = x_normalized[-1]
                last_y = y_normalized[-1]
                next_stroke = ink.strokes[stroke_idx + 1]
                
                # normalize the first point of next stroke
                next_x = (next_stroke[0][0] - x_min) / (x_max - x_min + 1e-8) 
                next_y = (next_stroke[1][0] - y_min) / (y_max - y_min + 1e-8)
                
                # time difference (this might be larger than normal)
                time_gap = next_stroke[2][0] - t_values[-1]
                
                # create a special feature vector indicating pen-up between strokes
                pen_up_feature = [
                    next_x - last_x,        # position jump
                    next_y - last_y,
                    time_gap,
                    0.0, 0.0,               # no direction (pen is up)
                    0.0,                    # no speed (pen is up)
                    0.0,                    # no curvature
                    1.0,                    # definite stroke boundary
                    next_x, next_y,         # position of start of next stroke
                    float(stroke_idx + 0.5) # between strokes
                ]
                
                ink_features.append(pen_up_feature)
        
        # convert vector list into tensor
        if not ink_features: return torch.zeros((0, 11), dtype=torch.float32) # if no tensor
        
        features_tensor = torch.tensor(ink_features, dtype=torch.float32)
        
        # normalize features to have zero mean and unit variance for non-binary features
        non_binary_indices = [0, 1, 2, 3, 4, 5, 6, 10]  # all except is_first_point and x, y position
        
        mean = features_tensor[:, non_binary_indices].mean(dim=0)
        std = features_tensor[:, non_binary_indices].std(dim=0) + 1e-8
        
        features_tensor[:, non_binary_indices] = (features_tensor[:, non_binary_indices] - mean) / std
        
        return features_tensor