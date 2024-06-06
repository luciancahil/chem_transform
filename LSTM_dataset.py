import selfies as sf
import torch
import torch_geometric
from torch_geometric.data import Dataset
import numpy as np 
import os
import re
import sklearn.utils.class_weight as cw
import LSTM_utils as utils

print(f"Torch version: {torch.__version__}")
print(f"Cudas available: {torch.cuda.device_count()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None, length=0):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        self.length = length
        self.class_weights = []

        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped """
        processed_files = [f for f in os.listdir(self.processed_dir) if not f.startswith("pre")]
        try:
            self.class_weights = torch.load(os.path.join(self.processed_dir, "0weights.pt"))
        except OSError as e: 
            pass
            
        if self.test:
            processed_files = [file for file in processed_files if "test" in file]
            if len(processed_files) == 0:
                return ["no_files.dummy"]
            last_file = sorted(processed_files)[-1]
            index = int(re.search(r'\d+', last_file).group())
            self.length = index + 1
            return [f'data_test_{i}.pt' for i in list(range(0, index))]
        else:
            processed_files = [file for file in processed_files if not "test" in file]
            if len(processed_files) == 0:
                return ["no_files.dummy"]
            last_file = sorted(processed_files)[-1]
            index = int(re.search(r'\d+', last_file).group())
            self.length = index + 1
            return [f'data_{i}.pt' for i in list(range(0, index))]
        

    def download(self):
        pass

    def smiles_to_input(self, smiles):
        input = [0]
        return (input + [utils.token_to_enum[c] for c in smiles])

    def process(self):
        f = open(self.raw_paths[0], 'r')
        chars = [i for i in range(len(utils.token_to_enum))]
        for line in f:
            try:
                array = self.smiles_to_input(line)
                if(len(array) > utils.INPUT_SIZE):
                    continue
            except Exception as e: 
                continue

            length = len(array)
            padding_len = utils.INPUT_SIZE -  length
            padding = [len(utils.enum_to_token)] * padding_len
            array += padding
            chars += array[0:length]

            data = {'x': torch.tensor(array), 'SMILES': line.strip()}
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                f'data_test_{self.length}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                f'data_{self.length}.pt'))
            
            self.length += 1
        
        self.class_weights = torch.tensor(cw.compute_class_weight(class_weight="balanced", classes=np.unique(chars), y=chars), dtype=torch.float)
        if not (self.test):
            torch.save(self.class_weights,
                    os.path.join(self.processed_dir, "0weights.pt"))
        print(f"Done. Stored {self.length} preprocessed molecules.")

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.length

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data