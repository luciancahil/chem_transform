import selfies as sf
import torch
import torch_geometric
from torch_geometric.data import Dataset
import numpy as np 
import os
import re
import utils

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
    
        if self.test:
            processed_files = [file for file in processed_files if "test" in file]
            if len(processed_files) == 0:
                return ["no_files.dummy"]
            last_file = sorted(processed_files)[-1]
            index = int(re.search(r'\d+', last_file).group())
            self.length = index
            return [f'data_test_{i}.pt' for i in list(range(0, index))]
        else:
            processed_files = [file for file in processed_files if not "test" in file]
            if len(processed_files) == 0:
                return ["no_files.dummy"]
            last_file = sorted(processed_files)[-1]
            index = int(re.search(r'\d+', last_file).group())
            self.length = index
            return [f'data_{i}.pt' for i in list(range(0, index))]
        

    def download(self):
        pass

    def smiles_to_input(self, smiles):
        encoded_selfies = sf.encoder(smiles.strip())
        symbols = list(sf.split_selfies(encoded_selfies))

        return [utils.token_to_enum[symbol] for symbol in symbols]

    def process(self):
        f = open(self.raw_paths[0], 'r')

        for line in f:
            try:
                array = self.smiles_to_input(line)
                if(len(array) > utils.INPUT_SIZE):
                    continue
            except(KeyError):
                continue
            
            padding_len = utils.INPUT_SIZE -  len(array)
            padding = [0] * padding_len
            array += padding
                
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