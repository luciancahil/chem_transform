import torch

token_to_enum = {'[START]': 0, 'C': 1, '\n': 2}
enum_to_token = ['[START]', 'C', '\n']
INPUT_SIZE = 30
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
