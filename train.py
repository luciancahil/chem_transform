from Transformer import Chem_Autoencoder
import utils
import torch
import selfies as sf
from dataset import MoleculeDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import torch.nn as nn

device = utils.DEVICE

feature_size = utils.INPUT_SIZE
num_token_type = 60
embed_size = 3
latent_size = 10
bottleneck_size = 5
train_dataset = MoleculeDataset(root="data/", filename="Train.csv")
test_dataset = MoleculeDataset(root="data/", filename="Test.csv", test=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = Chem_Autoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
kl_beta = 0.5

def calc_recon_loss(inputs, targets):
    cross = nn.CrossEntropyLoss()
    largest = torch.max(inputs, dim = 1)[1]
    corrects = torch.all(largest == targets, dim = 1)
    percentage_equal = torch.sum(corrects).item() / largest.shape[0]
    return cross(inputs, targets), percentage_equal

def kl_loss(mu=None, logstd=None):
    """
    Closed formula of the KL divergence for normal distributions
    """
    MAX_LOGSTD = 10
    logstd = logstd.clamp(max=MAX_LOGSTD)
    kl_div = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - torch.exp(2 * logstd), dim=1))

    # Limit numeric errors
    kl_div = kl_div.clamp(max=1000)
    return kl_div
 

def loss_fn(preds, targets, mus, log_vars):
    recon_loss, percent_equal = calc_recon_loss(preds, targets)
    kl = kl_loss(mus, log_vars)
    return kl*kl_beta + recon_loss, kl, percent_equal

def run_one_epoch(data_loader, curr_type, epoch):
    # Store per batch loss and accuracy 
    all_losses = []
    all_kldivs = []

    # Iterate over data loader
    for _, batch in enumerate(tqdm(data_loader)):
        accuracy = []
        # Some of the data points have invalid adjacency matrices 
        try:
            # Use GPU
            """  if(torch.cuda.device_count() > 0):
                batch = batch.to(f'cuda:{model.device_ids[0]}') """
            # Reset gradients
            optimizer.zero_grad() 
            # Call model
            outputs, mu, logvar = model(batch['x'], curr_type == 'Train') 

            loss, kl_div, percent_equal = loss_fn(outputs, batch['x'], mu, logvar,)
            accuracy.append(percent_equal)
            if curr_type == "Train":
                loss.backward()  
                optimizer.step() 
                scheduler.step(loss)

            # Store loss and metrics
            all_losses.append(loss.detach().cpu().numpy())
            #all_accs.append(acc)
            all_kldivs.append(kl_div.detach().cpu().numpy())
        except IndexError as error:
            # For a few graphs the edge information is not correct
            # Simply skip the batch containing those
            print("Error: ", error)
    
    # Perform sampling
    """ if curr_type == "Test":
        generated_mols = model.sample_mols(num=10000)
        print(f"Generated {generated_mols} molecules.")
        mlflow.log_metric(key=f"Sampled molecules", value=float(generated_mols), step=epoch) """
    print
    print(f"{curr_type} epoch {epoch} loss: ", np.array(all_losses).mean())
    print(f"{curr_type} epoch {epoch} accuracy: ", str(sum(accuracy)/len(accuracy)))

"""     mlflow.log_metric(key=f"{curr_type} Epoch Loss", value=float(np.array(all_losses).mean()), step=epoch)
    mlflow.log_metric(key=f"{curr_type} KL Divergence", value=float(np.array(all_kldivs).mean()), step=epoch)
    mlflow.pytorch.log_model(model, "model") """

for epoch in range(201): 
    model.train()
    run_one_epoch(train_loader, curr_type="Train", epoch=epoch)
    if epoch % 5 == 0:
        print("Start test epoch...")
        model.eval()
        run_one_epoch(test_loader, curr_type="Test", epoch=epoch)
    
torch.save(model, "model.sh")



