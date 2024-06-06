import torch
import torch.nn as nn
import LSTM_utils as utils
device = utils.DEVICE
import random

class Chem_Autoencoder(nn.Module):
    def __init__(self, num_chars = utils.INPUT_SIZE, num_token_type = len(utils.token_to_enum), embed_size = 1, kernel_sizes = [9, 9, 10], latent_size = 156):
        super(Chem_Autoencoder, self).__init__()
        self.num_chars = num_chars
        self.embed_size = embed_size
        self.num_token_type  = num_token_type
        # encoding
        self.embedding_layer = nn.Embedding(num_token_type + 1, embed_size)
        self.conv_layers = nn.ModuleList([nn.Conv1d(1, 1, size) for size in kernel_sizes])
        self.embed_latent_mu = nn.Linear(utils.INPUT_SIZE - sum(kernel_sizes) + len(kernel_sizes), latent_size)
        self.embed_latent_logvar = nn.Linear(utils.INPUT_SIZE - sum(kernel_sizes) + len(kernel_sizes), latent_size)

        # decoding
        self.gru = nn.GRU(embed_size, latent_size, batch_first=True) # GRU layer
        self.selection = nn.Linear(latent_size, num_token_type)
        self.dropout = nn.Dropout(p = 0.5)
    

    def encode(self, x, training):
        x = self.embedding_layer(x).squeeze(2)
        x = x.unsqueeze(1)

        for layer in self.conv_layers:
            x = self.dropout_if_train(x, layer, training)
        x = x.squeeze(1)
        log_var = self.embed_latent_logvar(x)
        x = self.embed_latent_mu(x)

        
        return x, log_var
    
    def dropout_if_train(self, x, layer, training):
        x = layer(x)

        if(training):
            x = self.dropout(x)
        
        return x
    
    def run_grus(self, token,  hidden):
        out, hidden = self.gru(token, hidden)
        out = self.selection(out)
        return out, hidden

    def decode(self, hidden, tokens, teacher_force_ratio):
        token = self.embedding_layer(tokens[:,0])
        first_output = torch.zeros((tokens.shape[0], 1, self.num_token_type))
        first_output[:, :, 0] = 1
        outputs = [first_output]


        hidden = hidden.unsqueeze(0)
        token = token.unsqueeze(1)

        for i in range(1, self.num_chars):
            out, hidden = self.run_grus(token, hidden)
            outputs.append(out)

            if random.uniform(0, 1) <= teacher_force_ratio and i < self.num_chars - 1:

                token = tokens[:, i + 1].unsqueeze(1)
            else:
                token = out.argmax(dim=2)  # Choose the token with the highest probability

            
            token = self.embedding_layer(token)

        outputs = torch.cat(outputs, dim = 1)
        
        return outputs

    def forward(self, tokens, training = True, teacher_force_ratio = 0.0):
        mu, log_var = self.encode(tokens, training)

        if(training):
            latent = self.reparameterize(mu, log_var)
        else:
            latent = mu
        
        
        outputs = self.decode(latent, tokens, teacher_force_ratio)
        
        return outputs, mu, log_var



        
        
    def reparameterize(self, x, log_var):
        # Get standard deviation
        std = torch.exp(log_var)
        # Returns random numbers from a normal distribution
        eps = torch.randn_like(std)
        # Return sampled values
        return eps.mul(std).add_(x)
        
    
    def interpolate(self, data_list, num_points):
        """
        make a list of latents.
        For each pair of latents, sample n points between them.

        See if a molecule is valid.
        If so, add the new_smiles to the smiles list. 
        return the smiles list.
        """
        latents = [self.encode(data) for data in data_list]

        num_latents = len(latents)
        alpha = 1./ num_points
        new_smiles = []


        for i in range(num_latents):
            print("I: " + str(i))
            for j in range(i + 1, num_latents):
                diff = torch.sub(latents[i], latents[j])

                for a in range(1, num_points):
                    new = torch.sub(latents[i], diff, alpha = alpha * a)
                    new_mol = self.generate_mol(new)
                    if new_smiles and "." not in new_smiles:
                        new_smiles.append(new_mol)
        
        print(new_smiles)
