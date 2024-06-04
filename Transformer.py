import torch
import torch.nn as nn
import utils
device = utils.DEVICE

class Chem_Autoencoder(nn.Module):
    def __init__(self, feature_size = 90, num_token_type = 60, embed_size = 20, large_embed_size = 40, latent_size = 100):
        super(Chem_Autoencoder, self).__init__()
        self.feature_size = feature_size
        self.embed_size = embed_size
        self.num_token_type  = num_token_type
        self.embedding = nn.Embedding(num_token_type, embed_size)
        self.encode_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=2)
        self.batch_norm_one = nn.BatchNorm1d(feature_size * embed_size)
        self.encode_linear_mu = nn.Linear(feature_size * embed_size, latent_size)
        self.encode_linear_logvar = nn.Linear(feature_size * embed_size, latent_size)
        self.relu = nn.LeakyReLU()

        # decoding

        self.decode_linear = nn.Linear(latent_size, feature_size * embed_size)
        self.decode_expansion = nn.Linear(embed_size, large_embed_size)
        self.mlp1 = nn.Linear(large_embed_size, large_embed_size)
        self.mlp2 = nn.Linear(large_embed_size, large_embed_size)
        self.mlp3 = nn.Linear(large_embed_size, large_embed_size)
        self.mlp4 = nn.Linear(large_embed_size, large_embed_size)
        self.mlp5 = nn.Linear(large_embed_size, large_embed_size)
        self.batch_norm_mlp = nn.BatchNorm1d(large_embed_size)
        self.decode_attention = nn.MultiheadAttention(embed_dim=large_embed_size, num_heads=2)
        self.to_one_hot = nn.Linear(large_embed_size, num_token_type)
        self.large_mlp1 = nn.Linear(num_token_type, num_token_type)
        self.final_output = nn.Linear(num_token_type, num_token_type)
        self.batch_norm_large = nn.BatchNorm1d(num_token_type)
        self.dropout = nn.Dropout(p = 0.1)
    

    def encode(self, x, training):
        x = x.to(device)
        self.embedding.to(device)
        x = self.embedding(x)
        x = self.encode_attention(x, x, x)[0]
        x = x.view(-1, self.embed_size * self.feature_size)
        x = self.batch_norm_one(x)
        log_var = self.encode_linear_logvar(x)
        if training:
            x = self.dropout(x)
        x = self.encode_linear_mu(x)
        return x, log_var
    
    def dropout_if_train(self, x, layer, norm, training):
        x = layer(x)
        x = self.relu(x)


        if(training):
            x = self.dropout(x)
        
        return x
    
    def decode(self, x, training):
        x = x.to(device)
        x = self.decode_linear(x)
        x = x.view(-1, self.feature_size, self.embed_size)
        x = self.dropout_if_train(x, self.decode_expansion, self.batch_norm_mlp, training)
        x = self.dropout_if_train(x, self.mlp1, self.batch_norm_mlp, training)
        x = self.dropout_if_train(x, self.mlp2, self.batch_norm_mlp, training)
        x = self.dropout_if_train(x, self.mlp3, self.batch_norm_mlp, training)
        x = self.dropout_if_train(x, self.mlp4, self.batch_norm_mlp, training)
        x = self.dropout_if_train(x, self.mlp5, self.batch_norm_mlp, training)

        x = self.decode_attention(x, x, x)[0]
        x = self.dropout_if_train(x, self.to_one_hot, self.batch_norm_large, training)
        x = self.dropout_if_train(x, self.large_mlp1, self.batch_norm_large, training)
        x = self.final_output(x)
        x = torch.permute(x, (0, 2, 1))
        return x


    def forward(self, x, training = True):
        mu, log_var = self.encode(x, training)
        if(training):
            x = self.reparameterize(mu, log_var)
            x = self.decode(x, training)
        else:
            x = self.decode(mu, training)
        return x, mu, log_var
        
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
