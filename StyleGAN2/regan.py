import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class Regan_training(nn.Module):

    def __init__(self, model, sparsity, train_on_sparse=False):
        super(Regan_training, self).__init__()

        self.model = model
        self.sparsity = sparsity
        self.train_on_sparse = train_on_sparse
        self.layers = []
        self.masks = []
        layers = list(self.model.named_parameters())

        for i in range(0, len(layers)):
            w = layers[i]
            if "to_rgb" not in w[0]:
                self.layers.append(w[1])
        self.reset_masks()

    def get_latent(self, input):
        return self.model.style(input)

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.model.style_dim, device=self.model.input.input.device
        )
        latent = self.model.style(latent_in).mean(0, keepdim=True)

        return latent

    def reset_masks(self):
        for w in self.layers:
            mask_w = torch.ones_like(w, dtype=bool)
            self.masks.append(mask_w)

        return self.masks
    
    def regrow_weights_by_gradient(self, regrow_fraction=0.1):
        for i, (w, m) in enumerate(zip(self.layers, self.masks)):
            if w.grad is None:
                continue

            grad = torch.abs(w.grad.detach())
            num_weights = grad.numel()
            k = int(num_weights * regrow_fraction)

            # Flatten gradient and get top-k indices
            topk = torch.topk(grad.view(-1), k=k)
            topk_indices = topk.indices

            # Create a regrow mask with same shape as weights
            regrow_mask = torch.zeros_like(w, dtype=torch.bool).view(-1)
            regrow_mask[topk_indices] = True
            regrow_mask = regrow_mask.view_as(w)

            # Update the binary mask (set those bits to False = unmasked = regrow)
            self.masks[i][regrow_mask] = False

    def update_masks(self):

        for i, w in enumerate(self.layers):
            q_w = torch.quantile(torch.abs(w), q=self.sparsity)
            mask_w = torch.where(torch.abs(w) < q_w, True, False)

            self.masks[i] = mask_w

    def turn_training_mode(self, mode):
        if mode == 'dense':
            self.train_on_sparse = False
        else:
            self.train_on_sparse = True
            self.update_masks()

    def apply_masks(self):
        for w, mask_w in zip(self.layers, self.masks):
            w.data[mask_w] = 0
            w.grad.data[mask_w] = 0

    def forward(self, x, return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,):
        return self.model(x, return_latents,
        inject_index,
        truncation,
        truncation_latent,
        input_is_latent,
        noise,
        randomize_noise)


    def save_generator(self, filepath: str):
        """
        Save only the generator's weights to disk.
        """
        # `self.model` is your full GAN; assume generator is `self.model.generator`
        torch.save(self.model.generator.state_dict(), filepath)
        print(f"Saved generator to {filepath}")

    @staticmethod
    def load_generator(filepath: str, device: torch.device, generator_cls, *args, **kwargs):
        """
        Load a standalone generator for inference.

        Args:
            filepath: path to .pth file
            device: torch.device('cpu') or 'cuda'
            generator_cls: the class of your generator (must match training definition)
            *args, **kwargs: any init args for generator_cls
        Returns:
            gen: an instance of generator_cls with loaded weights and set to eval mode
        """
        # 1) instantiate the generator
        gen = generator_cls(*args, **kwargs).to(device)
        # 2) load weights
        state = torch.load(filepath, map_location=device)
        gen.load_state_dict(state)
        # 3) set to eval and turn off grads
        gen.eval()
        for p in gen.parameters():
            p.requires_grad = False
        print(f"Loaded generator from {filepath}")
        return gen
