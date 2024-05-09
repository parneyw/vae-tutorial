from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.transforms import v2
from model import BVAE, VAEConfig

@dataclass
class TrainConfig:
    lr: float
    weight_decay: float
    num_epochs: int
    batch_size: int

TMP_CHKPT_PATH = '.tmp/'
DEFAULT_VAE_CONFIG = VAEConfig(input_dim=784, hidden_dim=512, latent_dim=2, act_fn=nn.Tanh())
DEFAULT_TRAIN_CONFIG = TrainConfig(1e-3, 1e-2, 10, 128)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Activation function string to module
AF_STOM = {
    'Tanh()': nn.Tanh(),
    'ReLU()': nn.ReLU(),
    'GeLU()': nn.GELU(),
    'SiLU()': nn.SiLU(),
}

def act_fn_lookup(af_str) -> nn.Module:
    """ Wrapper for dictionary lookup for cleanliness. """
    return AF_STOM[af_str]

def train(model: BVAE, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
          prev_updates: int, writer=None) -> int:
    """
    Trains the model on the given data.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader.
        loss_fn: The loss function.
        optimizer: The optimizer.
    """
    model.train()  # Set the model to training mode
    
    for batch_idx, (data, _) in enumerate(tqdm(dataloader)):
        n_upd = prev_updates + batch_idx
        
        data = data.to(device)
        
        optimizer.zero_grad()  # Zero the gradients
        
        output = model(data, 0.5)  # Forward pass
        loss = output.loss
        
        loss.backward()
        
        if n_upd % 100 == 0:
            # Calculate and log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
        
            print(f'Step {n_upd:,} (N samples: {n_upd*dataloader.batch_size:,}), Loss: {loss.item():.4f} (Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) Grad: {total_norm:.4f}')

            if writer is not None:
                global_step = n_upd
                writer.add_scalar('Loss/Train', loss.item(), global_step)
                writer.add_scalar('Loss/Train/BCE', output.loss_recon.item(), global_step)
                writer.add_scalar('Loss/Train/KLD', output.loss_kl.item(), global_step)
                writer.add_scalar('GradNorm/Train', total_norm, global_step)
            
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    
        
        optimizer.step()  # Update the model parameters
        
    return prev_updates + len(dataloader)

def test(model: BVAE, mcfg: VAEConfig, dataloader: torch.utils.data.DataLoader,
         cur_step: int, writer=None) -> float:
    """
    Tests the model on the given data.
    
    Args:
        model (nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): The data loader.
        cur_step (int): The current step.
        writer: The TensorBoard writer.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0
    
    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc='Testing'):
            data = data.to(device)
            data = data.view(data.size(0), -1)  # Flatten the data
            
            output = model(data, 0.5, compute_loss=True)  # Forward pass
            
            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.loss_kl.item()
            
    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(f'====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})')
    
    if writer is not None:
        writer.add_scalar('Loss/Test', test_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/BCE', test_recon_loss, global_step=cur_step)
        writer.add_scalar('Loss/Test/KLD', test_kl_loss, global_step=cur_step)
        
        # Log reconstructions
        writer.add_images('Test/Reconstructions', output.x_recon.view(-1, 1, 28, 28), global_step=cur_step)
        writer.add_images('Test/Originals', data.view(-1, 1, 28, 28), global_step=cur_step)

        # Log 16 random samples from the latent space
        z = torch.randn(16, mcfg.latent_dim).to(device)
        samples = model.decode(z)
        writer.add_images('Test/Samples', samples.view(-1, 1, 28, 28), global_step=cur_step)

    return test_loss

def save_chkpt(model: nn.Module, mcg: VAEConfig, tcg: TrainConfig, optimizer: torch.optim.Optimizer,
               epoch:int, step:int, loss:float, path: Path, exist_ok: bool, log:bool = True) -> None:
    """ Save a training run checkpoint. """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not exist_ok and path.exists():
        raise(FileExistsError)
    chkpt = {
        'epoch':epoch,
        'step':step,
        'model':model.state_dict(),
        'model_config':mcg,
        'train_config':tcg,
        'optimizer':optimizer.state_dict(),
        'loss':loss,
    }
    torch.save(chkpt, path)

    if log:
        print(f"Checkpoint saved to {path}.")


def main(args) -> None:
    print("Setting things up...")
    tcg = TrainConfig(args.lr, args.weight_decay, args.num_epochs, args.batch_size)
    mcg = VAEConfig(
        input_dim=784,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        act_fn=act_fn_lookup(args.act_fn),
    )

    transform = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Lambda(lambda x: x.view(-1) - 0.5),
    ])

    # Download and load the training data
    train_data = datasets.MNIST(
        '~/.pytorch/MNIST_data/', 
        download=True, 
        train=True, 
        transform=transform,
    )
    # Download and load the test data
    test_data = datasets.MNIST(
        '~/.pytorch/MNIST_data/', 
        download=True, 
        train=False, 
        transform=transform,
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=tcg.batch_size, 
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=tcg.batch_size,
        shuffle=False,
    )

    model = BVAE(mcg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcg.lr, weight_decay=tcg.weight_decay)

    run_name = f'vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    path = f'../runs/mnist/{run_name}'
    writer = SummaryWriter(path)

    test_loss = 0
    prev_updates = 0
    for epoch in range(tcg.num_epochs):
        print(f'Epoch {epoch+1}/{tcg.num_epochs}')
        prev_updates = train(model, train_loader, optimizer, prev_updates, writer=writer)
        test_loss = test(model, mcg, test_loader, prev_updates, writer=writer)

    writer.add_hparams(
        {'lr':tcg.lr, 'weight_decay':tcg.weight_decay,
            'hidden_dim':mcg.hidden_dim, 'latent_dim':mcg.latent_dim,
            'act_fn':str(mcg.act_fn)},
        {'hparam/epochs':tcg.num_epochs,'hparam/loss':test_loss, 'hparam/loss-per-px':test_loss/mcg.input_dim},
    )

    try:
        save_chkpt(model, mcg, tcg, optimizer, tcg.num_epochs, prev_updates, test_loss, Path(path+'/chkpt.pt'), exist_ok=False)
    except FileExistsError as fe:
        print(fe)
        print(f"Saving to {TMP_CHKPT_PATH+run_name+'/chkpt.pt'}")
        save_chkpt(model, mcg, tcg, optimizer, tcg.num_epochs, prev_updates, test_loss, Path(TMP_CHKPT_PATH+run_name+'/chkpt.pth'), exist_ok=False)        

if __name__=="__main__":
    dft = DEFAULT_TRAIN_CONFIG
    dcg = DEFAULT_VAE_CONFIG
    parser = ArgumentParser(prog='mnist-vae-train')
    parser.add_argument('--lr', help='Learning rate.', type=float, default=dft.lr)
    parser.add_argument('--weight_decay', help='Weight decay for AdamW.', type=float, default=dft.weight_decay)
    parser.add_argument('--num_epochs', help='Number of epochs to train for.', type=int, default=dft.num_epochs)
    parser.add_argument('--batch_size', help='Batch size, number of images to load per training step.', type=int, default=dft.batch_size)
    parser.add_argument('--hidden_dim', help='Dimensionality of hidden vectors.', type=int, default=dcg.hidden_dim)
    parser.add_argument('--latent_dim', help='Dimensionality of latent vectors.', type=int, default=dcg.latent_dim)
    parser.add_argument('--act_fn', help='Activation function to use on hidden layers of VAE.', type=str, default=str(dcg.act_fn))
    args = parser.parse_args()
    main(args)