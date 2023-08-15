# Importing the libraries
from model import TransformerModel
from lightning_utils import TransformerLightning, MakeDM
import lightning as L
from pytorch_lightning.loggers import CSVLogger
import torch

def main():
    # Hyperparameters and settings
    data_dir = "data//hp_all.txt"
    vocab_size = 92
    n_embed = 512
    n_heads = 16
    block_size = 32
    n_layers = 8
    dropout_rate = 0.1
    num_workers = 2
    batch_size = 512
    learning_rate = 1e-3

    # Creating the model
    transformer = TransformerModel(vocab_size, n_embed, n_heads, block_size, n_layers, dropout_rate)

    # Creating the data module for lightning
    dm = MakeDM(data_dir=data_dir, batch_size=batch_size, 
                num_workers=num_workers, block_size=block_size)

    # Preparing the data
    dm.prepare_data()

    # Creating the model for lightning
    lightning_model = TransformerLightning(model=transformer, learning_rate=learning_rate, vocab_size=vocab_size)


    # Creating the training loop
    trainer = L.Trainer(
        fast_dev_run=1,
        max_epochs=15,
        accelerator="cpu" if not torch.cuda.is_available() else "gpu",
        devices="auto",
        deterministic=True,
        logger=CSVLogger("logs/", name="hp-tf"),
        strategy="ddp_notebook" if torch.cuda.is_available() else None
    )

    # Fitting the model
    trainer.fit(model=lightning_model, datamodule=dm)

    # # Saving the model weights
    # torch.save(transformer.state_dict(), "model//model_weights.pth")

if __name__ == "__main__":
    main()