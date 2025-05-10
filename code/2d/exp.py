from train import main as train_main
from finetune import main as finetune_main


PRETEXT_EPOCHS = 20
FINETUNE_EPOCHS = 20


if __name__ == "__main__":
    train_main(PRETEXT_EPOCHS)
    finetune_main(FINETUNE_EPOCHS)

