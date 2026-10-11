from argparse import Namespace

from main import run_on_setting


if __name__ == "__main__":
    args = Namespace(
        dataset="Tiny",
        global_config="./config/granular_config/global.json",
    )
    result = run_on_setting(
        args,
        device="cpu",
        epochs=1,
        batch_size=-1,
        latent_dim=4,
        p=2,
        learning_rate=1e-3,
    )
    print("Tiny test ACC:", result["ACC"])
