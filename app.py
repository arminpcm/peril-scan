import hydra
from omegaconf import OmegaConf
from lib.application.application import Application
from hydra.core.global_hydra import GlobalHydra


@hydra.main(config_name="config", config_path=".", version_base=None)
def main(cfg: OmegaConf) -> None:
    # Clear Hydra's global state if already initialized
    if GlobalHydra().is_initialized():
        GlobalHydra().clear()

    app = Application(config=cfg)
    app.run()


if __name__ == "__main__":
    main()
