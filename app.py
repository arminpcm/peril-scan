import hydra
from omegaconf import OmegaConf
from lib.application.application import Application


@hydra.main(config_name="config", config_path=".", version_base=None)
def main(cfg: OmegaConf) -> None:
    app = Application(config=cfg)
    app.run()


if __name__ == "__main__":
    main()
