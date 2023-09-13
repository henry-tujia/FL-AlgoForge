import argparse
from train import Trainer
import pathlib
import datetime


if __name__ == "__main__":
    FILE_PATH = pathlib.Path(__file__).absolute().parent.parent
    dt = datetime.datetime.now()
    SAVE_PATH = FILE_PATH / "save" / f"""{dt.isoformat()}"""

    pathlib.Path.mkdir(SAVE_PATH, exist_ok=True, parents=True)

    parser = argparse.ArgumentParser(description="get yaml.")
    parser.add_argument(
        "--method",
        type=str,
        default="fedavg",
        help="fedavg or others",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/cpfs/user/haotan2/FL/Mutil-FL-Training-main/yamls/base.yaml",
        help="the path of yaml",
    )

    args = parser.parse_args()

    trainer = Trainer(args.method, pathlib.Path(args.config), SAVE_PATH)
    trainer.run()
