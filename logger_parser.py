import pathlib

SAVE_PATH = pathlib.Path("/root/autodl-fs/Mutil-FL-Training/save")

parterns = {
    "round":""
}

def logger_parser(file):
    with open(file,"r") as f:
        for line in f.readlines():
            pass
    pass


files = []

for file in SAVE_PATH.rglob("*.pt"):
    files.append(file)

for flie in files:
    pass