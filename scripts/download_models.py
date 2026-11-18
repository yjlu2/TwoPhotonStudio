import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from install import download_model
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--local'); a=p.parse_args()
    download_model(ROOT/'.local/models/cellpose/cpsam',a.local)

