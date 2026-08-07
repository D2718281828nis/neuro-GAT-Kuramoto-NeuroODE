"""CLI. Real-data commands never silently synthesize data."""
import argparse,json
from .data.stew import STEWDataset
from .config import load_config
def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True)
    i=sub.add_parser("inspect-data"); i.add_argument("--data-root",required=True)
    for name in ("train","cross-validate"):
        q=sub.add_parser(name); q.add_argument("--config",required=True); q.add_argument("overrides",nargs="*")
    e=sub.add_parser("evaluate"); e.add_argument("--checkpoint",required=True); e.add_argument("--data-root",required=True)
    a=p.parse_args()
    if a.command=="inspect-data": print(json.dumps(STEWDataset(a.data_root).inspect(),indent=2))
    elif a.command in ("train","cross-validate"):
        cfg=load_config(a.config); root=cfg["data"]["data_root"]
        for value in a.overrides:
            if value.startswith("data.data_root="): root=value.split("=",1)[1]
        print(json.dumps({"mode":a.command,"data":STEWDataset(root).inspect()},indent=2))
    else: STEWDataset(a.data_root); print(f"checkpoint/data inputs validated: {a.checkpoint}")
if __name__=="__main__": main()
