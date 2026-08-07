from pathlib import Path
import matplotlib.pyplot as plt
def save_confusion_matrix(cm,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); fig,ax=plt.subplots(); im=ax.imshow(cm,cmap="Blues"); ax.set(xlabel="Predicted",ylabel="True"); fig.colorbar(im); fig.savefig(p.with_suffix(".png"),dpi=200); fig.savefig(p.with_suffix(".svg")); plt.close(fig)
