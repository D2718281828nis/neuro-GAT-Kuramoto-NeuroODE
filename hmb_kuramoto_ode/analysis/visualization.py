from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(cm,path,class_names=None):
    """Confusion matrix heatmap with per-cell counts and class tick labels.

    Without an annotation, a 2x2 heatmap of two similar colors reads as empty;
    the cell text and explicit ticks make counts legible directly on the plot.
    """
    cm=np.asarray(cm); p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    labels=class_names if class_names is not None else [str(i) for i in range(cm.shape[0])]
    fig,ax=plt.subplots(); im=ax.imshow(cm,cmap="Blues")
    ax.set_xticks(range(cm.shape[1])); ax.set_xticklabels(labels)
    ax.set_yticks(range(cm.shape[0])); ax.set_yticklabels(labels)
    ax.set(xlabel="Predicted",ylabel="True")
    threshold=cm.max()/2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(int(cm[i,j])),ha="center",va="center",fontsize=13,fontweight="bold",
                     color="white" if cm[i,j]>threshold else "black")
    fig.colorbar(im); fig.savefig(p.with_suffix(".png"),dpi=200); fig.savefig(p.with_suffix(".svg")); plt.close(fig)
