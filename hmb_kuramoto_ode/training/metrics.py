import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix
def classification_metrics(y,prob):
    pred=np.asarray(prob).argmax(1); y=np.asarray(y); out={"accuracy":accuracy_score(y,pred),"balanced_accuracy":balanced_accuracy_score(y,pred),"macro_f1":f1_score(y,pred,average="macro"),"confusion_matrix":confusion_matrix(y,pred).tolist()}
    if len(np.unique(y))==2: out.update(auroc=roc_auc_score(y,np.asarray(prob)[:,1]),auprc=average_precision_score(y,np.asarray(prob)[:,1]))
    out["ece"]=float(np.mean(np.abs(np.asarray(prob).max(1)-(pred==y)))); return out
def link_metrics(y,score): return {"auroc":roc_auc_score(y,score),"average_precision":average_precision_score(y,score)}
