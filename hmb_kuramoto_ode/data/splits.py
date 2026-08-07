from sklearn.model_selection import GroupKFold
def grouped_folds(labels,subjects,n_splits=3): return list(GroupKFold(n_splits=n_splits).split(labels,labels,groups=subjects))
def assert_disjoint(*groups):
    sets=[set(g) for g in groups]
    if any(sets[i]&sets[j] for i in range(len(sets)) for j in range(i+1,len(sets))): raise ValueError("subject leakage: split subject IDs overlap")
