# writer: 조정효

def clustered_permutation_importance(clf,X,y,clusters,scoring, n_repeats): 
    """
    clf: fitted classifier
    X
    y
    clusters: (dict)
    scoring: 'acc','auc'
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    scr_before, scr_after = pd.Series(), pd.DataFrame(columns=clusters.keys()) 
    for i in range(n_repeats):
        prob=clf.predict_proba(X)[:,1]
        pred=clf.predict(X)
        if scoring == 'auc':
            scr_before.loc[i]=roc_auc_score(y, prob)
        elif scoring == 'acc':
            scr_before.loc[i]=accuracy_score(y, pred)
            
        for  j  in  scr_after.columns:
            X_=X.copy(deep=True) 
            for  k  in  clusters[j]:
                np.random.shuffle(X_[k].values)  #  shufﬂe  cluster 
            prob2=clf.predict_proba(X_)[1]
            pred2=clf.predict(X_)
            if scoring == 'auc':
                scr_after.loc[i,j]=roc_auc_score(y, prob2)
            elif scoring == 'acc':
                scr_after.loc[i,j]=accuracy_score(y, pred2)
    imp=(-1*scr_after).add(scr_before,axis=0)
    imp=imp/(1-scr_after).replace(0, np.nan)
    imp=pd.concat({'mean':imp.mean(),'std':imp.std()*imp.shape[0]**-.5},axis=1)
    imp.index = ['Cluster {}'.format(j) for j in scr_after.columns]
    imp.replace([-np.inf, np.nan], 0, inplace=True)
    return  imp