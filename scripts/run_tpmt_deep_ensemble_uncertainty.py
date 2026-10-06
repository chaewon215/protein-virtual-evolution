from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
LABEL = ROOT/"data/processed/tpmt_single_mutants.csv"
FEATURE = ROOT/"data/processed/esmc300m_tpmt/tpmt_variant_features.pt"
META = ROOT/"data/processed/esmc300m_tpmt/tpmt_variant_feature_metadata.csv"
OUT = ROOT/"results/tpmt/deep_ensemble_uncertainty"
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = {"random":"fold_random_5", "position":"fold_position_5"}
MEMBER_SEEDS = [42,43,44,45,46]
EPOCHS = 24
TRAIN_BATCH = 64
PRED_BATCH = 512
LR = 1e-3
WD = 1e-4
DROPOUT = 0.2
REPEATS = 1000
COVERAGES = [0.2,0.4,0.6,0.8,1.0]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class BranchFusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.wt = nn.Sequential(nn.Linear(960,64), nn.ReLU())
        self.local = nn.Sequential(nn.Linear(960,64), nn.ReLU())
        self.global_ = nn.Sequential(nn.Linear(960,64), nn.ReLU())
        self.head = nn.Sequential(
            nn.Dropout(DROPOUT), nn.Linear(192,64), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(64,1)
        )
    def forward(self,x):
        # x [B,2880]
        a = self.wt(x[:,0:960])          # [B,64]
        b = self.local(x[:,960:1920])    # [B,64]
        c = self.global_(x[:,1920:2880]) # [B,64]
        z = torch.cat([a,b,c], dim=1)    # [B,192]
        return self.head(z).squeeze(-1)  # [B]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def loader(X,y,seed):
    ds = TensorDataset(torch.tensor(X,dtype=torch.float32),
                       torch.tensor(y,dtype=torch.float32))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds,batch_size=TRAIN_BATCH,shuffle=True,generator=g)

def train_member(X,y,seed):
    seed_all(seed)
    model = BranchFusionMLP().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    loss_fn = nn.MSELoss()
    dl = loader(X,y,seed)
    final = np.nan
    for _ in range(EPOCHS):
        model.train(); total=0.0; n=0
        for xb,yb in dl:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(xb),yb)
            loss.backward(); opt.step()
            total += loss.item()*len(xb); n += len(xb)
        final = total/n
    return model,float(final)

@torch.inference_mode()
def predict(model,X):
    ds = TensorDataset(torch.tensor(X,dtype=torch.float32))
    dl = DataLoader(ds,batch_size=PRED_BATCH,shuffle=False)
    out=[]; model.eval()
    for (xb,) in dl: out.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(out)

def pred_metrics(y,p):
    return {
        "spearman": float(spearmanr(y,p).statistic),
        "rmse": float(np.sqrt(mean_squared_error(y,p))),
        "mae": float(mean_absolute_error(y,p)),
        "r2": float(r2_score(y,p)),
    }

def unc_metrics(y,mu,sigma):
    err=np.abs(y-mu)
    rho=float(spearmanr(sigma,err).statistic)
    thr=float(np.quantile(err,0.8))
    high=(err>=thr).astype(int)
    return {
        "mean_uncertainty": float(sigma.mean()),
        "uncertainty_error_rho": rho,
        "high_error_auroc": float(roc_auc_score(high,sigma)),
        "high_error_threshold": thr,
    }

def quintiles(y,mu,sigma):
    d=pd.DataFrame({"y":y,"mu":mu,"sigma":sigma})
    d["err"]=np.abs(d.y-d.mu)
    d["q"]=pd.qcut(d.sigma,5,labels=[1,2,3,4,5],duplicates="drop")
    rows=[]
    for q,g in d.groupby("q",observed=True):
        rows.append({
            "quintile":int(q),"n":len(g),
            "uncertainty_mean":float(g.sigma.mean()),
            "mae":float(g.err.mean()),
            "rmse":float(np.sqrt(np.mean((g.y-g.mu)**2))),
        })
    return pd.DataFrame(rows)

def risk_coverage(y,mu,sigma,seed=42):
    n=len(y); rng=np.random.default_rng(seed)
    unc_order=np.argsort(sigma)
    err=np.abs(y-mu); oracle_order=np.argsort(err)
    rows=[]; random_rows=[]
    for cov in COVERAGES:
        m=n if cov==1.0 else max(1,int(np.floor(cov*n)))
        uidx=unc_order[:m]; oidx=oracle_order[:m]
        urmse=float(np.sqrt(mean_squared_error(y[uidx],mu[uidx])))
        ormse=float(np.sqrt(mean_squared_error(y[oidx],mu[oidx])))
        rr=[]
        for rep in range(REPEATS):
            idx=rng.choice(n,size=m,replace=False)
            r=float(np.sqrt(mean_squared_error(y[idx],mu[idx])))
            rr.append(r); random_rows.append({"coverage":cov,"subset_n":m,"repeat":rep,"rmse":r})
        rr=np.asarray(rr)
        mean=float(rr.mean())
        rows.append({
            "coverage":cov,"subset_n":m,
            "uncertainty_selected_rmse":urmse,
            "random_rmse_mean":mean,
            "random_ci_low":float(np.quantile(rr,.025)),
            "random_ci_high":float(np.quantile(rr,.975)),
            "oracle_rmse":ormse,
            "rmse_reduction_vs_random":float((mean-urmse)/mean),
            "random_percentile":float(np.mean(rr<=urmse)),
        })
    return pd.DataFrame(rows),pd.DataFrame(random_rows)

def run_split(name,fold_col,X,df):
    out=OUT/name; out.mkdir(parents=True,exist_ok=True)
    y=df.DMS_score.to_numpy(float); folds=df[fold_col].to_numpy(int)
    member_oof=np.full((len(df),5),np.nan)
    fold_rows=[]; member_rows=[]

    for fold in sorted(np.unique(folds)):
        tr=np.flatnonzero(folds!=fold); te=np.flatnonzero(folds==fold)

        if name=="position":
            if set(df.iloc[tr].position) & set(df.iloc[te].position):
                raise RuntimeError(f"Position leakage in fold {fold}")

        xs=StandardScaler().fit(X[tr])
        Xtr=xs.transform(X[tr]); Xte=xs.transform(X[te])
        ys=StandardScaler().fit(y[tr].reshape(-1,1))
        ytr=ys.transform(y[tr].reshape(-1,1)).ravel()

        preds=[]
        for mi,base in enumerate(MEMBER_SEEDS):
            s=base+int(fold)*1000
            model,loss=train_member(Xtr,ytr,s)
            ps=predict(model,Xte)
            p=ys.inverse_transform(ps.reshape(-1,1)).ravel()
            member_oof[te,mi]=p; preds.append(p)
            member_rows.append({"fold":int(fold),"member":mi,"seed":s,
                                "final_train_loss_scaled":loss,**pred_metrics(y[te],p)})
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        mat=np.column_stack(preds) # [N_test,5]
        mu=mat.mean(1); sigma=mat.std(1,ddof=1)
        pm=pred_metrics(y[te],mu); um=unc_metrics(y[te],mu,sigma)
        fold_rows.append({"fold":int(fold),"n_train":len(tr),"n_test":len(te),**pm,**um})
        print(f"{name.upper()} fold {fold} | rho={pm['spearman']:.4f} "
              f"RMSE={pm['rmse']:.4f} unc-rho={um['uncertainty_error_rho']:.4f} "
              f"AUROC={um['high_error_auroc']:.4f}")

    if np.isnan(member_oof).any(): raise RuntimeError(f"Missing OOF predictions: {name}")
    mu=member_oof.mean(1); sigma=member_oof.std(1,ddof=1)
    pm=pred_metrics(y,mu); um=unc_metrics(y,mu,sigma)

    oof=df[["mutant","position","DMS_score",fold_col]].copy()
    oof["predicted_fitness_mu"]=mu; oof["uncertainty_sigma"]=sigma
    oof["absolute_error"]=np.abs(y-mu)
    for mi in range(5): oof[f"member_{mi}_prediction"]=member_oof[:,mi]

    q=quintiles(y,mu,sigma)
    risk,rnd=risk_coverage(y,mu,sigma,42)

    oof.to_csv(out/"oof_predictions.csv",index=False)
    pd.DataFrame(fold_rows).to_csv(out/"fold_metrics.csv",index=False)
    pd.DataFrame(member_rows).to_csv(out/"member_fold_metrics.csv",index=False)
    q.to_csv(out/"uncertainty_quintiles.csv",index=False)
    risk.to_csv(out/"risk_coverage_baselines.csv",index=False)
    rnd.to_csv(out/"random_selection_repeats.csv",index=False)

    (out/"summary.json").write_text(json.dumps({
        "split":name,"prediction":pm,"uncertainty":um,
        "member_prediction_shape":list(member_oof.shape),
        "mu_shape":list(mu.shape),"sigma_shape":list(sigma.shape),
        "tpmt_specific_hyperparameter_tuning":False,
        "risk_coverage":risk.to_dict(orient="records"),
    },indent=2),encoding="utf-8")

    return {"split":name,**pm,**um}

def main():
    print("="*130)
    print("Step 11-4: TPMT Deep Ensemble Uncertainty External Validation")
    print("="*130)
    print(f"Device={DEVICE} | members=5 | epochs/member={EPOCHS} | TPMT-specific tuning=NO")

    labels=pd.read_csv(LABEL,usecols=["mutant","DMS_score","position","fold_random_5","fold_position_5"])
    meta=pd.read_csv(META,usecols=["feature_index","mutant"])
    df=(meta.merge(labels,on="mutant",how="left",validate="one_to_one")
          .sort_values("feature_index").reset_index(drop=True))
    if df.DMS_score.isna().any(): raise ValueError("Feature/label alignment failed.")

    X=torch.load(FEATURE,map_location="cpu")["X"].float().numpy()
    if X.shape!=(len(df),2880): raise ValueError(f"Unexpected X shape: {X.shape}")
    print(f"X={X.shape} | y={df.DMS_score.shape}")

    rows=[]
    for name,col in SPLITS.items():
        print("\n"+"#"*130+f"\nSPLIT: {name.upper()}\n"+"#"*130)
        rows.append(run_split(name,col,X,df))

    comp=pd.DataFrame(rows)
    comp.to_csv(OUT/"split_comparison.csv",index=False)
    (OUT/"summary.json").write_text(json.dumps({
        "stage":"Step 11-4","protein":"TPMT","external_validation":True,
        "tpmt_specific_hyperparameter_tuning":False,
        "results":comp.to_dict(orient="records")
    },indent=2),encoding="utf-8")

    print("\n"+"="*150)
    print("TPMT Deep Ensemble Uncertainty Summary")
    print("="*150)
    print(comp.to_string(index=False))
    for name in SPLITS:
        q=pd.read_csv(OUT/name/"uncertainty_quintiles.csv")
        r=pd.read_csv(OUT/name/"risk_coverage_baselines.csv")
        print(f"\n{name.upper()} uncertainty quintiles\n{q.to_string(index=False)}")
        print(f"\n{name.upper()} risk coverage\n{r.to_string(index=False)}")
    print(f"\nSaved: {OUT}")
    print("uncertainty_sigma is ensemble disagreement, not a calibrated confidence interval.")

if __name__=="__main__":
    main()
