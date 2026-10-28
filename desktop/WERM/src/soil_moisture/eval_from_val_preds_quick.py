from pathlib import Path
import pandas as pd
import numpy as np

CSV = Path("/content/drive/MyDrive/WERM/outputs/runs/AA_70_30_ep30_20260127_044224/val_preds_best.csv")

def confusion(y_true, y_pred, n=3):
    cm = np.zeros((n,n), dtype=int)
    for t,p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def macro_f1(y_true, y_pred, n=3):
    f1s = []
    for c in range(n):
        tp = np.sum((y_true==c) & (y_pred==c))
        fp = np.sum((y_true!=c) & (y_pred==c))
        fn = np.sum((y_true==c) & (y_pred!=c))
        prec = tp/(tp+fp+1e-12)
        rec  = tp/(tp+fn+1e-12)
        f1 = 2*prec*rec/(prec+rec+1e-12)
        f1s.append(f1)
    return float(np.mean(f1s)), [float(x) for x in f1s]

def main():
    df = pd.read_csv(CSV)
    y = df["gt"].to_numpy().astype(int)
    p = df["pred"].to_numpy().astype(int)

    acc = float((y==p).mean())
    mf1, per = macro_f1(y,p,3)
    cm = confusion(y,p,3)

    print("VAL accuracy:", acc)
    print("VAL macro-F1:", mf1, "per_class:", per)
    print("Confusion matrix (rows=gt, cols=pred):")
    print(cm)

    # Wetness threshold sweep (ignore gt=1 to get Dry-like vs Wet-like)
    sub = df[df["gt"].isin([0,2])].copy()
    if len(sub) > 0:
        yb = (sub["gt"].to_numpy() == 2).astype(int)
        W = sub["W"].to_numpy()
        print("\nW threshold sweep (gt in {0,2} only):")
        for t in [0.3,0.4,0.5,0.6,0.7]:
            predb = (W >= t).astype(int)
            a = float((predb==yb).mean())
            print(f"t={t:.1f} acc={a:.3f} n={len(yb)}")
    else:
        print("\nNo gt in {0,2} subset found for W sweep.")

if __name__ == "__main__":
    main()
