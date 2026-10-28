import pandas as pd
from pathlib import Path

CSV = Path("/content/drive/MyDrive/WERM/outputs/fusion/fusion_results.csv")
OUT = Path("/content/drive/MyDrive/WERM/outputs/fusion/fusion_audit.csv")

df = pd.read_csv(CSV)

# 1) saturation / cliff checks
sat_tillage = (df["tillage_norm"] >= 0.999).mean()
floor_tillage = (df["tillage_norm"] <= 0.051).mean()
zero_risk = (df["risk_score"] <= 1e-12).mean()

# 2) redundancy checks (if tillage_eff is just residue, you’ll see it here)
corr = df[["moisture_norm","residue","tillage_eff","risk_score"]].corr()

# 3) quick “reason codes” for top/bottom
def reason_row(r):
    # simple, interpretable factors
    return {
        "wet": float(r["moisture_norm"]),
        "warm": float(r["gdd_norm"]),
        "low_residue": float(1.0 - r["residue"]),
        "tillage_eff": float(r["tillage_eff"]),
    }

top = df.head(10).copy()
bot = df.tail(10).copy()

top_reasons = [reason_row(r) for _, r in top.iterrows()]
bot_reasons = [reason_row(r) for _, r in bot.iterrows()]

top_reason_df = pd.DataFrame(top_reasons)
bot_reason_df = pd.DataFrame(bot_reasons)

top_out = pd.concat([top.reset_index(drop=True), top_reason_df], axis=1)
bot_out = pd.concat([bot.reset_index(drop=True), bot_reason_df], axis=1)

audit = pd.concat([
    pd.DataFrame([{
        "metric": "sat_tillage_norm(>=0.999)",
        "value": sat_tillage
    },{
        "metric": "floor_tillage_norm(<=0.051)",
        "value": floor_tillage
    },{
        "metric": "zero_risk(<=1e-12)",
        "value": zero_risk
    }]),
], axis=0)

OUT.parent.mkdir(parents=True, exist_ok=True)
audit.to_csv(OUT, index=False)

print("Saved audit:", OUT)
print("\n=== Saturation / cliff ===")
print("sat_tillage_norm:", sat_tillage)
print("floor_tillage_norm:", floor_tillage)
print("zero_risk:", zero_risk)

print("\n=== Correlation (sanity) ===")
print(corr)

print("\n=== Top 10 with reason factors ===")
print(top_out[["image","risk_score","wet","warm","low_residue","tillage_eff"]])

print("\n=== Bottom 10 with reason factors ===")
print(bot_out[["image","risk_score","wet","warm","low_residue","tillage_eff"]])