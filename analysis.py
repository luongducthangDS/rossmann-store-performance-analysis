"""
Retail Store & Pricing Performance
Dataset: Rossmann Store Sales (Kaggle) - 1,017,210 daily records, 1,115 stores, 2013-01 to 2015-07.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, json

DATA = "data"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv", parse_dates=["Date"], dtype={"StateHoliday": str})
store = pd.read_csv(f"{DATA}/store.csv")

df = train.merge(store, on="Store", how="left")
# closed days have 0 sales/customers by definition - not meaningful for performance comparison
df = df[df["Open"] == 1].copy()
df["basket_value"] = df["Sales"] / df["Customers"].replace(0, np.nan)

print(f"Rows (open days): {len(df):,}  Stores: {df['Store'].nunique()}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ---------- 1. Promo effectiveness, controlled for day-of-week ----------
promo_dow = df.groupby(["DayOfWeek", "Promo"])["Sales"].mean().unstack("Promo")
promo_dow.columns = ["no_promo", "promo"]
promo_dow["uplift_pct"] = promo_dow["promo"] / promo_dow["no_promo"] - 1
promo_dow.to_csv(f"{OUT}/promo_effect_by_dayofweek.csv")
print("\n=== Promo effect on Sales, by day of week ===")
print(promo_dow.round(2))

overall_uplift = df.groupby("Promo")["Sales"].mean()
overall_pct = overall_uplift[1] / overall_uplift[0] - 1
print(f"\nOverall promo uplift: {overall_pct:+.1%} (promo avg {overall_uplift[1]:.0f} vs non-promo {overall_uplift[0]:.0f})")

customers_uplift = df.groupby("Promo")["Customers"].mean()
customers_pct = customers_uplift[1] / customers_uplift[0] - 1
basket_uplift = df.groupby("Promo")["basket_value"].mean()
basket_pct = basket_uplift[1] / basket_uplift[0] - 1
print(f"Promo drives more FOOTFALL ({customers_pct:+.1%} customers) vs more BASKET SIZE ({basket_pct:+.1%} sales/customer)")

# ---------- 2. Store type / assortment performance ----------
type_perf = df.groupby("StoreType").agg(
    avg_daily_sales=("Sales", "mean"),
    avg_customers=("Customers", "mean"),
    avg_basket_value=("basket_value", "mean"),
    n_stores=("Store", "nunique"),
).round(2)
type_perf.to_csv(f"{OUT}/store_type_performance.csv")
print("\n=== Store type performance ===")
print(type_perf)

assortment_perf = df.groupby("Assortment").agg(
    avg_daily_sales=("Sales", "mean"),
    avg_basket_value=("basket_value", "mean"),
    n_stores=("Store", "nunique"),
).round(2)
assortment_perf.to_csv(f"{OUT}/assortment_performance.csv")
print("\n=== Assortment performance ===")
print(assortment_perf)

# ---------- 3. Competition distance impact ----------
store_level = df.groupby("Store").agg(
    avg_sales=("Sales", "mean"),
    avg_basket_value=("basket_value", "mean"),
    competition_distance=("CompetitionDistance", "first"),
).dropna()
store_level["comp_distance_bucket"] = pd.qcut(store_level["competition_distance"], 4, labels=["Very close", "Close", "Far", "Very far"])
comp_perf = store_level.groupby("comp_distance_bucket", observed=True).agg(
    avg_sales=("avg_sales", "mean"),
    avg_basket_value=("avg_basket_value", "mean"),
    n_stores=("avg_sales", "count"),
).round(2)
comp_perf.to_csv(f"{OUT}/competition_distance_performance.csv")
print("\n=== Performance by nearest-competitor distance (quartile buckets) ===")
print(comp_perf)
corr = store_level[["avg_sales", "competition_distance"]].corr().iloc[0, 1]
print(f"Correlation (avg store sales vs competition distance): {corr:.3f}")

# ---------- 4. Underperforming stores: high footfall, low basket value ----------
store_level["basket_rank_pct"] = store_level["avg_basket_value"].rank(pct=True)
store_level["sales_rank_pct"] = store_level["avg_sales"].rank(pct=True)
# candidates: sales in bottom 25% of their StoreType despite average/above-average footfall
store_meta = df.groupby("Store").agg(avg_customers=("Customers", "mean"), StoreType=("StoreType", "first")).reset_index()
store_level = store_level.reset_index().merge(store_meta, on="Store")
store_level["customer_rank_pct"] = store_level.groupby("StoreType")["avg_customers"].rank(pct=True)
store_level["sales_rank_pct_within_type"] = store_level.groupby("StoreType")["avg_sales"].rank(pct=True)
underperformers = store_level[(store_level["customer_rank_pct"] >= 0.5) & (store_level["sales_rank_pct_within_type"] <= 0.25)]
underperformers = underperformers.sort_values("sales_rank_pct_within_type")
underperformers.to_csv(f"{OUT}/underperforming_stores.csv", index=False)
print(f"\n{len(underperformers)} stores flagged: average-or-above footfall for their type, but bottom-25% sales within type (likely pricing/basket issue, not traffic issue)")

# chart: promo uplift by day of week
fig, ax = plt.subplots(figsize=(9, 5))
promo_dow["uplift_pct"].plot(kind="bar", ax=ax, color="#2E86AB")
ax.set_ylabel("Sales uplift from promo")
ax.set_title("Promo sales uplift by day of week")
ax.axhline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(f"{OUT}/promo_uplift_by_dow.png", dpi=150)
plt.close()

# chart: sales by store type
fig, ax = plt.subplots(figsize=(7, 5))
type_perf["avg_daily_sales"].plot(kind="bar", ax=ax, color="#A23B72")
ax.set_ylabel("Avg daily sales ($)")
ax.set_title("Average daily sales by store type")
plt.tight_layout()
plt.savefig(f"{OUT}/sales_by_store_type.png", dpi=150)
plt.close()

stats = {
    "rows_open_days": len(df),
    "stores": int(df["Store"].nunique()),
    "date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
    "promo_uplift_sales_pct": round(float(overall_pct), 4),
    "promo_uplift_customers_pct": round(float(customers_pct), 4),
    "promo_uplift_basket_value_pct": round(float(basket_pct), 4),
    "corr_sales_vs_competition_distance": round(float(corr), 4),
    "underperforming_stores_flagged": int(len(underperformers)),
}
with open(f"{OUT}/headline_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print("\n=== Headline stats ===")
print(json.dumps(stats, indent=2))
