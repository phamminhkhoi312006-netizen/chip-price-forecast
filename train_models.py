
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ----------------------------- CẤU HÌNH -----------------------------
SEED = 42
HORIZON = int(os.environ.get("HORIZON", 1))   # dự đoán giá sau HORIZON tháng (đổi bằng biến môi trường HORIZON)
LAGS = 2             # số lag của log-return giá
TRAIN_FRAC, VAL_FRAC = 0.60, 0.20   # phần còn lại là Test
MARKET_COLS = ["MU", "SKHynix", "Samsung", "SOX", "WDC", "USDKRW"]
TARGETS = {"dram": "dram_price.csv", "nand": "nand_price.csv"}

base_dir = Path(__file__).resolve().parent
out_dir = base_dir / ("results" if HORIZON == 1 else f"results_h{HORIZON}")
out_dir.mkdir(exist_ok=True)


def find_csv(name):
    for d in (base_dir / "data", base_dir):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"Không tìm thấy {name} trong data/ hoặc cùng thư mục với script")


# ----------------------------- BƯỚC 1: ĐỌC + KHẢO SÁT -----------------------------
def load_market():
    m = pd.read_csv(find_csv("market.csv"), parse_dates=["date"], index_col="date").sort_index()
    # Gộp ngày -> tháng (lấy giá trị hợp lệ cuối tháng). KHÔNG bfill; chỉ ffill từ quá khứ.
    last = m.resample("MS").last()
    feats = pd.DataFrame(index=last.index)
    for c in MARKET_COLS:
        feats[f"{c}_ret"] = np.log(last[c].ffill()).diff()
    feats["FEDFUNDS"] = last["FEDFUNDS"].ffill()
    return feats


def load_price(fname, col):
    p = pd.read_csv(find_csv(fname), parse_dates=["date"], index_col="date").sort_index()
    s = p[col].astype(float)
    s.index = s.index.to_period("M").to_timestamp()  # chuẩn hóa về đầu tháng
    return s


def audit(prices):
    print("=" * 70)
    print("BƯỚC 1 - KHẢO SÁT DỮ LIỆU")
    for k, s in prices.items():
        r = np.log(s).diff().dropna()
        z = (r - r.mean()) / r.std() if r.std() > 0 else r * 0
        print(f"  {k}: {len(s)} tháng ({s.index.min():%m/%Y} -> {s.index.max():%m/%Y}), "
              f"giá {s.min():.3f} -> {s.max():.3f}, "
              f"số tháng biến động bất thường (|z|>2): {(z.abs() > 2).sum()}")
    if len(prices) == 2:
        a, b = prices["dram"], prices["nand"]
        common = a.index.intersection(b.index)
        same = np.isclose(a[common], b[common], rtol=1e-3).mean()
        print(f"  Tỷ lệ tháng DRAM ≈ NAND: {same:.0%}  (nếu quá cao, hãy kiểm tra lại nguồn NAND)")
    print("=" * 70)


# ----------------------------- BƯỚC 2-3: TARGET + ĐẶC TRƯNG -----------------------------
def build_frame(price, market_feats):
    df = pd.DataFrame({"price": price})
    df["ret1"] = np.log(df["price"]).diff()
    for k in range(1, LAGS + 1):
        df[f"ret_lag{k}"] = df["ret1"].shift(k)
    df["ma3_gap"] = df["price"] / df["price"].rolling(3).mean() - 1
    df["vol3"] = df["ret1"].rolling(3).std()
    price_feats = ["ret1"] + [f"ret_lag{k}" for k in range(1, LAGS + 1)] + ["ma3_gap", "vol3"]

    df = df.join(market_feats, how="left")
    feats = price_feats + list(market_feats.columns)

    # target: log-return của giá sau HORIZON tháng (chỉ dùng thông tin tương lai ở nhãn)
    df["target"] = np.log(df["price"].shift(-HORIZON) / df["price"])
    df = df.dropna(subset=price_feats)          # bỏ các tháng khởi động (chưa đủ lag), KHÔNG điền giả
    return df, feats


# ----------------------------- METRICS (slide 15-16) -----------------------------
def evaluate(price_now, ret_true, ret_pred):
    y = price_now * np.exp(ret_true)
    p = price_now * np.exp(ret_pred)
    naive = price_now                       # naive: giá không đổi
    mae = mean_absolute_error(y, p)
    naive_mae = mean_absolute_error(y, naive)
    chg_true, chg_pred = np.sign(y - price_now), np.sign(p - price_now)
    mask = chg_true != 0                    # bỏ tháng giá đứng yên khi tính hướng
    return {
        "MAE": mae,
        "RMSE": np.sqrt(mean_squared_error(y, p)),
        "MAPE%": 100 * np.mean(np.abs((y - p) / y)),
        "sMAPE%": 100 * np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p))),
        "MASE": mae / naive_mae if naive_mae > 0 else np.nan,
        "DirAcc": float(np.mean(chg_true[mask] == chg_pred[mask]))
        if mask.any() and np.any(chg_pred != 0) else np.nan,   # naive không dự đoán hướng -> NaN
        "R2": r2_score(y, p) if len(y) > 1 else np.nan,
    }


# ----------------------------- MÔ HÌNH -----------------------------
def make_pipe(model):
    # Imputer nằm trong Pipeline => chỉ fit trên dữ liệu train của từng fold (chống rò rỉ)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def make_rf(**kw):
    return RandomForestRegressor(random_state=SEED, n_jobs=1, **kw)


def make_xgb(**kw):
    return XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=1,
                        random_state=SEED, verbosity=0, **kw)


RF_GRID = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [2, 3, 4, None],
    "model__min_samples_leaf": [1, 2, 3],
    "model__max_features": [0.3, 0.5, 1.0],
}
XGB_GRID = {
    "model__max_depth": [2, 3, 4],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.5, 0.8, 1.0],
    "model__min_child_weight": [1, 2, 3],
    "model__reg_lambda": [1, 5, 10],
}


def tune(pipe, grid, X, y, n_splits):
    search = RandomizedSearchCV(
        pipe, grid, n_iter=15, cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=1, refit=False,
    )
    search.fit(X, y)
    return {k.replace("model__", ""): v for k, v in search.best_params_.items()}


def fit_predict(make_model, params, X_fit, y_fit, X_pred):
    pipe = make_pipe(make_model(**params)).fit(X_fit, y_fit)
    return pipe, pipe.predict(X_pred)


# ----------------------------- CHẠY CHO TỪNG BIẾN MỤC TIÊU -----------------------------
def run_target(name, price, market_feats):
    df, feats = build_frame(price, market_feats)
    last_row = df.iloc[[-1]]                       # dòng mới nhất (chưa có nhãn) -> dự báo tương lai
    lab = df.dropna(subset=["target"])
    n = len(lab)
    n_tr, n_va = int(round(n * TRAIN_FRAC)), int(round(n * VAL_FRAC))
    tr, va, te = lab.iloc[:n_tr], lab.iloc[n_tr:n_tr + n_va], lab.iloc[n_tr + n_va:]
    if min(len(tr), len(va), len(te)) < 2:
        raise ValueError(f"[{name}] Quá ít dữ liệu để chia Train/Val/Test: {len(tr)}/{len(va)}/{len(te)}")
    trva = pd.concat([tr, va])
    n_splits = 3 if len(tr) < 40 else 5

    print(f"\n[{name.upper()}] mẫu có nhãn = {n} | Train {len(tr)} ({tr.index.min():%m/%Y}-{tr.index.max():%m/%Y})"
          f" | Val {len(va)} | Test {len(te)} ({te.index.min():%m/%Y}-{te.index.max():%m/%Y})")
    if n < 40:
        print(f"  !! CẢNH BÁO: chỉ có {n} mẫu. Kết quả rất không ổn định; cần chuỗi giá dài hơn để kết luận chắc chắn.")

    Xtr, ytr = tr[feats], tr["target"]
    Xva, yva = va[feats], va["target"]
    Xte = te[feats]
    rows, preds_test, params_used = [], {}, {}

    # --- Naive baseline: giá không đổi (log-return = 0) ---
    zero = lambda d: np.zeros(len(d))
    for split, d in [("val", va), ("test", te)]:
        rows.append({"target": name, "model": "Naive", "split": split,
                     **evaluate(d["price"].values, d["target"].values, zero(d))})
    preds_test["Naive"] = zero(te)

    # --- Random Forest ---
    rf_params = tune(make_pipe(make_rf()), RF_GRID, Xtr, ytr, n_splits)
    _, rf_val = fit_predict(make_rf, rf_params, Xtr, ytr, Xva)
    rf_final, rf_test = fit_predict(make_rf, rf_params, trva[feats], trva["target"], Xte)
    params_used["RandomForest"] = rf_params

    # --- XGBoost: tune cấu trúc cây, rồi early stopping trên Validation để chọn số cây ---
    xgb_params = tune(make_pipe(make_xgb(n_estimators=300)), XGB_GRID, Xtr, ytr, n_splits)
    imp = SimpleImputer(strategy="median").fit(Xtr)
    es = make_xgb(**{**xgb_params, "n_estimators": 1000}, early_stopping_rounds=20, eval_metric="mae")
    es.fit(imp.transform(Xtr), ytr, eval_set=[(imp.transform(Xva), yva)], verbose=False)
    xgb_params["n_estimators"] = int(es.best_iteration) + 1
    _, xgb_val = fit_predict(make_xgb, xgb_params, Xtr, ytr, Xva)
    xgb_final, xgb_test = fit_predict(make_xgb, xgb_params, trva[feats], trva["target"], Xte)
    params_used["XGBoost"] = xgb_params

    for mname, pv, pt in [("RandomForest", rf_val, rf_test), ("XGBoost", xgb_val, xgb_test)]:
        rows.append({"target": name, "model": mname, "split": "val",
                     **evaluate(va["price"].values, yva.values, pv)})
        rows.append({"target": name, "model": mname, "split": "test",
                     **evaluate(te["price"].values, te["target"].values, pt)})
        preds_test[mname] = pt

    # --- Dự báo tương lai: khóa cấu hình, fit trên toàn bộ dữ liệu có nhãn ---
    forecast = {}
    for mname, mk in [("RandomForest", make_rf), ("XGBoost", make_xgb)]:
        full = make_pipe(mk(**params_used[mname])).fit(lab[feats], lab["target"])
        r = float(full.predict(last_row[feats])[0])
        target_month = last_row.index[0] + pd.DateOffset(months=HORIZON)
        forecast[mname] = {"target": name, "model": mname, "forecast_month": f"{target_month:%Y-%m}",
                           "last_price": float(last_row["price"].iloc[0]),
                           "pred_logret": r, "pred_price": float(last_row["price"].iloc[0] * np.exp(r))}

    # --- Feature importance (mô hình cuối) ---
    imps = pd.DataFrame({
        "feature": feats,
        "RandomForest": rf_final.named_steps["model"].feature_importances_,
        "XGBoost": xgb_final.named_steps["model"].feature_importances_,
    }).sort_values("RandomForest", ascending=False)
    imps.insert(0, "target", name)

    # --- Biểu đồ ---
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(df.index, df["price"], "k-o", ms=3, label="Giá thực")
    when = te.index + pd.DateOffset(months=HORIZON)
    for mname, style in [("Naive", "g--"), ("RandomForest", "b-s"), ("XGBoost", "r-^")]:
        ax.plot(when, te["price"].values * np.exp(preds_test[mname]), style, ms=5, label=mname)
    ax.set_title(f"{name.upper()}: dự đoán trên tập Test (horizon = {HORIZON} tháng)")
    ax.set_ylabel("Giá"); ax.legend(); ax.grid(alpha=.3)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(out_dir / f"{name}_test.png", dpi=150); plt.close(fig)

    return rows, list(forecast.values()), imps, params_used


def main():
    market_feats = load_market()
    prices = {k: load_price(f, f"{k}_price") for k, f in TARGETS.items()}
    audit(prices)

    all_rows, all_fc, all_imp, all_params = [], [], [], {}
    for name, price in prices.items():
        rows, fc, imps, params = run_target(name, price, market_feats)
        all_rows += rows; all_fc += fc; all_imp.append(imps); all_params[name] = params

    res = pd.DataFrame(all_rows)
    res.to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(all_fc).to_csv(out_dir / "forecast.csv", index=False)
    pd.concat(all_imp).to_csv(out_dir / "feature_importance.csv", index=False)
    pd.Series({f"{t}/{m}": str(p) for t, d in all_params.items() for m, p in d.items()}) \
        .to_csv(out_dir / "best_params.csv", header=["params"])

    pd.set_option("display.width", 160)
    print("\n" + "=" * 70 + "\nKẾT QUẢ (MASE < 1 nghĩa là tốt hơn naive)\n" + "=" * 70)
    print(res.round(3).to_string(index=False))
    print("\nDỰ BÁO THÁNG TIẾP THEO\n" + pd.DataFrame(all_fc).round(3).to_string(index=False))
    print(f"\nĐã lưu kết quả vào: {out_dir}")


if __name__ == "__main__":
    main()
