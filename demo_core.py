
import ast
import re

import numpy as np
import pandas as pd

import train_models as tm

DEFAULTS = {
    "RandomForest": dict(n_estimators=200, max_depth=3, min_samples_leaf=2, max_features=0.5),
    "XGBoost": dict(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.85,
                    colsample_bytree=0.8, min_child_weight=2, reg_lambda=1),
}
MODELS = [("RandomForest", tm.make_rf), ("XGBoost", tm.make_xgb)]
LABEL = {"dram": "DRAM", "nand": "NAND"}


def load_params(horizon=1):
    """Đọc cấu hình đã tinh chỉnh từ results/best_params.csv (hoặc results_h<N>/ nếu horizon > 1)."""
    folder = tm.base_dir / ("results" if horizon == 1 else f"results_h{horizon}")
    path = folder / "best_params.csv"
    params = {}
    if path.exists():
        for key, text in pd.read_csv(path, index_col=0)["params"].items():
            try:
                params[key] = ast.literal_eval(re.sub(r"np\.\w+\(([^)]*)\)", r"\1", text))
            except (ValueError, SyntaxError):
                pass
    return params


def dc_label(name):
    return LABEL[name]


def _fit(make_model, params, lab, feats):
    return tm.make_pipe(make_model(**params)).fit(lab[feats], lab["target"])


def horizon_to(target_month):
    """Số tháng từ tháng cuối cùng có dữ liệu giá đến tháng muốn dự đoán (ví dụ '2026-12')."""
    last = min(tm.load_price(f, f"{n}_price").index.max() for n, f in tm.TARGETS.items())
    h = (pd.Period(target_month, "M") - pd.Period(last, "M")).n
    if h < 1:
        raise ValueError(f"Tháng {target_month} không nằm sau dữ liệu (dữ liệu đến {last:%m/%Y}).")
    return h


def get_results(horizon=1):
    """Trả về dict theo từng loại chip: dữ liệu, dự báo sau `horizon` tháng của Naive / RF / XGBoost."""
    tm.HORIZON = horizon                       # build_frame đọc biến này khi tạo nhãn
    market = tm.load_market()
    saved = load_params(horizon)
    out = {}
    for name, fname in tm.TARGETS.items():
        price = tm.load_price(fname, f"{name}_price")
        df, feats = tm.build_frame(price, market)
        last = df.iloc[[-1]]                       # tháng mới nhất (chưa biết giá tháng sau)
        lab = df.dropna(subset=["target"])
        p0 = float(last["price"].iloc[0])
        preds = {"Naive": p0}
        for mname, mk in MODELS:
            pipe = _fit(mk, saved.get(f"{name}/{mname}", DEFAULTS[mname]), lab, feats)
            preds[mname] = p0 * float(np.exp(pipe.predict(last[feats])[0]))
        out[name] = {"df": df, "feats": feats, "last": last, "price_now": p0, "preds": preds,
                     "month_now": last.index[0], "horizon": horizon,
                     "next_month": last.index[0] + pd.DateOffset(months=horizon)}
    return out


def backtest_table(res, name, n_last=20):
    """Mỗi tháng t: huấn luyện chỉ bằng các mẫu mà nhãn đã biết tại t, dự đoán giá tháng t+1."""
    tm.HORIZON = res.get("horizon", 1)        # đảm bảo đúng horizon cả khi get_results lấy từ cache
    saved = load_params(res.get("horizon", 1))
    df, feats = res["df"], res["feats"]
    lab = df.dropna(subset=["target"])
    rows = []
    for t in lab.index[-n_last:]:
        known = lab[(lab.index + pd.DateOffset(months=tm.HORIZON)) <= t]
        if len(known) < 20:
            continue
        p0 = float(lab.loc[t, "price"])
        row = {"month": t, "next_month": t + pd.DateOffset(months=tm.HORIZON), "price_now": p0,
               "actual": p0 * float(np.exp(lab.loc[t, "target"])), "Naive": p0}
        for mname, mk in MODELS:
            pipe = _fit(mk, saved.get(f"{name}/{mname}", DEFAULTS[mname]), known, feats)
            row[mname] = p0 * float(np.exp(pipe.predict(lab.loc[[t], feats])[0]))
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    target_month = sys.argv[1] if len(sys.argv) > 1 else "2026-12"    # python demo_core.py 2026-12
    h = horizon_to(target_month)
    results = get_results(h)
    print(f"\nDự đoán giá tháng {target_month} (cách tháng cuối có dữ liệu {h} tháng)")
    for nm, r in results.items():
        print(f"\n{dc_label(nm)}: giá tháng {r['month_now']:%m/%Y} = {r['price_now']:.4f}  ->  dự đoán {r['next_month']:%m/%Y}")
        for m, v in r["preds"].items():
            print(f"  {m:<13} {v:.4f}  ({(v / r['price_now'] - 1) * 100:+.1f}%)")
   
