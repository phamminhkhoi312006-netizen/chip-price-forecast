"""Demo Streamlit: dự báo giá chip nhớ DRAM & NAND.

Chạy cục bộ:
    streamlit run streamlit_app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

import demo_core as dc
import train_models as tm

st.set_page_config(
    page_title="Dự báo giá chip DRAM & NAND",
    page_icon="💾",
    layout="wide",
)

MODEL_NAME = {
    "Naive": "Naive (giữ giá)",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
}
SERIES_COLOR = {
    "Giá thực": "#111827",
    "Naive (giữ giá)": "#9ca3af",
    "Random Forest": "#2563eb",
    "XGBoost": "#dc2626",
}
CHIP_ICON = {"dram": "🧠", "nand": "💽"}
MAX_HORIZON = 12


# ----------------------------- DỮ LIỆU / CACHE -----------------------------
@st.cache_data
def data_range():
    """(tháng đầu, tháng cuối) của chuỗi giá từng loại chip."""
    spans = {}
    for name, fname in tm.TARGETS.items():
        s = tm.load_price(fname, f"{name}_price")
        spans[name] = (s.index.min(), s.index.max())
    return spans


@st.cache_data(show_spinner="Đang huấn luyện mô hình…")
def get_results(horizon: int, stamp: int):
    return dc.get_results(horizon)


@st.cache_data
def load_metrics(horizon: int, stamp: int):
    path = results_dir(horizon) / "metrics.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner="Đang chạy backtest…")
def run_backtest(horizon: int, n_last: int, stamp: int):
    res = get_results(horizon, stamp)
    return {name: dc.backtest_table(r, name, n_last=n_last) for name, r in res.items()}


def results_dir(horizon: int):
    return tm.base_dir / ("results" if horizon == 1 else f"results_h{horizon}")


def results_stamp(horizon: int) -> int:
    """mtime mới nhất trong results*/ — đổi file kết quả là cache tự làm mới."""
    folder = results_dir(horizon)
    if not folder.exists():
        return 0
    return max((p.stat().st_mtime_ns for p in folder.glob("*")), default=0)


def fmt(value: float) -> str:
    """Định dạng giá gọn cho cả DRAM (~4.85) và NAND (~0.079)."""
    return f"{value:,.4g}"


def pct(value: float, base: float) -> str:
    return f"{(value / base - 1) * 100:+.1f}%"


# ----------------------------- BIỂU ĐỒ -----------------------------
def color_scale():
    return alt.Scale(domain=list(SERIES_COLOR), range=list(SERIES_COLOR.values()))


def forecast_chart(res, months: int = 36):
    """Giá lịch sử + điểm dự báo của 3 mô hình."""
    df = res["df"]
    hist = df["price"].tail(months)
    frames = [
        pd.DataFrame({"date": hist.index, "value": hist.values, "series": "Giá thực"})
    ]
    last_date, last_price = df.index[-1], float(df["price"].iloc[-1])
    for model, value in res["preds"].items():
        label = MODEL_NAME[model]
        frames.append(pd.DataFrame({
            "date": [last_date, res["next_month"]],
            "value": [last_price, value],
            "series": [label, label],
        }))
    data = pd.concat(frames, ignore_index=True)

    base = alt.Chart(data).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("value:Q", title="Giá", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=color_scale(), title=None,
                        sort=["Giá thực", "Naive (giữ giá)", "Random Forest", "XGBoost"]),
    )
    line = base.mark_line(strokeWidth=2.2).encode(
        tooltip=[
            alt.Tooltip("date:T", format="%m/%Y", title="Tháng"),
            alt.Tooltip("series:N", title="Chuỗi"),
            alt.Tooltip("value:Q", format=".4f", title="Giá"),
        ]
    )
    points = base.mark_point(size=70, filled=True).transform_filter(
        alt.datum.series != "Giá thực"
    )
    now_rule = alt.Chart(pd.DataFrame({"date": [last_date]})).mark_rule(
        color="#6b7280", strokeDash=[4, 4]
    ).encode(x="date:T")
    return (line + points + now_rule).properties(height=330)


def backtest_chart(bt: pd.DataFrame):
    long = bt.melt(
        id_vars=["month"],
        value_vars=["actual", "Naive", "RandomForest", "XGBoost"],
        var_name="series",
        value_name="value",
    )
    long["series"] = long["series"].map(lambda s: "Giá thực" if s == "actual" else MODEL_NAME[s])
    base = alt.Chart(long).encode(
        x=alt.X("month:T", title=None),
        y=alt.Y("value:Q", title="Giá", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=color_scale(), title=None,
                        sort=["Giá thực", "Naive (giữ giá)", "Random Forest", "XGBoost"]),
    )
    line = base.mark_line(strokeWidth=2.2, point=alt.OverlayMarkDef(size=35)).encode(
        tooltip=[
            alt.Tooltip("month:T", format="%m/%Y", title="Tháng"),
            alt.Tooltip("series:N", title="Chuỗi"),
            alt.Tooltip("value:Q", format=".4f", title="Giá"),
        ]
    )
    return line.properties(height=300)


# ----------------------------- SIDEBAR -----------------------------
spans = data_range()
last_month = min(end for _, end in spans.values())

with st.sidebar:
    st.header("⚙️ Thiết lập")
    periods = [(last_month, i) for i in range(1, MAX_HORIZON + 1)]
    options = {
        f"{pd.Period(last_month, 'M') + h:%m/%Y}  ·  sau {h} tháng": h
        for last_month, h in periods
    }
    choice = st.selectbox("Tháng cần dự báo", list(options), index=0)
    horizon = options[choice]

    if not (results_dir(horizon) / "best_params.csv").exists():
        st.info(
            f"Chưa có bộ tham số tinh chỉnh cho h={horizon}. "
            "Mô hình sẽ dùng tham số mặc định (chất lượng có thể thấp hơn).",
            icon="ℹ️",
        )

    st.divider()
    st.markdown("**Mô hình so sánh**")
    st.markdown("- Naive: giá tháng sau = giá tháng này\n"
                "- Random Forest (scikit-learn)\n"
                "- XGBoost")
    st.divider()
    st.caption("Dữ liệu giá chip: `data/dram_price.csv`, `data/nand_price.csv`\n\n"
               "Biến thị trường: MU, SK Hynix, Samsung, WDC, SOX, USD/KRW, FEDFUNDS")

# ----------------------------- HEADER -----------------------------
st.title("💾 Dự báo giá chip nhớ DRAM & NAND")
st.caption(
    f"Dữ liệu giá đến **{last_month:%m/%Y}** · Dự báo **log-return** bằng Random Forest / XGBoost "
    f"· Mốc dự báo: **{pd.Period(last_month, 'M') + horizon:%m/%Y}** (h = {horizon} tháng)"
)
st.warning(
    "Kết quả chỉ mang tính minh hoạ học thuật. Ở tập test, các mô hình học máy chưa vượt trội "
    "so với naive (MASE ≈ 1) — xem tab **Hiệu năng mô hình**.",
    icon="⚠️",
)

try:
    results = get_results(horizon, results_stamp(horizon))
except Exception as exc:  # noqa: BLE001 - hiển thị lỗi thân thiện trên UI
    st.error(f"Không chạy được mô hình: {exc}")
    st.stop()

tab_forecast, tab_metrics, tab_backtest, tab_about = st.tabs(
    ["📈 Dự báo", "🎯 Hiệu năng mô hình", "🧪 Backtest", "ℹ️ Giới thiệu"]
)

# ----------------------------- TAB 1: DỰ BÁO -----------------------------
with tab_forecast:
    st.subheader("Tổng quan")
    overview = []
    for name, res in results.items():
        row = {
            "Loại chip": dc.dc_label(name),
            "Giá hiện tại": res["price_now"],
            **{MODEL_NAME[m]: v for m, v in res["preds"].items()},
        }
        overview.append(row)
    overview = pd.DataFrame(overview)
    st.dataframe(
        overview,
        hide_index=True,
        column_config={
            "Giá hiện tại": st.column_config.NumberColumn(format="%.4f"),
            **{MODEL_NAME[m]: st.column_config.NumberColumn(format="%.4f") for m in MODEL_NAME},
        },
    )
    st.caption(f"Đơn vị giá theo dữ liệu gốc · Tháng dự báo: {results['dram']['next_month']:%m/%Y}")

    st.divider()

    for name, res in results.items():
        label = dc.dc_label(name)
        with st.container():
            st.subheader(f"{CHIP_ICON[name]} {label} — giá tháng {res['month_now']:%m/%Y}")

            prev_price = float(res["df"]["price"].iloc[-2])
            cols = st.columns(4)
            cols[0].metric(
                "Giá hiện tại",
                fmt(res["price_now"]),
                f"{pct(res['price_now'], prev_price)} so với tháng trước",
                delta_color="off",
                border=True,
            )
            for col, (model, value) in zip(cols[1:], res["preds"].items()):
                col.metric(
                    MODEL_NAME[model],
                    fmt(value),
                    pct(value, res["price_now"]),
                    delta_color="off",
                    border=True,
                )

            st.altair_chart(forecast_chart(res), width="stretch")

            with st.expander("Bảng dự báo chi tiết"):
                detail = pd.DataFrame([
                    {"Mô hình": MODEL_NAME[m], "Giá dự báo": v,
                     "Thay đổi (%)": (v / res["price_now"] - 1) * 100}
                    for m, v in res["preds"].items()
                ])
                st.dataframe(
                    detail,
                    hide_index=True,
                    column_config={
                        "Giá dự báo": st.column_config.NumberColumn(format="%.4f"),
                        "Thay đổi (%)": st.column_config.NumberColumn(format="%+.2f%%"),
                    },
                )

# ----------------------------- TAB 2: HIỆU NĂNG -----------------------------
with tab_metrics:
    metrics = load_metrics(horizon, results_stamp(horizon))
    if metrics is None:
        st.info(
            f"Chưa có `metrics.csv` cho h={horizon}. "
            f"Chạy `HORIZON={horizon} python train_models.py` để tạo kết quả đánh giá.",
            icon="📄",
        )
    else:
        st.subheader("Chỉ số trên tập Test")
        test = metrics[metrics["split"] == "test"].copy()
        test["target"] = test["target"].map(dc.LABEL)
        test["DirAcc"] = test["DirAcc"] * 100
        test = test.drop(columns=["split"])

        show = test.rename(columns={
            "target": "Loại chip", "model": "Mô hình", "MAPE%": "MAPE (%)",
            "sMAPE%": "sMAPE (%)", "DirAcc": "Đúng hướng (%)",
        })
        st.dataframe(
            show.style.format({
                "MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}",
                "sMAPE (%)": "{:.2f}", "MASE": "{:.3f}",
                "Đúng hướng (%)": "{:.0f}", "R2": "{:.3f}",
            }, na_rep="—").apply(
                lambda s: ["background-color: #dcfce7" if v < 1 else ""
                           for v in s], subset=["MASE"],
            ),
            hide_index=True,
        )
        st.caption("MASE < 1 (nền xanh) nghĩa là tốt hơn naive · DirAcc để trống với naive "
                   "vì naive không dự đoán hướng.")

        best = show.loc[show["MASE"].idxmin()]
        if best["MASE"] < 1:
            st.success(f"Tốt nhất trên test: **{best['Mô hình']}** cho **{best['Loại chip']}** "
                       f"(MASE = {best['MASE']:.3f}).")
        else:
            st.warning("Trên tập test, không mô hình nào tốt hơn naive (MASE ≥ 1). "
                       "Nên xem kết quả như minh hoạ phương pháp, không phải dự báo đầu tư.")

        with st.expander("Chỉ số trên tập Validation"):
            val = metrics[metrics["split"] == "val"].copy()
            val["target"] = val["target"].map(dc.LABEL)
            val["DirAcc"] = val["DirAcc"] * 100
            val = val.drop(columns=["split"]).rename(columns={
                "target": "Loại chip", "model": "Mô hình", "MAPE%": "MAPE (%)",
                "sMAPE%": "sMAPE (%)", "DirAcc": "Đúng hướng (%)",
            })
            st.dataframe(
                val.style.format({
                    "MAE": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}",
                    "sMAPE (%)": "{:.2f}", "MASE": "{:.3f}",
                    "Đúng hướng (%)": "{:.0f}", "R2": "{:.3f}",
                }, na_rep="—"),
                hide_index=True,
            )

# ----------------------------- TAB 3: BACKTEST -----------------------------
with tab_backtest:
    st.subheader("Backtest cuốn chiếu (walk-forward)")
    st.caption(
        "Tại mỗi tháng, mô hình chỉ được huấn luyện bằng các mẫu mà nhãn đã biết tại thời điểm đó, "
        "rồi dự đoán giá tháng kế tiếp. Đây là kiểm tra trung thực hơn so với tập test cố định."
    )
    n_last = st.slider("Số tháng backtest gần nhất", min_value=6, max_value=36, value=18, step=1)

    if st.button("Chạy backtest", type="primary"):
        st.session_state["backtest_done"] = True
    if st.session_state.get("backtest_done"):
        backtests = run_backtest(horizon, n_last, results_stamp(horizon))
        for name, res in results.items():
            bt = backtests.get(name)
            st.markdown(f"#### {CHIP_ICON[name]} {dc.dc_label(name)}")
            if bt is None or bt.empty:
                st.info("Không đủ dữ liệu để backtest.")
                continue

            errors = []
            for model in ["Naive", "RandomForest", "XGBoost"]:
                err = (bt[model] - bt["actual"]).abs()
                errors.append({
                    "Mô hình": MODEL_NAME[model],
                    "MAE": err.mean(),
                    "MAPE (%)": (err / bt["actual"]).mean() * 100,
                })
            st.dataframe(
                pd.DataFrame(errors),
                hide_index=True,
                column_config={
                    "MAE": st.column_config.NumberColumn(format="%.4f"),
                    "MAPE (%)": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.altair_chart(backtest_chart(bt), width="stretch")
            with st.expander("Bảng số liệu backtest"):
                table = bt.copy()
                table["month"] = table["month"].dt.strftime("%m/%Y")
                table["next_month"] = table["next_month"].dt.strftime("%m/%Y")
                table = table.rename(columns={
                    "month": "Tháng", "next_month": "Dự báo cho", "price_now": "Giá hiện tại",
                    "actual": "Giá thực tế", **{m: MODEL_NAME[m] for m in MODEL_NAME},
                })
                st.dataframe(table, hide_index=True,
                             column_config={c: st.column_config.NumberColumn(format="%.4f")
                                            for c in table.columns if c not in ("Tháng", "Dự báo cho")})
    else:
        st.info("Bấm **Chạy backtest** để bắt đầu (mất khoảng vài chục giây vì phải huấn luyện lại "
                "nhiều lần).", icon="⏱️")

# ----------------------------- TAB 4: GIỚI THIỆU -----------------------------
with tab_about:
    st.subheader("Pipeline")
    st.markdown(
        """
1. **Dữ liệu**: giá DRAM/NAND theo tháng (2018 → nay) và biến thị trường hằng ngày
   (MU, SK Hynix, Samsung, WDC, SOX, USD/KRW, FEDFUNDS). Dữ liệu ngày được gộp về tháng,
   chỉ `ffill` từ quá khứ.
2. **Đặc trưng**: log-return giá tháng hiện tại, 2 lag, độ lệch so với MA3, độ biến động 3 tháng,
   log-return tháng của các cổ phiếu/ chỉ số thị trường và lãi suất Fed.
3. **Nhãn**: log-return của giá sau *h* tháng (`log(P[t+h]) - log(P[t])`).
4. **Mô hình**: Random Forest và XGBoost trong Pipeline có `SimpleImputer` (fit riêng từng fold
   để chống rò rỉ dữ liệu). Siêu tham số đã tinh chỉnh bằng `RandomizedSearchCV` +
   `TimeSeriesSplit`, lưu tại `results*/best_params.csv`.
5. **Dự báo**: fit lại trên toàn bộ dữ liệu có nhãn, dự đoán log-return rồi quy đổi
   `giá dự báo = giá hiện tại × exp(log-return)`.
        """
    )

    st.subheader("Cách chạy cục bộ")
    st.code(
        "pip install -r requirements.txt\n"
        "streamlit run streamlit_app.py",
        language="bash",
    )

    st.subheader("Hạn chế")
    st.markdown(
        """
- Chuỗi giá chỉ ~100 tháng → kết quả nhạy với cách chia train/val/test.
- Giá chip chịu ảnh hưởng của chu kỳ cung cầu, tin tức, chính sách — mô hình chỉ dùng
  đặc trưng lịch sử nên không nắm được cú sốc.
- MASE trên tập test xấp xỉ 1: mô hình chưa chắc tốt hơn "giá không đổi".
- Đây là demo học thuật, **không phải khuyến nghị đầu tư**.
        """
    )

st.divider()
st.caption("Demo Streamlit · Nguồn dữ liệu: `data/` · Mô hình: scikit-learn + XGBoost")
