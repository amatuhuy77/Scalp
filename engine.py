"""
engine.py - Mesin analisis & backtest XAUUSD Scalper Pro (v3)

Tanpa dependensi Streamlit/yfinance supaya bisa diuji terpisah.

Prinsip utama
-------------
1. Hanya candle yang SUDAH TUTUP yang dipakai (tidak ada lookahead).
2. Hierarki timeframe, bukan voting:
      M15 = arah (boleh/tidak boleh BUY atau SELL)
      M5  = setup (pullback ke EMA 50 + konfirmasi momentum)
      M1  = pemicu (timing entry)
   TF yang lebih tinggi SELALU menang. M1 Buy saat M5 Sell = diabaikan.
3. Backtest memakai aturan yang sama persis dengan sinyal live.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd

TF_MIN = {"1m": 1, "5m": 5, "15m": 15}
WARMUP = 210  # candle awal yang EMA 200-nya belum matang


# ----------------------------------------------------------------------------
# Parameter
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    # Sinyal
    min_score: int = 70          # skor minimum setup (0-100)
    adx_min: float = 18.0        # filter pasar sideways
    pullback_atr: float = 0.8    # toleransi jarak ke EMA 50 (x ATR)
    # Trade
    sl_atr: float = 1.5          # jarak SL (x ATR setup)
    rr: float = 1.5              # rasio TP : SL
    spread: float = 0.30         # biaya spread+slippage ($ per oz)
    max_hold: int = 60           # maks candle ditahan
    # Filter waktu & disiplin
    use_session: bool = True
    sess_start_utc: int = 7      # London open
    sess_end_utc: int = 20       # NY sore
    max_losses_day: int = 2      # stop trading hari itu setelah N loss
    # Modal
    capital: float = 50.0
    risk_pct: float = 1.0
    contract: float = 100.0      # oz per 1.0 lot (Standard 100, Cent ~1)
    min_lot: float = 0.01
    lot_step: float = 0.01
    small_lot_mode: str = "min"  # "min" = pakai lot minimum, "skip" = lewati
    # Perlindungan berita / lonjakan
    slip_sl: float = 0.15        # slippage tambahan ($/oz) saat SL terisi
    use_spike: bool = True       # tahan sinyal setelah lonjakan volatilitas
    spike_mult: float = 3.0      # candle dianggap lonjakan jika range >= N x ATR
    spike_hold_min: int = 15     # lama sinyal ditahan setelah lonjakan (menit)


WEIGHTS = {"trend": 30, "adx": 15, "macd": 20, "rsi": 15, "pullback": 20}


# ----------------------------------------------------------------------------
# Indikator (metode Wilder, sama dengan MT4/MT5/TradingView)
# ----------------------------------------------------------------------------
def _rma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]

    delta = c.diff()
    up, dn = delta.clip(lower=0), (-delta).clip(lower=0)
    rs = _rma(up, 14) / (_rma(dn, 14) + 1e-12)
    d["RSI"] = 100 - 100 / (1 + rs)

    for n in (20, 50, 200):
        d[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()

    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["MACD"] = macd
    d["MACD_SIG"] = macd.ewm(span=9, adjust=False).mean()
    d["MACD_H"] = d["MACD"] - d["MACD_SIG"]

    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    d["ATR"] = _rma(tr, 14)

    um, dm = h.diff(), -l.diff()
    plus = pd.Series(np.where((um > dm) & (um > 0), um, 0.0), index=d.index)
    minus = pd.Series(np.where((dm > um) & (dm > 0), dm, 0.0), index=d.index)
    pdi = 100 * _rma(plus, 14) / (d["ATR"] + 1e-12)
    mdi = 100 * _rma(minus, 14) / (d["ATR"] + 1e-12)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
    d["ADX"] = _rma(dx, 14)
    return d


def trend_bias(d: pd.DataFrame) -> pd.Series:
    """+1 bullish, -1 bearish, 0 netral (selaras EMA 50/200 dan harga)."""
    up = (d["Close"] > d["EMA50"]) & (d["EMA50"] > d["EMA200"])
    dn = (d["Close"] < d["EMA50"]) & (d["EMA50"] < d["EMA200"])
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=d.index)


# ----------------------------------------------------------------------------
# Setup (dinilai pada TF setup, mis. M5)
# ----------------------------------------------------------------------------
def setup_components(d: pd.DataFrame, side: int, p: Params) -> pd.DataFrame:
    """Komponen boolean untuk satu arah (side=+1 buy, -1 sell)."""
    s = side
    comp = pd.DataFrame(index=d.index)
    comp["trend"] = (s * (d["Close"] - d["EMA50"]) > 0) & (s * (d["EMA50"] - d["EMA200"]) > 0)
    comp["adx"] = d["ADX"] >= p.adx_min
    comp["macd"] = (s * d["MACD_H"] > 0) & (s * d["MACD_H"].diff() >= 0)
    if s == 1:
        comp["rsi"] = (d["RSI"] > 45) & (d["RSI"] < 68)
        extreme = d["Low"].rolling(3, min_periods=1).min()
        dist = extreme - d["EMA50"]
    else:
        comp["rsi"] = (d["RSI"] < 55) & (d["RSI"] > 32)
        extreme = d["High"].rolling(3, min_periods=1).max()
        dist = d["EMA50"] - extreme
    # harga sempat turun/naik mendekati EMA 50 dalam 3 candle terakhir
    comp["pullback"] = dist <= p.pullback_atr * d["ATR"]
    return comp


def setup_score(comp: pd.DataFrame) -> pd.Series:
    return sum(comp[k].astype(int) * w for k, w in WEIGHTS.items())


def setup_flags(d: pd.DataFrame, side: int, p: Params) -> pd.Series:
    """Setup valid = trend & pullback wajib, skor >= minimum, indikator matang."""
    comp = setup_components(d, side, p)
    ok = comp["trend"] & comp["pullback"] & (setup_score(comp) >= p.min_score)
    valid = pd.Series(np.arange(len(d)) >= WARMUP, index=d.index) & d["ATR"].notna()
    return ok & valid


def trigger(d: pd.DataFrame, side: int) -> pd.Series:
    """Candle pemicu: warna searah dan close di sisi EMA 20."""
    return (side * (d["Close"] - d["Open"]) > 0) & (side * (d["Close"] - d["EMA20"]) > 0)


# ----------------------------------------------------------------------------
# Penyelarasan antar timeframe TANPA lookahead
# ----------------------------------------------------------------------------
def _ns(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.values.astype("datetime64[ns]").astype("int64")


def align(values, src_idx, src_tf: str, dst_idx, dst_tf: str, fill=np.nan) -> np.ndarray:
    """
    Ambil nilai candle src TERAKHIR YANG SUDAH TUTUP pada saat candle dst tutup.
    Index yfinance = waktu BUKA candle, jadi waktu tutup = index + durasi TF.
    """
    src_avail = _ns(src_idx) + TF_MIN[src_tf] * 60 * 10**9
    dst_close = _ns(dst_idx) + TF_MIN[dst_tf] * 60 * 10**9
    pos = np.searchsorted(src_avail, dst_close, side="right") - 1
    out = np.full(len(dst_idx), fill, dtype=float)
    ok = pos >= 0
    out[ok] = np.asarray(values, dtype=float)[pos[ok]]
    return out


def session_mask(idx: pd.DatetimeIndex, p: Params) -> np.ndarray:
    if not p.use_session:
        return np.ones(len(idx), dtype=bool)
    h = idx.tz_convert("UTC").hour
    return np.asarray((h >= p.sess_start_utc) & (h < p.sess_end_utc))


def spike_raw(d: pd.DataFrame, p: Params) -> pd.Series:
    """True pada candle yang range-nya >= N x ATR candle sebelumnya (kausal)."""
    rng = d["High"] - d["Low"]
    return (rng >= p.spike_mult * d["ATR"].shift(1)).fillna(False)


def spike_mask(d: pd.DataFrame, tf: str, p: Params) -> np.ndarray:
    """Candle yang sinyalnya harus ditahan (lonjakan + masa tunggu)."""
    if not p.use_spike:
        return np.zeros(len(d), dtype=bool)
    hold = max(1, int(np.ceil(p.spike_hold_min / TF_MIN[tf])))
    sp = spike_raw(d, p).astype(float)
    return (sp.rolling(hold, min_periods=1).max() > 0).values


# ----------------------------------------------------------------------------
# Pembangun sinyal (dipakai backtest)
# ----------------------------------------------------------------------------
def build_signals(frames: dict, entry_tf: str, setup_tf: str, gate_tf: Optional[str],
                  p: Params):
    """
    frames: {"1m": df, "5m": df, "15m": df} sudah ber-indikator, index UTC.
    gate_tf=None -> mode naif (tanpa filter TF lebih tinggi).
    Return: long_sig, short_sig, atr (semua array sepanjang entry df)
    """
    e = frames[entry_tf]
    s = frames[setup_tf]
    n = len(e)

    long_ok = align(setup_flags(s, 1, p).values, s.index, setup_tf, e.index, entry_tf, 0) > 0.5
    short_ok = align(setup_flags(s, -1, p).values, s.index, setup_tf, e.index, entry_tf, 0) > 0.5
    atr = align(s["ATR"].values, s.index, setup_tf, e.index, entry_tf)

    if gate_tf is not None:
        g = frames[gate_tf]
        b = align(trend_bias(g).values, g.index, gate_tf, e.index, entry_tf, 0)
        long_ok &= b > 0.5
        short_ok &= b < -0.5

    long_sig = long_ok & trigger(e, 1).values
    short_sig = short_ok & trigger(e, -1).values
    sess = session_mask(e.index, p)
    long_sig &= sess
    short_sig &= sess
    blk = spike_mask(e, entry_tf, p)
    long_sig &= ~blk
    short_sig &= ~blk
    # Hanya candle yang punya data ATR valid
    valid = ~np.isnan(atr)
    return long_sig & valid, short_sig & valid, atr


# ----------------------------------------------------------------------------
# Backtester
# ----------------------------------------------------------------------------
def _calc_lot(equity: float, sl_dist: float, p: Params):
    risk_usd = equity * p.risk_pct / 100.0
    raw = risk_usd / (sl_dist * p.contract)
    lot = np.floor(raw / p.lot_step + 1e-9) * p.lot_step
    if lot < p.min_lot:
        if p.small_lot_mode == "skip":
            return None
        lot = p.min_lot
    return round(float(lot), 4)


def backtest(e: pd.DataFrame, long_sig, short_sig, atr, p: Params):
    """
    Entry di OPEN candle berikutnya setelah sinyal. SL/TP dicek pakai High/Low.
    Jika SL & TP tersentuh di candle yang sama -> dianggap SL (konservatif).
    """
    o = e["Open"].values.astype(float)
    h = e["High"].values.astype(float)
    l = e["Low"].values.astype(float)
    c = e["Close"].values.astype(float)
    days = np.asarray(e.index.tz_convert("UTC").normalize().values.astype("datetime64[D]").astype("int64"))
    n = len(e)

    equity = p.capital
    pos, entry, sl, tp, lot, sl_dist, entry_i = 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    eff_risk = 0.0
    day_losses: dict = {}
    trades = []
    ruined = False

    for i in range(1, n):
        if pos == 0:
            sig = 1 if long_sig[i - 1] else (-1 if short_sig[i - 1] else 0)
            if sig and day_losses.get(days[i], 0) < p.max_losses_day and atr[i - 1] > 0:
                sl_dist = p.sl_atr * atr[i - 1]
                lt = _calc_lot(equity, sl_dist, p)
                if lt is not None:
                    pos, lot, entry_i = sig, lt, i
                    entry = o[i]
                    sl = entry - pos * sl_dist
                    tp = entry + pos * sl_dist * p.rr
                    eff_risk = lot * p.contract * sl_dist / equity * 100
        if pos != 0:
            if pos == 1:
                hit_sl, hit_tp = l[i] <= sl, h[i] >= tp
            else:
                hit_sl, hit_tp = h[i] >= sl, l[i] <= tp
            ex, why = None, ""
            if hit_sl:
                ex, why = sl - pos * p.slip_sl, "SL"
                # gap melewati SL -> keluar di open
                if i > entry_i and ((pos == 1 and o[i] < sl) or (pos == -1 and o[i] > sl)):
                    ex = o[i]
            elif hit_tp:
                ex, why = tp, "TP"
            elif i - entry_i >= p.max_hold:
                ex, why = c[i], "WAKTU"
            if ex is not None:
                net = pos * (ex - entry) - p.spread
                pnl = net * lot * p.contract
                equity += pnl
                if pnl < 0:
                    day_losses[days[i]] = day_losses.get(days[i], 0) + 1
                trades.append({
                    "entry_time": e.index[entry_i], "exit_time": e.index[i],
                    "side": "BUY" if pos == 1 else "SELL",
                    "entry": entry, "exit": ex, "sl": sl, "tp": tp, "lot": lot,
                    "reason": why, "sl_dist": sl_dist, "R": net / sl_dist, "pnl": pnl,
                    "equity": equity, "risk_pct": eff_risk,
                })
                pos = 0
                if equity <= 0:
                    ruined = True
                    break
    return pd.DataFrame(trades), ruined


def stats(tr: pd.DataFrame, capital: float, ruined: bool = False) -> dict:
    if tr is None or tr.empty:
        return {"trades": 0}
    w = tr[tr["pnl"] > 0]
    ls = tr[tr["pnl"] <= 0]
    gp, gl = w["pnl"].sum(), -ls["pnl"].sum()
    eq = np.concatenate([[capital], tr["equity"].values])
    peak = np.maximum.accumulate(eq)
    dd = ((peak - eq) / np.where(peak > 0, peak, 1)).max() * 100
    streak = best = 0
    for v in (tr["pnl"] <= 0).values:
        streak = streak + 1 if v else 0
        best = max(best, streak)
    return {
        "trades": int(len(tr)),
        "winrate": len(w) / len(tr) * 100,
        "pf": (gp / gl) if gl > 0 else float("inf"),
        "exp_R": float(tr["R"].mean()),
        "avg_win_R": float(w["R"].mean()) if len(w) else 0.0,
        "avg_loss_R": float(ls["R"].mean()) if len(ls) else 0.0,
        "net_pct": (tr["equity"].iloc[-1] / capital - 1) * 100,
        "final_equity": float(tr["equity"].iloc[-1]),
        "max_dd": float(dd),
        "max_losing_streak": int(best),
        "avg_risk_pct": float(tr["risk_pct"].mean()),
        "ruined": ruined,
    }


def stats_R_only(tr: pd.DataFrame) -> dict:
    """Statistik tanpa ukuran lot (untuk membandingkan periode in/out-of-sample)."""
    if tr is None or tr.empty:
        return {"trades": 0}
    gp = tr.loc[tr["R"] > 0, "R"].sum()
    gl = -tr.loc[tr["R"] <= 0, "R"].sum()
    return {"trades": int(len(tr)), "winrate": float((tr["R"] > 0).mean() * 100),
            "pf": float(gp / gl) if gl > 0 else float("inf"), "exp_R": float(tr["R"].mean())}


def run_full(frames: dict, entry_tf: str, p: Params, hierarchical: bool = True):
    """
    hierarchical=True : M15 gate -> M5 setup -> entry TF trigger
    hierarchical=False: hanya TF entry sendiri (kondisi 'bingung' M1 vs M5)
    """
    if hierarchical:
        setup_tf = "5m"
        gate_tf = "15m"
    else:
        setup_tf, gate_tf = entry_tf, None
    ls, ss, atr = build_signals(frames, entry_tf, setup_tf, gate_tf, p)
    tr, ruined = backtest(frames[entry_tf], ls, ss, atr, p)
    return tr, ruined


def split_in_out(tr: pd.DataFrame, e_index: pd.DatetimeIndex, frac: float = 0.7):
    """Bagi trade menjadi periode awal (in-sample) dan akhir (out-of-sample)."""
    if tr.empty:
        return tr, tr, None
    cut = e_index[int(len(e_index) * frac)]
    return tr[tr["entry_time"] < cut], tr[tr["entry_time"] >= cut], cut


def sensitivity(frames: dict, entry_tf: str, p: Params,
                scores=(60, 70, 85), rrs=(1.0, 1.5, 2.0)) -> pd.DataFrame:
    """Peta ketahanan: kalau hanya 1 sel yang untung, strateginya rapuh."""
    rows = []
    for sc in scores:
        for rr in rrs:
            tr, _ = run_full(frames, entry_tf, replace(p, min_score=sc, rr=rr), True)
            st = stats_R_only(tr)
            rows.append({"Skor min": sc, "RR": rr, "Trade": st["trades"],
                         "Win %": st.get("winrate", np.nan),
                         "PF": st.get("pf", np.nan), "Expectancy (R)": st.get("exp_R", np.nan)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Snapshot LIVE: keputusan hierarki satu arah yang konsisten
# ----------------------------------------------------------------------------
def closed_only(d: pd.DataFrame, tf: str, now: pd.Timestamp) -> pd.DataFrame:
    end = d.index + pd.Timedelta(minutes=TF_MIN[tf])
    return d[end <= now]


def lot_info(max_risk_usd: float, sl_dist: float, p: Params) -> dict:
    raw = max_risk_usd / (sl_dist * p.contract) if sl_dist > 0 else 0.0
    floor_lot = np.floor(raw / p.lot_step + 1e-9) * p.lot_step
    lot = max(p.min_lot, floor_lot)
    real_risk = lot * p.contract * sl_dist
    return {"lot": round(float(lot), 4), "raw": raw, "real_risk": real_risk,
            "too_small": floor_lot < p.min_lot}


def live_snapshot(frames: dict, p: Params, now: pd.Timestamp, max_risk_usd: float) -> dict:
    c1 = closed_only(frames["1m"], "1m", now)
    c5 = closed_only(frames["5m"], "5m", now)
    c15 = closed_only(frames["15m"], "15m", now)
    for name, x in (("M1", c1), ("M5", c5), ("M15", c15)):
        if len(x) < WARMUP:
            return {"error": f"Data {name} belum cukup ({len(x)} candle)."}

    b15 = int(trend_bias(c15).iloc[-1])
    b5 = int(trend_bias(c5).iloc[-1])
    r5, r1 = c5.iloc[-1], c1.iloc[-1]
    m1_dir = 1 if (r1["Close"] > r1["EMA20"] and r1["MACD_H"] > 0) else (
        -1 if (r1["Close"] < r1["EMA20"] and r1["MACD_H"] < 0) else 0)

    side = b15
    comps, score = {}, 0
    if side != 0:
        cp = setup_components(c5, side, p).iloc[-1]
        comps = {k: bool(cp[k]) for k in WEIGHTS}
        score = sum(w for k, w in WEIGHTS.items() if comps[k])
    trig = bool(side != 0 and trigger(c1, side).iloc[-1])

    sess_ok = bool(session_mask(c1.index[-1:], p)[0])
    atr5 = float(r5["ATR"])
    dist50 = float(abs(r5["Close"] - r5["EMA50"]))

    # --- Keputusan: berurutan, TF tinggi menang ---
    if b15 == 0:
        status, tone = "TIDAK TRADE", "wait"
        why = "M15 belum punya arah jelas (harga di antara/menembus EMA). Tunggu tren terbentuk."
        broken = "m15"
    elif b5 != b15:
        status, tone = "TUNGGU M5", "wait"
        why = (f"M15 {'naik' if b15 > 0 else 'turun'}, tetapi M5 belum searah. "
               "Ini biasanya koreksi - jangan lawan M5.")
        broken = "m5"
    elif not (comps["trend"] and comps["pullback"]):
        status, tone = "TUNGGU PULLBACK", "wait"
        why = (f"Arah sudah searah, tetapi harga masih jauh dari EMA 50 M5 "
               f"(${dist50:.2f} > batas ${p.pullback_atr * atr5:.2f}). Jangan kejar harga.")
        broken = "m5"
    elif score < p.min_score or not comps["adx"]:
        status, tone = "TUNGGU M5", "wait"
        why = (f"Skor setup {score}% < minimum {p.min_score}%"
               + ("" if comps["adx"] else " dan pasar sideways (ADX rendah)") + ".")
        broken = "m5"
    elif not sess_ok:
        status, tone = "DI LUAR JAM", "wait"
        why = (f"Di luar sesi ramai (UTC {p.sess_start_utc}:00-{p.sess_end_utc}:00). "
               "Spread melebar dan sinyal kurang andal.")
        broken = "sesi"
    elif not trig:
        counter = (m1_dir == -side)
        status, tone = "SIAP, TUNGGU M1", "ready"
        why = ("M1 sedang melawan arah (koreksi) - JANGAN ikut M1. Tunggu candle M1 kembali searah."
               if counter else "Setup M5 valid. Tunggu satu candle M1 searah yang menutup di atas/bawah EMA 20.")
        broken = "m1"
    else:
        status, tone = ("ENTRY BUY" if side > 0 else "ENTRY SELL"), ("buy" if side > 0 else "sell")
        why = "M15, M5, dan M1 searah. Semua syarat terpenuhi."
        broken = None

    plan = None
    if side != 0 and broken in (None, "m1"):
        price = float(r1["Close"])
        sl_dist = p.sl_atr * atr5
        plan = {
            "side": "BUY" if side > 0 else "SELL",
            "entry": price,
            "sl": price - side * sl_dist,
            "tp": price + side * sl_dist * p.rr,
            "sl_dist": sl_dist,
            **lot_info(max_risk_usd, sl_dist, p),
        }

    return {
        "status": status, "tone": tone, "why": why, "broken": broken,
        "b15": b15, "b5": b5, "m1": m1_dir, "side": side,
        "score": score, "comps": comps, "plan": plan,
        "atr5": atr5, "dist50": dist50, "ema50_5": float(r5["EMA50"]),
        "rsi5": float(r5["RSI"]), "adx5": float(r5["ADX"]),
        "last_close": float(frames["1m"]["Close"].iloc[-1]),
        "last_bar_end": c1.index[-1] + pd.Timedelta(minutes=1),
        "sess_ok": sess_ok,
    }


# ----------------------------------------------------------------------------
# Penjaga berita (live)
# ----------------------------------------------------------------------------
def news_guard(now: pd.Timestamp, events: list, m1: pd.DataFrame, p: Params) -> dict:
    """
    events: [{"name", "time" (Timestamp UTC), "before" (menit), "after" (menit)}]
    Prioritas: jendela berita terjadwal, lalu lonjakan volatilitas M1.
    """
    out = {"blocked": False, "kind": None, "reason": "", "until": None, "next": None}
    upcoming = []
    for ev in events:
        t0 = ev["time"] - pd.Timedelta(minutes=int(ev["before"]))
        t1 = ev["time"] + pd.Timedelta(minutes=int(ev["after"]))
        if t0 <= now <= t1:
            left = (ev["time"] - now).total_seconds() / 60
            when = (f"rilis {left:.0f} menit lagi" if left > 0
                    else f"sudah rilis {abs(left):.0f} menit lalu")
            out.update(blocked=True, kind="news", until=max(t1, out["until"] or t1),
                       reason=f"{ev['name']} {when}")
        elif ev["time"] > now:
            upcoming.append(ev)
    if upcoming:
        nx = min(upcoming, key=lambda x: x["time"])
        out["next"] = {"name": nx["name"], "minutes": (nx["time"] - now).total_seconds() / 60}

    if p.use_spike and len(m1) > 30:
        recent = m1.tail(60)
        sp = spike_raw(m1, p).tail(60)
        if sp.any():
            last = recent.index[np.where(sp.values)[0][-1]]
            until = last + pd.Timedelta(minutes=1 + p.spike_hold_min)
            if now < until and not out["blocked"]:
                mult = float((recent.loc[last, "High"] - recent.loc[last, "Low"])
                             / m1["ATR"].shift(1).loc[last])
                out.update(blocked=True, kind="spike", until=until,
                           reason=f"Candle M1 sebesar {mult:.1f}× ATR (lonjakan volatilitas)")
    return out


# ----------------------------------------------------------------------------
# Monte Carlo
# ----------------------------------------------------------------------------
def monte_carlo(tr: pd.DataFrame, p: Params, n_trades: int = 100, n_sims: int = 1000,
                seed: int = 7) -> Optional[dict]:
    """
    Bootstrap: trade (R dan jarak SL) diambil acak dengan pengembalian, lot dihitung
    ulang dari ekuitas saat itu dengan aturan lot minimum yang sama dengan backtest.
    """
    if tr is None or len(tr) < 10:
        return None
    rng = np.random.default_rng(seed)
    R, SL = tr["R"].values.astype(float), tr["sl_dist"].values.astype(float)
    pick = rng.integers(0, len(R), size=(n_sims, n_trades))
    eq = np.full(n_sims, float(p.capital))
    paths = np.empty((n_trades + 1, n_sims))
    paths[0] = eq
    peak, maxdd = eq.copy(), np.zeros(n_sims)
    for t in range(n_trades):
        r, s = R[pick[:, t]], SL[pick[:, t]]
        alive = eq > 0
        raw = eq * p.risk_pct / 100.0 / (s * p.contract)
        lot = np.floor(raw / p.lot_step + 1e-9) * p.lot_step
        fallback = p.min_lot if p.small_lot_mode == "min" else 0.0
        lot = np.where(lot < p.min_lot, fallback, lot)
        eq = np.where(alive, np.maximum(eq + r * s * p.contract * lot, 0.0), 0.0)
        paths[t + 1] = eq
        peak = np.maximum(peak, eq)
        maxdd = np.maximum(maxdd, np.where(peak > 0, (peak - eq) / peak, 0.0))
    lo = paths.min(axis=0)
    final = paths[-1]
    bands = np.percentile(paths, [5, 50, 95], axis=1)
    return {
        "n_trades": n_trades, "n_sims": n_sims, "bands": bands,
        "p_loss30": float((lo <= 0.7 * p.capital).mean() * 100),
        "p_near_mc": float((lo <= 0.2 * p.capital).mean() * 100),
        "p_profit": float((final > p.capital).mean() * 100),
        "median_final": float(np.median(final)),
        "p5_final": float(np.percentile(final, 5)),
        "p95_final": float(np.percentile(final, 95)),
        "dd95": float(np.percentile(maxdd, 95) * 100),
    }


# ----------------------------------------------------------------------------
# Jurnal trade
# ----------------------------------------------------------------------------
JOURNAL_COLS = ["waktu", "arah", "entry", "sl", "exit", "lot", "catatan"]


def journal_prepare(j: pd.DataFrame) -> pd.DataFrame:
    d = j.copy()
    for c in ("entry", "sl", "exit"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["arah"] = d["arah"].astype(str).str.upper().str.strip()
    d = d[d["arah"].isin(["BUY", "SELL"])].dropna(subset=["entry", "sl", "exit"])
    risk = (d["entry"] - d["sl"]).abs()
    d = d[risk > 0].copy()
    sgn = np.where(d["arah"] == "BUY", 1.0, -1.0)
    d["R"] = sgn * (d["exit"] - d["entry"]) / (d["entry"] - d["sl"]).abs()
    return d


def journal_stats(j: pd.DataFrame) -> dict:
    d = journal_prepare(j)
    if d.empty:
        return {"trades": 0}
    w, ls = d[d["R"] > 0], d[d["R"] <= 0]
    return {"trades": int(len(d)), "winrate": float((d["R"] > 0).mean() * 100),
            "exp_R": float(d["R"].mean()),
            "avg_loss_R": float(ls["R"].mean()) if len(ls) else 0.0,
            "avg_win_R": float(w["R"].mean()) if len(w) else 0.0}
