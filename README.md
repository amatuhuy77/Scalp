# XAUUSD Scalper Pro v3

Aplikasi Streamlit untuk sinyal scalping emas (XAUUSD) dengan satu keputusan dari tiga timeframe, plus backtest bawaan.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run xauusd_advanced_app.py
```

Untuk Streamlit Community Cloud: push semua file (termasuk folder `.streamlit/`) lalu pilih `xauusd_advanced_app.py` sebagai file utama.

## Masalah yang diselesaikan: M1 Buy, M5 Sell

Versi lama menilai tiap timeframe sendiri-sendiri lalu merata-ratakannya, sehingga M1 dan M5 bisa saling bertentangan. Versi ini memakai hierarki:

| Lapisan | Peran | Aturan |
|---|---|---|
| M15 | Arah | Hanya BUY jika harga > EMA50 > EMA200; hanya SELL jika sebaliknya |
| M5 | Setup | Searah M15, pullback ke EMA50, ADX cukup, MACD dan RSI mendukung, skor >= minimum |
| M1 | Pemicu | Satu candle searah yang menutup di sisi EMA20 |

Timeframe yang lebih tinggi selalu menang. M1 yang melawan M5 diabaikan.

## Perbaikan akurasi

- Hanya candle yang sudah tutup yang dipakai (tidak ada lookahead; diuji otomatis).
- RSI, ATR, ADX memakai metode Wilder (sama dengan MT4/MT5/TradingView).
- Tren dan pullback wajib, bukan sekadar poin skor.
- Filter ADX (hindari sideways) dan filter sesi London-NY.
- Peringatan data tertunda saat pasar tutup.
- Lot minimum broker diperhitungkan: aplikasi menampilkan risiko nyata, bukan angka ideal.
- Berhenti trading setelah N loss dalam sehari.

## Backtest

Tab **Backtest** menjalankan aturan yang sama persis dengan sinyal live:

- Entry di open candle berikutnya; SL dianggap kena lebih dulu jika SL dan TP tersentuh dalam satu candle.
- Spread/slippage dihitung (default $0.30 per oz).
- Perbandingan **Naif (1 timeframe)** vs **Hierarki** untuk melihat dampak konflik M1/M5.
- Pemisahan periode awal (70%) dan akhir (30%) untuk mendeteksi overfitting.
- Uji ketahanan parameter 3x3.

Batas data Yahoo Finance: M1 hanya 7 hari, M5/M15 sekitar 60 hari.

## Fitur v3.1

- **Penjaga berita**: jadwal berita manual (tab Berita) menahan sinyal sebelum dan sesudah rilis; candle M1 >= 3x ATR juga menahan sinyal 15 menit.
- **Slippage SL** di backtest (default $0.15/oz) supaya hasil tidak terlalu optimis.
- **Monte Carlo**: 1.000 simulasi dari trade backtest, menampilkan peluang modal turun 30% dan mendekati margin call.
- **Jurnal trade**: catat trade demo/nyata, bandingkan expectancy dan rata-rata rugi dengan backtest.
- Jadwal berita dan jurnal tidak tersimpan permanen di Streamlit gratis; gunakan tombol unduh/muat CSV.

## Batasan

- Sumber data `GC=F` (futures) bisa berbeda beberapa dolar dari XAUUSD spot broker.
- Backtest bukan jaminan hasil. Uji di akun demo sebelum memakai uang asli.
- Trading emas berleverage dapat menghabiskan seluruh modal.

## Struktur

- `xauusd_advanced_app.py` - antarmuka Streamlit
- `engine.py` - indikator, sinyal, backtester (tanpa dependensi Streamlit)
- `.streamlit/config.toml` - tema
