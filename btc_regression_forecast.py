# btc_regression_forecast.py
# Proje 3 – Basit Regresyon ile Bitcoin'de 1 Gün İleri Fiyat Tahmini

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_btc(days=365):
    """CoinGecko’dan son `days` günlük BTC/USD verisini çek ve günlük ortalama fiyatı döndür."""
    cg = CoinGeckoAPI()
    data = cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='usd', days=days)
    df = pd.DataFrame(data['prices'], columns=['ts_ms', 'price'])
    df['ts'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True)
    df = df.set_index('ts').drop(columns=['ts_ms'])
    daily = df['price'].resample('D').mean().rename('close')
    return daily.to_frame()

def rsi(series, window=14):
    """14-günlük RSI hesaplama."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)

def macd(series, fast=12, slow=26, signal=9):
    """MACD, signal ve histogram değerleri."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def make_features(df):
    """Özellik setini hazırla ve yarınki fiyatı hedef olarak ekle."""
    out = df.copy()
    out['close_lag1']  = out['close'].shift(1)
    out['ma7']         = out['close'].rolling(7).mean()
    out['ma30']        = out['close'].rolling(30).mean()
    out['rsi14']       = rsi(out['close'])
    macd_line, sig, hist = macd(out['close'])
    out['macd']        = macd_line
    out['macd_signal'] = sig
    out['macd_hist']   = hist
    out['target']      = out['close'].shift(-1)
    return out.dropna()

def evaluate(true, pred, name):
    """Gerçek vs. tahmin için RMSE, MAE, MAPE hesapla ve yazdır."""
    mse  = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")

def main():
    # 1) Veri çek ve özellik seti oluştur
    df   = get_btc(days=365)
    data = make_features(df)
    feature_cols = [c for c in data.columns if c != 'target']
    X, y = data[feature_cols], data['target']

    # 2) Train/Test split (son 60 gün test)
    split = 60
    X_train, X_test = X.iloc[:-split], X.iloc[-split:]
    y_train, y_test = y.iloc[:-split], y.iloc[-split:]

    # 3a) Naif model (yarın = dünün fiyatı)
    naive_pred = X_test['close_lag1'].values
    evaluate(y_test.values, naive_pred, "Naif (lag1)")

    # 3b) Linear Regression
    pipe_lr = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
    pipe_lr.fit(X_train, y_train)
    pred_lr = pipe_lr.predict(X_test)
    evaluate(y_test.values, pred_lr, "LinearRegression")

    # 3c) Ridge Regression
    pipe_rg = Pipeline([('scaler', StandardScaler()), ('rg', Ridge(alpha=5))])
    pipe_rg.fit(X_train, y_train)
    pred_rg = pipe_rg.predict(X_test)
    evaluate(y_test.values, pred_rg, "Ridge(alpha=5)")

    # 4) Grafikler ve dosyaya kaydetme
    # a) Gerçek vs. Tahmin (Ridge)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.index, y_test.values, label='Gerçek (Close)', linewidth=2)
    ax.plot(y_test.index, pred_rg,         label='Tahmin (Ridge)', linewidth=2)
    ax.set_title('BTC – 1 Gün İleri Fiyat Tahmini (Test)')
    ax.set_xlabel('Tarih'); ax.set_ylabel('USD')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('btc_forecast_test.png', dpi=150)
    plt.show()

    # b) Tahmin Hatası (Residual) – Ridge
    fig, ax = plt.subplots(figsize=(12, 3.5))
    residuals = y_test.values - pred_rg
    ax.plot(y_test.index, residuals, linewidth=1.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title('Tahmin Hatası (Residual) – Ridge')
    ax.set_xlabel('Tarih'); ax.set_ylabel('USD')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('btc_forecast_residuals.png', dpi=150)
    plt.show()

    # c) Özellik Önemleri (Linear Regression)
    coefs = pipe_lr.named_steps['lr'].coef_
    imp   = pd.Series(np.abs(coefs), index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 4))
    imp.head(12).plot(kind='bar', ax=ax)
    ax.set_title('Özellik Önemleri (|β|) – LinearRegression')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('btc_feature_importance.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
from pycoingecko import CoinGeckoAPI
import pandas as pd

cg = CoinGeckoAPI()
raw = cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='usd', days=365)
df = pd.DataFrame(raw['prices'], columns=['ts_ms','price'])
df['ts'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True)
df = df.set_index('ts').resample('D').mean().drop(columns=['ts_ms'])
print(df.loc['2025-07-15'])

