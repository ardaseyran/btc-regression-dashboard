# btc_regression_extended.py
# Proje 3’ü geliştiriyoruz: ek modeller + ek göstergeler + TimeSeriesSplit CV + GridSearchCV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycoingecko import CoinGeckoAPI

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# --- 1) Veri Çekme & Temel Özellikler ---
def get_btc(days=365):
    cg = CoinGeckoAPI()
    raw = cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='usd', days=days)
    df = pd.DataFrame(raw['prices'], columns=['ts_ms','price'])
    df['ts'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True)
    df = df.set_index('ts').drop(columns=['ts_ms'])
    daily = df['price'].resample('D').mean().rename('close')
    return daily.to_frame()

# RSI, MACD fonksiyonları (önceden gördüğümüz gibi)
def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100/(1+rs))).fillna(50)

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    hist      = macd_line - sig_line
    return macd_line, sig_line, hist

# --- 2) Ek Göstergeler ve Feature Mühendisliği ---
def make_features(df):
    out = df.copy()
    # Lag ve ortalamalar
    out['close_lag1'] = out['close'].shift(1)
    out['ma7']        = out['close'].rolling(7).mean()
    out['ma30']       = out['close'].rolling(30).mean()
    # RSI
    out['rsi14']      = rsi(out['close'])
    # MACD
    macd_line, sig, hist = macd(out['close'])
    out['macd']        = macd_line
    out['macd_signal'] = sig
    out['macd_hist']   = hist
    # 20-günlük volatilite ve Bollinger bant genişliği
    out['std20']       = out['close'].rolling(20).std()
    out['boll_w']      = out['std20'] * 4
    # Hedef: ertesi gün kapanış
    out['target']      = out['close'].shift(-1)
    return out.dropna()

# --- 3) Değerlendirme Metriği ---
def evaluate(true, pred, name):
    mse  = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    print(f"{name:12s} → RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%")
    return rmse, mae, mape

# --- 4) Ana Akış ---
def main():
    # 4.1 Veri ve özellik hazırlığı
    df   = get_btc(365)
    data = make_features(df)
    features = [c for c in data.columns if c!='target']
    X, y = data[features], data['target']

    # 4.2 Train/Test split (son 60 gün test)
    split = 60
    X_train, X_test = X.iloc[:-split], X.iloc[-split:]
    y_train, y_test = y.iloc[:-split], y.iloc[-split:]

    # 4.3 Modeller ve parametre ızgaraları
    tscv = TimeSeriesSplit(n_splits=5)
    grid_defs = {
        'Ridge':      (Ridge(max_iter=5000),         {'alpha': [0.1, 1, 5, 10]}),
        'Lasso':      (Lasso(max_iter=5000),         {'alpha': [0.001, 0.01, 0.1, 1]}),
        'ElasticNet': (ElasticNet(max_iter=5000),    {'alpha': [0.1, 1], 'l1_ratio': [0.2, 0.5, 0.8]}),
        'RF':         (RandomForestRegressor(),      {'n_estimators': [50,100], 'max_depth': [5,10,None]}),
        'XGB':        (XGBRegressor(objective='reg:squarederror', use_label_encoder=False),
                       {'n_estimators': [50,100], 'max_depth': [3,5], 'learning_rate': [0.01,0.1]})
    }

    best_models = {}
    print("\n=== HYPERPARAMETER TUNING (TimeSeriesSplit CV) ===")
    for name, (model, params) in grid_defs.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('est', model)])
        param_grid = {'est__' + k: v for k,v in params.items()}
        gs = GridSearchCV(pipe, param_grid=param_grid,
                          cv=tscv,
                          scoring='neg_root_mean_squared_error',
                          n_jobs=-1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        cv_rmse = -gs.best_score_
        print(f"{name:8s} → best_params={gs.best_params_}, CV_RMSE={cv_rmse:.2f}")
        best_models[name] = best

    # 4.4 Test performansı
    print("\n=== TEST PERFORMANSI ===")
    results = {}
    for name, model in best_models.items():
        pred = model.predict(X_test)
        results[name] = evaluate(y_test.values, pred, name)

    # Naif benchmark
    print()
    naive_pred = X_test['close_lag1'].values
    evaluate(y_test.values, naive_pred, 'Naif(lag1)')

    # 4.5 En iyi modeli seçip grafiğini kaydet
    best_name = min(results, key=lambda n: results[n][0])
    best_model = best_models[best_name]
    print(f"\n→ En iyi model: {best_name}")

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(y_test.index, y_test, label='Gerçek', lw=2)
    ax.plot(y_test.index, best_model.predict(X_test),
            label=f'Tahmin ({best_name})', lw=2)
    ax.set_title(f'BTC 1 Gün Tahmin – {best_name}')
    ax.set_xlabel('Tarih')
    ax.set_ylabel('Fiyat (USD)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('btc_best_forecast_extended.png', dpi=150)
    plt.show()

if __name__=='__main__':
    main()
