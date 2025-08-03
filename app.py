# app.py
import streamlit as st
import pandas as pd
import numpy as np

from pycoingecko import CoinGeckoAPI
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(layout="wide")
st.title("🚀 Bitcoin 1-Gün İleri Fiyat Tahmin Dashboard")

@st.cache_data
def load_data(days=365):
    cg = CoinGeckoAPI()
    raw = cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='usd', days=days)
    df = pd.DataFrame(raw['prices'], columns=['ts_ms','price'])
    df['ts'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True)
    df = df.set_index('ts').drop(columns=['ts_ms'])
    return df['price'].resample('D').mean().to_frame('close')

def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0,np.nan)
    return (100 - (100/(1+rs))).fillna(50)

def macd(series, fast=12, slow=26, signal=9):
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    sig_line  = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, sig_line, macd_line - sig_line

@st.cache_data
def make_features(df):
    out = df.copy()
    out['close_lag1'] = out['close'].shift(1)
    out['ma7']        = out['close'].rolling(7).mean()
    out['ma30']       = out['close'].rolling(30).mean()
    out['rsi14']      = rsi(out['close'])
    macd_l, sig, hist = macd(out['close'])
    out['macd']        = macd_l
    out['macd_signal'] = sig
    out['macd_hist']   = hist
    out['std20']       = out['close'].rolling(20).std()
    out['boll_w']      = out['std20'] * 4
    out['target']      = out['close'].shift(-1)
    return out.dropna()

@st.cache_data
def train_pipelines(X_train, y_train):
    pipes = {
        'Naive': None,
        'Ridge': Pipeline([('scaler', StandardScaler()),
                           ('est',   Ridge(alpha=5))]),
        'Lasso': Pipeline([('scaler', StandardScaler()),
                           ('est',   Lasso(alpha=1, max_iter=5000))]),
        'ElasticNet': Pipeline([('scaler', StandardScaler()),
                                ('est',   ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000))]),
        'RF': Pipeline([('est', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))]),
        'XGB': Pipeline([('est', XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                                              objective='reg:squarederror', random_state=42))])
    }
    for name, pipe in pipes.items():
        if name!='Naive':
            pipe.fit(X_train, y_train)
    return pipes

def evaluate(true, pred):
    mse  = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true-pred)/true)) * 100
    return rmse, mae, mape

# ────────────────────────────────────────────────────────────
# Load & prepare data
df   = load_data()
data = make_features(df)
features = [c for c in data.columns if c!='target']
X, y = data[features], data['target']

# Split
split = 60
X_train, X_test = X.iloc[:-split], X.iloc[-split:]
y_train, y_test = y.iloc[:-split], y.iloc[-split:]

# Train
pipes = train_pipelines(X_train, y_train)

# Evaluate all
results, preds = [], {}
for name, pipe in pipes.items():
    if name=='Naive':
        pred = X_test['close_lag1'].values
    else:
        pred = pipe.predict(X_test)
    rmse, mae, mape = evaluate(y_test.values, pred)
    results.append((name, rmse, mae, mape))
    preds[name] = pred

metrics = pd.DataFrame(results, columns=['Model','RMSE','MAE','MAPE'])\
            .set_index('Model').sort_values('MAPE')

# Sidebar
model_choice = st.sidebar.selectbox("Model Seçin", metrics.index.tolist(), index=metrics.index.get_loc('Lasso'))
st.sidebar.write("## CV & Test Metrikleri")
st.sidebar.dataframe(metrics)

# Main panel
st.subheader("Seçilen Model: " + model_choice)
df_plot = pd.DataFrame({
    'Gerçek':      y_test,
    'Tahmin':      preds[model_choice]
}, index=y_test.index)
st.line_chart(df_plot)

# Residuals
st.subheader("Residuals (Gerçek − Tahmin)")
st.line_chart(pd.DataFrame({
    'Residual': y_test.values - preds[model_choice]
}, index=y_test.index))

# Feature importance for linear models
if model_choice in ['Ridge','Lasso','ElasticNet']:
    st.subheader("Özellik Önemleri (|β|)")
    coefs = pipes[model_choice].named_steps['est'].coef_
    fi = pd.Series(np.abs(coefs), index=features).sort_values(ascending=False)
    st.bar_chart(fi)
st.write("🚀 Dashboard yüklendi!")
