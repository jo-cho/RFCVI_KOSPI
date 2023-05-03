# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 23:18:11 2021

@author: JHCho
"""
import pandas as pd
import numpy as np
import ta


def ohlcv(df, adj=False):
    df.columns = [i.lower() for i in df.columns]
    close = pd.to_numeric(df.close)
    open = pd.to_numeric(df.open)
    high = pd.to_numeric(df.high)
    low = pd.to_numeric(df.low)
    volume = pd.to_numeric(df.volume)
    if adj is True:
        adjclose = pd.to_numeric(df['adj close'])
        df_ohlcv = pd.DataFrame([open,high,low,close,volume,adjclose]).T
    else:
        df_ohlcv = pd.DataFrame([open,high,low,close,volume]).T
    return df_ohlcv


def mom_std(df, windows_mom, windows_std):
    mkt = df.copy()

    for i in windows_mom:
        mkt = mkt.join(df.volume.pct_change(i).round(6).rename('vol_change_{}'.format(i)))
        mkt = mkt.join(df.close.pct_change(i).round(6).rename('ret_{}'.format(i)))

    for i in windows_std:
        mkt = mkt.join(df.close.rolling(i).std().rename('std_{}'.format(i)))
        mkt = mkt.join(df.volume.rolling(i).std().rename('vol_std_{}'.format(i)))
    mkt = mkt.drop(columns=df.columns)
    mkt = mkt.reindex(sorted(mkt.columns),axis=1)
    return mkt


#----------------------------------------- TA


from ta.momentum import (
    RSIIndicator,
    WilliamsRIndicator
)
from ta.trend import (
    MACD,
    ADXIndicator,
    DPOIndicator
)
from ta.volatility import (
    AverageTrueRange,
    UlcerIndex
)
from ta.volume import (
    ForceIndexIndicator,
    MFIIndicator
)

def get_my_ta(
    df_: pd.DataFrame,
    fillna: bool = False,
    mt = 1
) -> pd.DataFrame:

    df = ohlcv(df_)
    open=df.open ; high=df.high ; low=df.low ; close=df.close ; volume=df.volume

    # VOLUME
    # Force Index
    df["Trend-Volume FI({})".format(13*mt)] = ForceIndexIndicator(
        close=close, volume=volume, window=13*mt, fillna=fillna
    ).force_index()

    # Money Flow Indicator
    df["Trend-Volume MFI({})".format(14*mt)] = MFIIndicator(
        high=high,
        low=low,
        close=close,
        volume=volume,
        window=14*mt,
        fillna=fillna,
    ).money_flow_index()

    #VOLATILITY

    # Average True Range
    df["Volatility ATR({})".format(14*mt)] = AverageTrueRange(
        close=close, high=high, low=low, window=14*mt, fillna=fillna
    ).average_true_range()


    # Standard Deviation
    df["Volatility STD({})".format(20*mt)] = close.rolling(20*mt).std()
    
    # MACD
    indicator_macd = MACD(
        close=close, window_slow=25*mt, window_fast=12*mt, window_sign=9*mt, fillna=fillna
    )
    df["Trend MACD_Diff ({},{},{})".format(26*mt,12*mt,9*mt)] = indicator_macd.macd_diff()
    df["Trend MACD ({},{})".format(26 * mt, 12 * mt)] = indicator_macd.macd()

    # Average Directional Movement Index (ADX)
    indicator_adx = ADXIndicator(
        high=high, low=low, close=close, window=14*mt, fillna=fillna
    )
    df["Trend ADX({})".format(14*mt)] = indicator_adx.adx()

    # DPO Indicator
    df["Trend DPO({})".format(20*mt)] = DPOIndicator(
        close=close, window=20*mt, fillna=fillna
    ).dpo()

    # Relative Strength Index (RSI)
    df["Trend RSI({})".format(14*mt)] = RSIIndicator(
        close=close, window=14*mt, fillna=fillna
    ).rsi()

    # Williams R Indicator
    df["Trend WR({})".format(14*mt)] = WilliamsRIndicator(
        high=high, low=low, close=close, lbp=14*mt, fillna=fillna
    ).williams_r()
    
    df = df.drop(columns=['open','high','low','close','volume'])
    return df

def get_my_ta_windows(df,mts):
    TA = []
    for mt in mts:
        TA.append(get_my_ta(df,mt=mt))
    TA = pd.concat(TA,axis=1)
    TA = TA.loc[:,~TA.columns.duplicated()]
    TA = TA.reindex(sorted(TA.columns),axis=1)
    return TA
