# Gold Price Direction Classifier (ML Trading System)

## Overview
This project implements a complete machine-learning-based trading system to predict the next-period direction of gold prices.

The system covers the full lifecycle:
- Data ingestion
- Feature engineering
- Model training
- Signal generation
- Backtesting
- Risk management
- Walk-forward validation

## Problem Statement
Predict whether the gold price will move UP or DOWN in the next trading period using historical price behavior and technical indicators.

## Features Used
- Lagged returns
- RSI
- MACD
- EMA (10, 20)
- Bollinger Bands (Lower)
- High-Low price range

## Model
- CatBoost Classifier
- Class imbalance handled using class weights
- Probability-based decision thresholding

## Evaluation
- Precision / Recall / F1-score
- Transaction-cost-aware backtesting
- Stop-loss & take-profit risk control
- Walk-forward validation to prevent data leakage

## Results
- Stable walk-forward accuracy (~52%)
- High-quality trading signals
- Profitable after costs and risk controls

## Usage
Train & evaluate:
```bash
python main.py
