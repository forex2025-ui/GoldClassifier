def backtest_strategy(df, model, feature_cols, threshold=0.55):

    X = df[feature_cols]
    probs = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["prob_up"] = probs
    df["signal"] = (df["prob_up"] >= threshold).astype(int)

    # Next-day return
    df["next_return"] = df["Close"].pct_change().shift(-1)

    # Stop-loss and take-profit levels
    stop_loss = -0.01
    take_profit = 0.02

    def apply_sl_tp(ret):
        if ret <= stop_loss:
            return stop_loss
        elif ret >= take_profit:
            return take_profit
        else:
            return ret

    # Strategy return with SL/TP
    df["strategy_return"] = df["signal"] * df["next_return"]
    df["strategy_return"] = df["strategy_return"].apply(apply_sl_tp)

    # Transaction cost
    transaction_cost = 0.0005
    df["strategy_return"] -= df["signal"] * transaction_cost

    df.dropna(inplace=True)

    # Metrics
    total_trades = int(df["signal"].sum())
    win_rate = (df[df["strategy_return"] > 0].shape[0]) / max(total_trades, 1)
    cumulative_return = (1 + df["strategy_return"]).prod() - 1

    print("\n📊 BACKTEST RESULTS (SL/TP Applied)")
    print(f"Total Trades      : {total_trades}")
    print(f"Win Rate          : {win_rate:.2%}")
    print(f"Cumulative Return : {cumulative_return:.2%}")

    return df
