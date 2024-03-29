# Backtester IMC Prosperity 2024

This folder contains our backtester for Prosperity 2. It is an adaptation of Niklas Jona's backtester [project](https://github.com/n-0/backtest-imc-prosperity-2023) from last year. With our modification it also produces logs that get accepted by jmerle's [project](https://github.com/jmerle/imc-prosperity-2-visualizer) visualizer for this year.

## Order matching
We introduced some changes to the order matching logic to more accurately capture the trading dynamics for this year. Orders returned by the `Trader.run` method, are matched against the `OrderDepth`of the state provided to the method call. The trader always gets their trade and trades from bots are ignored. Any order from the trader that matches with the orderbook (there is an overlap with it) will be executed at the level of the orderbook. If the new position that would result from an order exceeds the specified limit of the symbol, all following orders (including the failing one) are cancelled.

## After All
If a trader has a method called `after_last_round`, it will be called after the logs have been written.
This is useful for plotting something with matplotlib for example (but don't forget to remove the import,
when you upload your algorithm). We have not touched this part of [Niklas](https://github.com/n-0/backtest-imc-prosperity-2023), so it might need additional debugging.

## General usage
There is a folder called `training` with the csv files for all rounds. You can adjust `TRAINING_DATA_PREFIX`
to the full path of `training` directory on your system, at the top of `backtester.py`. Leaving it as is should work fine for simply downloading this repo.

Now import your Trader at the top of `backtester.py` (in the repo the Trader from `current_algo.py` is used). It might be easier to understand the changes needed to be made when going from the uploadable version to the local version to consider the file `no_trades_algo.py`.
Then run
```bash
python backtester.py
```
This executes
```python
if __name__ == "__main__":
    trader = Trader()
    simulate_alternative(round = 0, day = -2, trader, False, max_time = 199000)
```
Here we see the central method, `simulate_alternative`.

## Logging with jmerle's visualizer
Because the `backtester` doesn't read from the stdout nor stderr, logs produced have an empty `Submission logs:` section (still limit exceeds are printed).
Furthermore the default `Logger` from jmerle's project won't do the trick, the adjustments seen in `no_trades_algo.py` make it compatible. Subsequently, every print statement needs to be of the form

```python
self.logger.print("Here we can print something")
```

## Additional Bugs
It is possible that the changes made for compatibility introduced further bugs. It is noticeable that the PnL in the backtester is significantly lower (around half) then the PnL in the uploaded files and the trading volume between bots reduces around half as well. The number of trades executed by the trader go down by a factor more than 10. The reason for this could be that although the backtester works fine, the backtester has significantly less action in it since the trader is only able to trade against the orderbook, whereas in the simulation the trader can send orders that bots then choose to interact with.

The more grim outlook is that there are still bugs in the funtionality of the backtester. Most likely these could be found:
* When creating the orderbook out of the csv files with the function `process_prices` or `process_trades`.
* When processing orders with the function `clear_order_book` (although I did rewrite that function already).
* When calculating PnL via `trades_position_pnl_run` (although this would not explain the low number of trades).
