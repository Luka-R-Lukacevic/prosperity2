from static_regression_fixed_price_history import Trader

from datamodel import *
from typing import Any  #, Callable
import numpy as np
import pandas as pd
import statistics
import copy
import uuid
import random
import os
from datetime import datetime

# Timesteps used in training files
TIME_DELTA = 100
# Please put all! the price and log files into
# the same directory or adjust the code accordingly
TRAINING_DATA_PREFIX = "./training"

ALL_SYMBOLS = [
    'AMETHYSTS',
    'STARFRUIT',
    'PEARLS',
    'BANANAS',
    'COCONUTS',
    'PINA_COLADAS',
    'DIVING_GEAR',
    'BERRIES',
    'DOLPHIN_SIGHTINGS',
    'BAGUETTE',
    'DIP',
    'UKULELE',
    'PICNIC_BASKET'
]
POSITIONABLE_SYMBOLS = [
    'AMETHYSTS',
    'STARFRUIT',
    'PEARLS',
    'BANANAS',
    'COCONUTS',
    'PINA_COLADAS',
    'DIVING_GEAR',
    'BERRIES',
    'BAGUETTE',
    'DIP',
    'UKULELE',
    'PICNIC_BASKET'
]
nullth_round = ['AMETHYSTS', 'STARFRUIT']
first_round = ['PEARLS', 'BANANAS']
snd_round = first_round + ['COCONUTS',  'PINA_COLADAS']
third_round = snd_round + ['DIVING_GEAR', 'DOLPHIN_SIGHTINGS', 'BERRIES']
fourth_round = third_round + ['BAGUETTE', 'DIP', 'UKULELE', 'PICNIC_BASKET']
fifth_round = fourth_round # + secret, maybe pirate gold?

SYMBOLS_BY_ROUND = {
    0: nullth_round,
    1: first_round,
    2: snd_round,
    3: third_round,
    4: fourth_round,
    5: fifth_round,
}

nullth_round_pst = ['AMETHYSTS', 'STARFRUIT']
first_round_pst = ['PEARLS', 'BANANAS']
snd_round_pst = first_round_pst + ['COCONUTS',  'PINA_COLADAS']
third_round_pst = snd_round_pst + ['DIVING_GEAR', 'BERRIES']
fourth_round_pst = third_round_pst + ['BAGUETTE', 'DIP', 'UKULELE', 'PICNIC_BASKET']
fifth_round_pst = fourth_round_pst # + secret, maybe pirate gold?

SYMBOLS_BY_ROUND_POSITIONABLE = {
    0: nullth_round_pst,
    1: first_round_pst,
    2: snd_round_pst,
    3: third_round_pst,
    4: fourth_round_pst,
    5: fifth_round_pst,
}

def process_prices(df_prices, round, time_limit) -> dict[int, TradingState]:
    states = {}
    for _, row in df_prices.iterrows():
        time = int(row["timestamp"])
        if time > time_limit:
            break

        product = row["product"]
        state = states.setdefault(time, TradingState("", time, {}, {}, {}, {}, {}, {}))

        if product not in state.position and product in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
            state.position[product] = 0
            state.own_trades[product] = []
            state.market_trades[product] = []

        state.listings[product] = Listing(product, product, "1")
        if product == "DOLPHIN_SIGHTINGS":
            state.observations["DOLPHIN_SIGHTINGS"] = row['mid_price']

        depth = state.order_depths.setdefault(product, OrderDepth())

        # Populating buy and sell orders ensuring price keys are integers
        depth.buy_orders = {int(row[f"bid_price_{i}"]): int(row[f"bid_volume_{i}"])
                            for i in range(1, 4) if row[f"bid_price_{i}"] > 0}

        depth.sell_orders = {int(row[f"ask_price_{i}"]): -int(row[f"ask_volume_{i}"])
                             for i in range(1, 4) if row[f"ask_price_{i}"] > 0}

    return states


import random

def process_trades(df_trades, states: dict[int, TradingState], time_limit, names=True):
    for _, trade in df_trades.iterrows():
        time = int(trade['timestamp'])
        if time > time_limit:
            break

        symbol = trade['symbol']
        trade_quantity = trade['quantity']
        trade_price = int(trade['price'])  # Convert trade price to integer
        order_depth = states[time].order_depths[symbol]

        if symbol not in states[time].market_trades:
            states[time].market_trades[symbol] = []
        if symbol not in states[time].order_depths:
            states[time].order_depths[symbol] = OrderDepth()

        user_buyer = (str(trade['buyer']) == "SUBMISSION")
        user_seller = (str(trade['seller']) == "SUBMISSION")

        if user_buyer:
            #t = Trade(symbol, trade_price, trade_quantity, None, str(trade['seller']), time)
            #states[time].market_trades[symbol].append(t)
            if order_depth.sell_orders.get(trade_price, 0) ==0:
                order_depth.sell_orders[trade_price] = order_depth.sell_orders.get(trade_price, 0)-trade_quantity

        if user_seller:
            #t = Trade(symbol, trade_price, trade_quantity, str(trade['buyer']), None, time)
            #states[time].market_trades[symbol].append(t)
            if order_depth.buy_orders.get(trade_price, 0) ==0:
                order_depth.buy_orders[trade_price] = order_depth.buy_orders.get(trade_price, 0) + trade_quantity
        else:
            t = Trade(symbol, trade_price, trade_quantity, str(trade['buyer']), str(trade['seller']), time)
            states[time].market_trades[symbol].append(t)

        # Randomly decide whether to adjust the bid or ask price in case both are not the User
        adjust_price = random.choice(['bid', 'ask'])

        user = user_buyer or user_seller
        
        if adjust_price == 'bid' and not user:
            # Decrease bid price by 1 (making sure it's at least 1)
            adjusted_bid_price = trade_price - 1
            order_depth.buy_orders[adjusted_bid_price] = order_depth.buy_orders.get(adjusted_bid_price, 0) + trade_quantity
            order_depth.sell_orders[trade_price] = order_depth.sell_orders.get(trade_price, 0) -trade_quantity
        
        elif adjust_price == 'ask' and not user:
            # Increase ask price by 1
            adjusted_ask_price = trade_price + 1
            order_depth.sell_orders[adjusted_ask_price] = order_depth.sell_orders.get(adjusted_ask_price, 0) - trade_quantity
            order_depth.buy_orders[trade_price] = order_depth.buy_orders.get(trade_price, 0) + trade_quantity
        
        order_depth.buy_orders = dict(sorted(order_depth.buy_orders.items(), key=lambda item: item[0], reverse=True))
        order_depth.sell_orders = dict(sorted(order_depth.sell_orders.items(), key=lambda item: item[0], reverse=False))
    return states







       
current_limits = {
    "AMETHYSTS":20,
    "STARFRUIT": 20,
    'PEARLS': 20,
    'BANANAS': 20,
    'COCONUTS': 600,
    'PINA_COLADAS': 300,
    'DIVING_GEAR': 50,
    'BERRIES': 250,
    'BAGUETTE': 150,
    'DIP': 300,
    'UKULELE': 70,
    'PICNIC_BASKET': 70,
}

def calc_mid(states: dict[int, TradingState], round: int, time: int, max_time: int) -> dict[str, float]:
    medians_by_symbol = {}
    non_empty_time = time
    for psymbol in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
        hitted_zero = False
        while len(states[non_empty_time].order_depths[psymbol].sell_orders.keys()) == 0 or len(states[non_empty_time].order_depths[psymbol].buy_orders.keys()) == 0:
            # little hack
            if time == 0 or hitted_zero and time != max_time:
                hitted_zero = True
                non_empty_time += TIME_DELTA
            else:
                non_empty_time -= TIME_DELTA
        min_ask = min(states[non_empty_time].order_depths[psymbol].sell_orders.keys())
        max_bid = max(states[non_empty_time].order_depths[psymbol].buy_orders.keys())
        median_price = statistics.median([min_ask, max_bid])
        medians_by_symbol[psymbol] = median_price
    return medians_by_symbol


# Setting a high time_limit can be harder to visualize
# print_position prints the position before! every Trader.run
def simulate_alternative(
        round: int, 
        day: int, 
        trader, 
        time_limit=999900, 
        names=True, 
        halfway=True,
        monkeys=False,
        monkey_names=['Caesar', 'Camilla', 'Peter']
    ):
    prices_path = os.path.join(TRAINING_DATA_PREFIX, f'prices_round_{round}_day_{day}.csv')
    trades_path = os.path.join(TRAINING_DATA_PREFIX, f'trades_round_{round}_day_{day}_wn.csv')
    if not names:
        trades_path = os.path.join(TRAINING_DATA_PREFIX, f'trades_round_{round}_day_{day}_nn.csv')
    df_prices = pd.read_csv(prices_path, sep=';')
    df_trades = pd.read_csv(trades_path, sep=';', dtype={ 'seller': str, 'buyer': str })

    states = process_prices(df_prices, round, time_limit)
    states = process_trades(df_trades, states, time_limit, names)
    ref_symbols = list(states[0].position.keys())
    max_time = max(list(states.keys()))

    # handling these four is rather tricky 
    profits_by_symbol: dict[int, dict[str, float]] = { 0: dict(zip(ref_symbols, [0.0]*len(ref_symbols))) }
    balance_by_symbol: dict[int, dict[str, float]] = { 0: copy.deepcopy(profits_by_symbol[0]) }
    credit_by_symbol: dict[int, dict[str, float]] = { 0: copy.deepcopy(profits_by_symbol[0]) }
    unrealized_by_symbol: dict[int, dict[str, float]] = { 0: copy.deepcopy(profits_by_symbol[0]) }

    states, trader, profits_by_symbol, balance_by_symbol = trades_position_pnl_run(states, max_time, profits_by_symbol, balance_by_symbol, credit_by_symbol, unrealized_by_symbol)
    create_log_file(round, day, states, profits_by_symbol, balance_by_symbol, trader)
    profit_balance_monkeys = {}
    trades_monkeys = {}
    if monkeys:
        profit_balance_monkeys, trades_monkeys, profit_monkeys, balance_monkeys, monkey_positions_by_timestamp = monkey_positions(monkey_names, states, round)
        print("End of monkey simulation reached.")
        print(f'PNL + BALANCE monkeys {profit_balance_monkeys[max_time]}')
        print(f'Trades monkeys {trades_monkeys[max_time]}')
    if hasattr(trader, 'after_last_round'):
        if callable(trader.after_last_round): #type: ignore
            trader.after_last_round(profits_by_symbol, balance_by_symbol) #type: ignore


def trades_position_pnl_run(
        states: dict[int, TradingState],
        max_time: int, 
        profits_by_symbol: dict[int, dict[str, float]], 
        balance_by_symbol: dict[int, dict[str, float]], 
        credit_by_symbol: dict[int, dict[str, float]], 
        unrealized_by_symbol: dict[int, dict[str, float]], 
        ):
        for time, state in states.items():
            position = copy.deepcopy(state.position)
            orders, data = trader.run(state)
            if time+TIME_DELTA <= max_time:
                states[time+TIME_DELTA].traderData = data
            trades = clear_order_book(orders, state.order_depths, time, halfway)
            mids = calc_mid(states, round, time, max_time)
            if profits_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                profits_by_symbol[time + TIME_DELTA] = copy.deepcopy(profits_by_symbol[time])
            if credit_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                credit_by_symbol[time + TIME_DELTA] = copy.deepcopy(credit_by_symbol[time])
            if balance_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                balance_by_symbol[time + TIME_DELTA] = copy.deepcopy(balance_by_symbol[time])
            if unrealized_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                unrealized_by_symbol[time + TIME_DELTA] = copy.deepcopy(unrealized_by_symbol[time])
                for psymbol in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
                    unrealized_by_symbol[time + TIME_DELTA][psymbol] = mids[psymbol]*position[psymbol]
            valid_trades = []
            failed_symbol = []
            grouped_by_symbol = {}
            if len(trades) > 0:
                for trade in trades:
                    if trade.symbol in failed_symbol:
                        continue
                    n_position = position[trade.symbol] + trade.quantity
                    if abs(n_position) > current_limits[trade.symbol]:
                        print('ILLEGAL TRADE, WOULD EXCEED POSITION LIMIT, KILLING ALL REMAINING ORDERS')
                        trade_vars = vars(trade)
                        trade_str = ', '.join("%s: %s" % item for item in trade_vars.items())
                        print(f'Stopped at the following trade: {trade_str}')
                        print(f"All trades that were sent:")
                        for trade in trades:
                            trade_vars = vars(trade)
                            trades_str = ', '.join("%s: %s" % item for item in trade_vars.items())
                            print(trades_str)
                        failed_symbol.append(trade.symbol)
                    else:
                        valid_trades.append(trade) 
                        position[trade.symbol] += trade.quantity
            FLEX_TIME_DELTA = TIME_DELTA
            if time == max_time:
                FLEX_TIME_DELTA = 0
            for valid_trade in valid_trades:
                    if grouped_by_symbol.get(valid_trade.symbol) == None:
                        grouped_by_symbol[valid_trade.symbol] = []
                    grouped_by_symbol[valid_trade.symbol].append(valid_trade)
                    credit_by_symbol[time + FLEX_TIME_DELTA][valid_trade.symbol] += -valid_trade.price * valid_trade.quantity
            if states.get(time + FLEX_TIME_DELTA) != None:
                states[time + FLEX_TIME_DELTA].own_trades = grouped_by_symbol
                for psymbol in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
                    unrealized_by_symbol[time + FLEX_TIME_DELTA][psymbol] = mids[psymbol]*position[psymbol]
                    if position[psymbol] == 0 and states[time].position[psymbol] != 0:
                        profits_by_symbol[time + FLEX_TIME_DELTA][psymbol] += credit_by_symbol[time + FLEX_TIME_DELTA][psymbol] #+unrealized_by_symbol[time + FLEX_TIME_DELTA][psymbol]
                        credit_by_symbol[time + FLEX_TIME_DELTA][psymbol] = 0
                        balance_by_symbol[time + FLEX_TIME_DELTA][psymbol] = 0
                    else:
                        balance_by_symbol[time + FLEX_TIME_DELTA][psymbol] = credit_by_symbol[time + FLEX_TIME_DELTA][psymbol] + unrealized_by_symbol[time + FLEX_TIME_DELTA][psymbol]

            if time == max_time:
                print("End of simulation reached. All positions left are liquidated")
                # i have the feeling this already has been done, and only repeats the same values as before
                for osymbol in position.keys():
                    profits_by_symbol[time + FLEX_TIME_DELTA][osymbol] += credit_by_symbol[time + FLEX_TIME_DELTA][osymbol] + unrealized_by_symbol[time + FLEX_TIME_DELTA][osymbol]
                    balance_by_symbol[time + FLEX_TIME_DELTA][osymbol] = 0
            if states.get(time + FLEX_TIME_DELTA) != None:
                states[time + FLEX_TIME_DELTA].position = copy.deepcopy(position)
        return states, trader, profits_by_symbol, balance_by_symbol

def monkey_positions(monkey_names: list[str], states: dict[int, TradingState], round):
    profits_by_symbol: dict[int, dict[str, dict[str, float]]] = { 0: {} }
    balance_by_symbol: dict[int, dict[str, dict[str, float]]] =  { 0: {} }
    credit_by_symbol: dict[int, dict[str, dict[str, float]]] = { 0: {} }
    unrealized_by_symbol: dict[int, dict[str, dict[str, float]]] = { 0: {} }
    prev_monkey_positions: dict[str, dict[str, int]] = {}
    monkey_positions: dict[str, dict[str, int]] = {}
    trades_by_round: dict[int, dict[str, list[Trade]]]  = { 0: dict(zip(monkey_names,  [[] for x in range(len(monkey_names))])) }
    profit_balance: dict[int, dict[str, dict[str, float]]] = { 0: {} }

    monkey_positions_by_timestamp: dict[int, dict[str, dict[str, int]]] = {}

    for monkey in monkey_names:
        ref_symbols = list(states[0].position.keys())
        profits_by_symbol[0][monkey] = dict(zip(ref_symbols, [0.0]*len(ref_symbols)))
        balance_by_symbol[0][monkey] = copy.deepcopy(profits_by_symbol[0][monkey])
        credit_by_symbol[0][monkey] = copy.deepcopy(profits_by_symbol[0][monkey])
        unrealized_by_symbol[0][monkey] = copy.deepcopy(profits_by_symbol[0][monkey])
        profit_balance[0][monkey] = copy.deepcopy(profits_by_symbol[0][monkey])
        monkey_positions[monkey] = dict(zip(SYMBOLS_BY_ROUND_POSITIONABLE[round], [0]*len(SYMBOLS_BY_ROUND_POSITIONABLE[round])))
        prev_monkey_positions[monkey] = copy.deepcopy(monkey_positions[monkey])

    for time, state in states.items():
        already_calculated = False
        for monkey in monkey_names:
            position = copy.deepcopy(monkey_positions[monkey])
            mids = calc_mid(states, round, time, max_time)
            if trades_by_round.get(time + TIME_DELTA) == None:
                trades_by_round[time + TIME_DELTA] =  copy.deepcopy(trades_by_round[time])

            for psymbol in POSITIONABLE_SYMBOLS:
                if already_calculated:
                    break
                if state.market_trades.get(psymbol):
                    for market_trade in state.market_trades[psymbol]:
                        if trades_by_round[time].get(market_trade.buyer) != None:
                            trades_by_round[time][market_trade.buyer].append(Trade(psymbol, market_trade.price, market_trade.quantity))
                        if trades_by_round[time].get(market_trade.seller) != None:
                            trades_by_round[time][market_trade.seller].append(Trade(psymbol, market_trade.price, -market_trade.quantity))
            already_calculated = True

            if profit_balance.get(time + TIME_DELTA) == None and time != max_time:
                profit_balance[time + TIME_DELTA] = copy.deepcopy(profit_balance[time])
            if profits_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                profits_by_symbol[time + TIME_DELTA] = copy.deepcopy(profits_by_symbol[time])
            if credit_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                credit_by_symbol[time + TIME_DELTA] = copy.deepcopy(credit_by_symbol[time])
            if balance_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                balance_by_symbol[time + TIME_DELTA] = copy.deepcopy(balance_by_symbol[time])
            if unrealized_by_symbol.get(time + TIME_DELTA) == None and time != max_time:
                unrealized_by_symbol[time + TIME_DELTA] = copy.deepcopy(unrealized_by_symbol[time])
                for psymbol in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
                    unrealized_by_symbol[time + TIME_DELTA][monkey][psymbol] = mids[psymbol]*position[psymbol]
            valid_trades = []
            if trades_by_round[time].get(monkey) != None:  
                valid_trades = trades_by_round[time][monkey]
            FLEX_TIME_DELTA = TIME_DELTA
            if time == max_time:
                FLEX_TIME_DELTA = 0
            for valid_trade in valid_trades:
                    position[valid_trade.symbol] += valid_trade.quantity
                    credit_by_symbol[time + FLEX_TIME_DELTA][monkey][valid_trade.symbol] += -valid_trade.price * valid_trade.quantity
            if states.get(time + FLEX_TIME_DELTA) != None:
                for psymbol in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
                    unrealized_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol] = mids[psymbol]*position[psymbol]
                    if position[psymbol] == 0 and prev_monkey_positions[monkey][psymbol] != 0:
                        profits_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol] += credit_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol]
                        credit_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol] = 0
                        balance_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol] = 0
                    else:
                        balance_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol] = credit_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol] + unrealized_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol]
                    profit_balance[time + FLEX_TIME_DELTA][monkey][psymbol] = profits_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol] + balance_by_symbol[time + FLEX_TIME_DELTA][monkey][psymbol]
            prev_monkey_positions[monkey] = copy.deepcopy(monkey_positions[monkey])
            monkey_positions[monkey] = position
            if time == max_time:
                # i have the feeling this already has been done, and only repeats the same values as before
                for osymbol in position.keys():
                    profits_by_symbol[time + FLEX_TIME_DELTA][monkey][osymbol] += credit_by_symbol[time + FLEX_TIME_DELTA][monkey][osymbol] + unrealized_by_symbol[time + FLEX_TIME_DELTA][monkey][osymbol]
                    balance_by_symbol[time + FLEX_TIME_DELTA][monkey][osymbol] = 0
        monkey_positions_by_timestamp[time] = copy.deepcopy(monkey_positions)
    return profit_balance, trades_by_round, profits_by_symbol, balance_by_symbol, monkey_positions_by_timestamp


def cleanup_order_volumes(org_orders: List[Order]) -> List[Order]:
    orders = {}
    for order in org_orders:
        if order.price in orders:
            orders[order.price].quantity += order.quantity
        else:
            orders[order.price] = copy.copy(order)
    return list(orders.values())

def match_and_update_order_depth(order, symbol_order_depth, order_type):
    #print(order, symbol_order_depth, order_type, 123)
    # Convert prices to int and filter potential matches based on the order type
    potential_matches = {int(k): v for k, v in (symbol_order_depth.buy_orders if order_type == 'sell' else symbol_order_depth.sell_orders).items()}
    price_check = lambda p: p >= order.price if order_type == 'sell' else p <= order.price
    potential_prices = [price for price in potential_matches if price_check(price)]

    # Handle case where there are no matching prices
    if not potential_prices:
        return None, 0

    # Determine the best price for trade and calculate trade volume
    best_price = max(potential_prices) if order_type == 'sell' else min(potential_prices)
    match_qty = potential_matches[best_price]
    trade_volume = min(abs(order.quantity), abs(match_qty))
    if order_type == 'sell':
        trade_volume = -trade_volume  # Make volume negative for sell orders

    # Update the order depth with the matched trade
    potential_matches[best_price] += trade_volume
    if potential_matches[best_price] == 0:  # Remove the price level if quantity becomes 0
        del potential_matches[best_price]

    # Update the original order depth with modified values
    if order_type == 'sell':
        symbol_order_depth.buy_orders = potential_matches
    else:
        symbol_order_depth.sell_orders = potential_matches
    #print(order, symbol_order_depth, order_type,456)
    return best_price, trade_volume


def clear_order_book(trader_orders, order_depth, time, halfway):
    trades = []
    #print(time, "Orders: ", trader_orders)
    for symbol, orders in trader_orders.items():
        if symbol in order_depth:
            symbol_order_depth = copy.deepcopy(order_depth[symbol])
            cleaned_orders = cleanup_order_volumes(copy.deepcopy(orders))
            #print(cleaned_orders, symbol_order_depth)
            while cleaned_orders:
                #print("cleaned_orders new round")
                order = cleaned_orders[0]  # Always work with the first order in the list
                order_type = 'sell' if order.quantity < 0 else 'buy'
                #print("order, order_depth, order type",order, symbol_order_depth, order_type)
                best_price, trade_volume = match_and_update_order_depth(order, symbol_order_depth, order_type)
                #print("best price and vol",best_price, trade_volume)
                if best_price is not None:
                    trades.append(Trade(symbol, best_price, trade_volume, "YOU" if order_type == "buy" else "BOT", "BOT" if order_type == "buy" else "YOU", time))
                    
                    adjusted_quantity = order.quantity - trade_volume

                    # If the order is fully matched or there's no more quantity to match, remove it from the list
                    if adjusted_quantity == 0:
                        cleaned_orders.pop(0)
                    else:
                        order.quantity = adjusted_quantity
                else:
                    cleaned_orders.pop(0)
                #print(cleaned_orders, "HElllouu")

    return trades







                            
csv_header = "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss\n"
log_header = [
    'Sandbox logs:\n'
]
trades_header = ['Trade History:\n']

def create_log_file(round: int, day: int, states: dict[int, TradingState], profits_by_symbol: dict[int, dict[str, float]], balance_by_symbol: dict[int, dict[str, float]], trader: Trader):
    file_name = uuid.uuid4()
    timest = datetime.timestamp(datetime.now())
    max_time = max(list(states.keys()))
    log_path = os.path.join('logs', f'{timest}_{file_name}.log')
    with open(log_path, 'w', encoding="utf-8", newline='\n') as f:
        f.writelines(log_header)
        for time, state in states.items():
            if hasattr(trader, 'logger'):
                if hasattr(trader.logger, 'local_logs') != None:
                    if trader.logger.local_logs.get(time) != None:
                        log_entry = {
                                        "sandboxLog": "",
                                        "lambdaLog": trader.logger.local_logs[time],
                                        "timestamp": time
                                    }
                        f.write(json.dumps(log_entry, indent=2))
                        f.write("\n")
                        continue
            if time != 0:
                f.write(f'{time}\n')

        f.write(f'\n\n')
        f.write('Submission logs:\n\n\n')
        f.write('Activities log:\n')
        f.write(csv_header)
        for time, state in states.items():
            for symbol in SYMBOLS_BY_ROUND[round]:
                f.write(f'{day};{time};{symbol};')
                bids_length = len(state.order_depths[symbol].buy_orders)
                bids = list(state.order_depths[symbol].buy_orders.items())
                bids_prices = list(state.order_depths[symbol].buy_orders.keys())
                bids_prices.sort()
                asks_length = len(state.order_depths[symbol].sell_orders)
                asks_prices = list(state.order_depths[symbol].sell_orders.keys())
                asks_prices.sort()
                asks = list(state.order_depths[symbol].sell_orders.items())
                if bids_length >= 3:
                    f.write(f'{bids[0][0]};{bids[0][1]};{bids[1][0]};{bids[1][1]};{bids[2][0]};{bids[2][1]};')
                elif bids_length == 2:
                    f.write(f'{bids[0][0]};{bids[0][1]};{bids[1][0]};{bids[1][1]};;;')
                elif bids_length == 1:
                    f.write(f'{bids[0][0]};{bids[0][1]};;;;;')
                else:
                    f.write(f';;;;;;')
                if asks_length >= 3:
                    f.write(f'{asks[0][0]};{asks[0][1]};{asks[1][0]};{asks[1][1]};{asks[2][0]};{asks[2][1]};')
                elif asks_length == 2:
                    f.write(f'{asks[0][0]};{asks[0][1]};{asks[1][0]};{asks[1][1]};;;')
                elif asks_length == 1:
                    f.write(f'{asks[0][0]};{asks[0][1]};;;;;')
                else:
                    f.write(f';;;;;;')
                if len(asks_prices) == 0 or max(bids_prices) == 0:
                    if symbol == 'DOLPHIN_SIGHTINGS':
                        dolphin_sightings = state.observations['DOLPHIN_SIGHTINGS']
                        f.write(f'{dolphin_sightings};{0.0}\n')
                    else:
                        f.write(f'{0};{0.0}\n')
                else:
                    actual_profit = 0.0
                    if symbol in SYMBOLS_BY_ROUND_POSITIONABLE[round]:
                            actual_profit = profits_by_symbol[time][symbol] + balance_by_symbol[time][symbol]
                    min_ask = min(asks_prices)
                    max_bid = max(bids_prices)
                    median_price = statistics.median([min_ask, max_bid])
                    f.write(f'{median_price};{actual_profit}\n')
                    if time == max_time:
                        if profits_by_symbol[time].get(symbol) != None:
                            print(f'Final profit for {symbol} = {actual_profit}')
        print(f"\nSimulation on round {round} day {day} for time {max_time} complete")


# Adjust accordingly the round and day to your needs
if __name__ == "__main__":
    trader = Trader()
    max_time = 199000
    if max_time < 10:
        max_time *= 100000
    round = 0
    day = -2
    names = False
    halfway = True
    print(f"Running simulation on round {round} day {day} for time {max_time}")
    print("Remember to change the trader import")
    simulate_alternative(round, day, trader, max_time, names, halfway, False)
