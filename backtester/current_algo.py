from typing import Any, List
import string
import json
import numpy as np
import re

from collections import OrderedDict

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId


class Logger:
    # Set this to true, if u want to create
    # local logs
    local: bool
    # this is used as a buffer for logs
    # instead of stdout
    local_logs: dict[int, str] = {}

    def __init__(self, local=False) -> None:
        self.logs = ""
        self.local = local

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], trader_data: str) -> None:
        output = json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":"), sort_keys=False)
        if self.local:
            self.local_logs[state.timestamp] = output

        self.logs = ""



    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            state.traderData,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, int(listing.denomination)])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            sorted_buy_keys = sorted(order_depth.buy_orders.keys(), key=lambda x: float(x), reverse=True)
            ordered_buy = OrderedDict((int(key), order_depth.buy_orders[key]) for key in sorted_buy_keys)
            sell = dict([(int(key), order_depth.sell_orders[key]) for key in order_depth.sell_orders.keys()])
            compressed[symbol] = [dict(ordered_buy), sell]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        return [observations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed


Limits = {"AMETHYSTS" : 20, "STARFRUIT" : 20}


class Trader:
    logger = Logger(local=True)
    def __init__(self):
        self.last5k = {key: [] for key in Limits}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        conversions = 0
        trader_data = ""

        trade_vol = 0
        my_vol = 0
        if state.traderData != "":
            numbers = re.findall(r'\d+', state.traderData)
            trade_vol = int(numbers[0])
            my_vol = int(numbers[1])
    
        self.logger.print("traderData: " + state.traderData)
        self.logger.print("Observations: " + str(state.observations))
        result = {}
        for product in state.listings:
            
            Limit = Limits[product]
            if product not in state.market_trades.keys():
                continue
            trades: List[Trade] = state.market_trades[product]
            orders: List[Order] = []

            for trade in trades:
                self.last5k[product].append(trade)
                trade_vol+=trade.quantity

            if product in state.own_trades.keys():
                my_trades: List[Trade] = state.own_trades[product]
                for trade in my_trades:
                    my_vol += abs(trade.quantity)

            #average price of last 5k timestamps
            volume = 0
            price = 0
            for trade in self.last5k[product]:
                if trade.timestamp + 15000 < state.timestamp:
                    self.last5k[product].remove(trade)        
                volume += abs(trade.quantity)
                price += abs(trade.quantity) * trade.price
            fair_value_average = price / volume

            #regression
            x = []
            y = []

            for trade in self.last5k[product]:
                if trade.timestamp in x: 
                    continue
                trades_at_timestamp = [tr for tr in self.last5k[product] if tr.timestamp == trade.timestamp]
                x.append(trade.timestamp)
                price_at_timestamp = sum([tr.price for tr in trades_at_timestamp]) / len(trades_at_timestamp)
                y.append(price_at_timestamp)

            x = np.array(x)
            y = np.array(y)
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]

            fair_value_regression = m*state.timestamp + c

            self.logger.print("Fair price : " + str(fair_value_average))
            self.logger.print("Fair price regression: " + str(fair_value_regression))

            if product == "AMETHYSTS": fair_value_regression = 10000

            bid = round(fair_value_regression) -2
            ask = bid+4

            if (state.position.get(product, 0) / Limit > 0.5):
                bid -= 1
                ask -= 1
            if (state.position.get(product, 0) / Limit < -0.5):
                bid += 1
                ask += 1

            orders.append(Order(product, bid, Limit - state.position.get(product, 0)))
            orders.append(Order(product, ask, - Limit - state.position.get(product, 0)))
            
            result[product] = orders
    
    
        trader_data = f'The trade volume is {trade_vol} and {my_vol}.'
        

        self.logger.flush(state, result, trader_data)
        return result, trader_data