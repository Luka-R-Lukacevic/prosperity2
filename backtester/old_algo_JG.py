from typing import Any, List
import string
import json
import numpy as np
import re

from collections import OrderedDict

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId

import jsonpickle

class TraderData:
    def __init__(self, my_vol=0, trade_vol=0):
        self.my_vol = my_vol
        self.trade_vol = trade_vol



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

        result = {}
        for product in state.listings:
            Limit = Limits[product]
            if product not in state.market_trades.keys():
                continue
            trades: List[Trade] = state.market_trades[product]
            orders: List[Order] = []

            for trade in trades:
                self.last5k[product].append(trade)

            volume = 0
            price = 0
            for trade in self.last5k[product]:
                if trade.timestamp + 5000 < state.timestamp:
                    self.last5k[product].remove(trade)        
                volume += abs(trade.quantity)
                price += abs(trade.quantity) * trade.price
            estimated_fair_value = price / volume
    
            bid = int (estimated_fair_value) -3
            ask = bid+6

            if (state.position.get(product, 0) / Limit > 0.5):
                bid -= 2
                ask -= 2
            if (state.position.get(product, 0) / Limit < -0.5):
                bid += 2
                ask += 2
                    
            orders.append(Order(product, bid, Limit - state.position.get(product, 0)))
            orders.append(Order(product, ask, - Limit - state.position.get(product, 0)))
            
            result[product] = orders
    
    
        trader_data = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        #conversions = 100

        self.logger.flush(state, result, trader_data)
        return result, trader_data