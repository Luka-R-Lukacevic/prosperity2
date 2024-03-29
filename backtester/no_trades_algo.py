from typing import Any, List
import string
import json
import numpy as np
import re

from collections import OrderedDict

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId


class Logger:
    local: bool
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


# This is provisionary, if no other algorithm works.
# Better to loose nothing, then dreaming of a gain.
class Trader:

    logger = Logger(local=True)

    def run(self, state: TradingState):
        trader_data = state.traderData
        
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}
        self.logger.flush(state, result, trader_data)
        return result, trader_data
