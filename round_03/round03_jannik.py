import copy
import json
import math
import jsonpickle
import statistics as stat
import collections
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from datamodel import (Listing, Observation, Order, OrderDepth,
                       ProsperityEncoder, Symbol, Trade, TradingState)

empty_dict = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 'ORCHIDS' : 0, 'STRAWBERRIES' : 0, 'CHOCOLATE' : 0, 'ROSES' : 0, 'GIFT_BASKET' : 0}


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
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
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

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
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()


class Trader:

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS' : 100,'STRAWBERRIES' : 350, 'CHOCOLATE' : 250, 'ROSES' : 60, 'GIFT_BASKET' : 60}
    volume_traded = copy.deepcopy(empty_dict)
    
    
    def calc_next_price_starfruit(self, cache, state: TradingState):
        
        y = np.array(cache)
        x = np.array([state.timestamp - 100 * x for x in range(10, 0, -1)])
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        fair_value_regression = m * state.timestamp + c
        
        return int(round(fair_value_regression))


    def values_extract(self, order_dict, buy = 0):
        
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            
            if(buy==0):
                vol *= -1
            
            tot_vol += vol
            
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val


    def calc_orchids_profit_margin(self, import_tariff_obs, transport_fees_obs):
        cost = import_tariff_obs + transport_fees_obs
        
        margin = 1
        if cost >= -1:
            margin = 0.5
        else:
            margin = (-1/2) * cost
        
        return margin


    def compute_orders_amethysts(self, product, state: TradingState):
        
        orders: list[Order] = []
        
        order_depth: OrderDepth = state.order_depths[product]
        
        acc_bid = 10000
        acc_ask = 10000

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product] < 0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT[product]:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT[product] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        mprice_actual = (best_sell_pr + best_buy_pr) / 2
        mprice_ours = (acc_bid + acc_ask) / 2

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid - 1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask + 1)

        if (cpos < self.POSITION_LIMIT[product]) and (self.position[product] < 0):
            num = min(40, self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid - 1), num))
            cpos += num

        if (cpos < self.POSITION_LIMIT[product]) and (self.position[product] > 15):
            num = min(40, self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid - 1), num))
            cpos += num

        if cpos < self.POSITION_LIMIT[product]:
            num = min(40, self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product]>0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMIT[product]:
                order_for = max(-vol, -self.POSITION_LIMIT[product] - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if (cpos > -self.POSITION_LIMIT[product]) and (self.position[product] > 0):
            num = max(-40, -self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, max(undercut_sell - 1, acc_ask + 1), num))
            cpos += num

        if (cpos > -self.POSITION_LIMIT[product]) and (self.position[product] < -15):
            num = max(-40, -self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, max(undercut_sell+1, acc_ask + 1), num))
            cpos += num

        if cpos > -self.POSITION_LIMIT[product]:
            num = max(-40, -self.POSITION_LIMIT[product] - cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders


    def compute_orders_starfruit(self, product, state: TradingState, new_starfruit_cache):
        
        orders: list[Order] = []
        
        order_depth: OrderDepth = state.order_depths[product]
        
        LIMIT = self.POSITION_LIMIT[product]
        
        if len(new_starfruit_cache) == 10:
            new_starfruit_cache.pop(0)

        vol_ask, bs_starfruit = self.values_extract(collections.OrderedDict(sorted(state.order_depths[product].sell_orders.items())))
        vol_bid, bb_starfruit = self.values_extract(collections.OrderedDict(sorted(state.order_depths[product].buy_orders.items(), reverse=True)), 1)

        starfruit_mid = (bs_starfruit + bb_starfruit) / 2
        new_starfruit_cache.append(starfruit_mid)
        
        n = len(new_starfruit_cache)

        INF = 1e9
        
        starfruit_lb = -INF
        starfruit_ub = INF
        
        if n == 10:
            fair = self.calc_next_price_starfruit(new_starfruit_cache, state)
            starfruit_lb = fair - 1
            starfruit_ub = fair + 1
            
        else:
            if n > 0:
                starfruit_lb = starfruit_mid - 1  #for the first 10 timestamps we use the current mid price as our price prediction
                starfruit_ub = starfruit_mid + 1
        
        acc_bid = starfruit_lb
        acc_ask = starfruit_ub

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product] < 0) and (ask == acc_bid + 1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]
        

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product] > 0) and (bid + 1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT - cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return new_starfruit_cache, orders


    def compute_orders_orchids(self, product, state: TradingState):

        order_depth: OrderDepth = state.order_depths[product]
        orders: list[Order] = []
        
        position_limit = self.POSITION_LIMIT[product]
        
        conversions = 0
        
        target_inventory = 0
        
        bid_price_obs = state.observations.conversionObservations[product].bidPrice
        ask_price_obs = state.observations.conversionObservations[product].askPrice
        transport_fees_obs = state.observations.conversionObservations[product].transportFees
        export_tariff_obs = state.observations.conversionObservations[product].exportTariff
        import_tariff_obs = state.observations.conversionObservations[product].importTariff	
        sunlight_obs = state.observations.conversionObservations[product].sunlight              
        humidity_obs = state.observations.conversionObservations[product].humidity              
        
        import_price = ask_price_obs + import_tariff_obs + transport_fees_obs
        export_price = bid_price_obs - export_tariff_obs - transport_fees_obs

        conversions += - self.position[product] + target_inventory

        margin = self.calc_orchids_profit_margin(import_tariff_obs, transport_fees_obs)
        
        orders.append(Order(product, int(round(import_price + margin)), - position_limit))
        orders.append(Order(product, int(round(export_price - 2)), position_limit))
        
        return orders, conversions


    def compute_orders_basket(self, state: TradingState):
        '''
        One Basket is made up of: 6 Strawberries, 4 Chocolates, 1 Rose
        That means Strawberries are the limiting factor here,
        position maximum should be at 58 positions overall, which give the following limits:
        Strawberries = 348
        Chocolates = 232
        Roses = 58
        GIFT_BASKET = 58
        '''
        
        mean_premium = 379.4905
        stdev_premium = 76.4243809265
        
        z_score_for_max_orders = 2
        z_score_for_strategy_start = 0.7
        
        product_1 = 'STRAWBERRIES'
        product_2 = 'CHOCOLATE'
        product_3 = 'ROSES'
        product_4 = 'GIFT_BASKET'
        
        product_1_orders: list[Order] = []
        product_2_orders: list[Order] = []
        product_3_orders: list[Order] = []
        product_4_orders: list[Order] = []
        
        product_1_order_multiplier = 6
        product_2_order_multiplier = 4
        product_3_order_multiplier = 1
        product_4_order_multiplier = 1
        
        product_1_inventory_limit = 348
        product_2_inventory_limit = 232
        product_3_inventory_limit = 58
        product_4_inventory_limit = 58
        
        trade_opportunities = 58
        
        product_1_max_orders = product_1_order_multiplier * trade_opportunities
        product_2_max_orders = product_2_order_multiplier * trade_opportunities
        product_3_max_orders = product_3_order_multiplier * trade_opportunities
        product_4_max_orders = product_4_order_multiplier * trade_opportunities
        
        order_depth_product_1: OrderDepth = state.order_depths[product_1]
        order_depth_product_2: OrderDepth = state.order_depths[product_2]
        order_depth_product_3: OrderDepth = state.order_depths[product_3]
        order_depth_product_4: OrderDepth = state.order_depths[product_4]
        
        initial_position_product_1 = state.position.get(product_1, 0)
        initial_position_product_2 = state.position.get(product_2, 0)
        initial_position_product_3 = state.position.get(product_3, 0)
        initial_position_product_4 = state.position.get(product_4, 0)
        
        best_bid_product_1 = max(order_depth_product_1.buy_orders.keys())
        best_ask_product_1 = min(order_depth_product_1.sell_orders.keys())
        best_bid_volume_product_1 = abs(list(order_depth_product_1.buy_orders.values())[0])
        best_bid_avail_units_product_1 = int(best_bid_volume_product_1 / product_1_order_multiplier)
        best_ask_volume_product_1 = abs(list(order_depth_product_1.sell_orders.values())[0])
        best_ask_avail_units_product_1 = int(best_ask_volume_product_1 /  product_1_order_multiplier)
        mid_price_product_1 = stat.fmean([best_bid_product_1, best_ask_product_1])
        
        best_bid_product_2 = max(order_depth_product_2.buy_orders.keys())
        best_ask_product_2 = min(order_depth_product_2.sell_orders.keys())
        best_bid_volume_product_2 = abs(list(order_depth_product_2.buy_orders.values())[0])
        best_bid_avail_units_product_2 = int(best_bid_volume_product_2 / product_2_order_multiplier)
        best_ask_volume_product_2 = abs(list(order_depth_product_2.sell_orders.values())[0])
        best_ask_avail_units_product_2 = int(best_ask_volume_product_2 /  product_2_order_multiplier)
        mid_price_product_2 = stat.fmean([best_bid_product_2, best_ask_product_2])
        
        best_bid_product_3 = max(order_depth_product_3.buy_orders.keys())
        best_ask_product_3 = min(order_depth_product_3.sell_orders.keys())
        best_bid_volume_product_3 = abs(list(order_depth_product_3.buy_orders.values())[0])
        best_bid_avail_units_product_3 = int(best_bid_volume_product_3 / product_3_order_multiplier)
        best_ask_volume_product_3 = abs(list(order_depth_product_3.sell_orders.values())[0])
        best_ask_avail_units_product_3 = int(best_ask_volume_product_3 /  product_3_order_multiplier)
        mid_price_product_3 = stat.fmean([best_bid_product_3, best_ask_product_3])
        
        best_bid_product_4 = max(order_depth_product_4.buy_orders.keys())
        best_ask_product_4 = min(order_depth_product_4.sell_orders.keys())
        best_bid_volume_product_4 = abs(list(order_depth_product_4.buy_orders.values())[0])
        best_bid_avail_units_product_4 = int(best_bid_volume_product_4 / product_4_order_multiplier)
        best_ask_volume_product_4 = abs(list(order_depth_product_4.sell_orders.values())[0])
        best_ask_avail_units_product_4 = int(best_ask_volume_product_4 /  product_4_order_multiplier)
        mid_price_product_4 = stat.fmean([best_bid_product_4, best_ask_product_4])
        
        current_mid_premium = product_4_order_multiplier * mid_price_product_4 - product_3_order_multiplier * mid_price_product_3 - product_2_order_multiplier * mid_price_product_2 - product_1_order_multiplier * mid_price_product_1
        #current_mid_ratio = product_4_order_multiplier * mid_price_product_4 / ((product_1_order_multiplier * mid_price_product_1) + (product_2_order_multiplier * mid_price_product_2) + (product_3_order_multiplier * mid_price_product_3))
        
        
        z_score_mid = (current_mid_premium - mean_premium) / stdev_premium
        
        
        if z_score_mid > z_score_for_strategy_start:
            
            #current_ratio = best_bid_product_4/ (best_ask_product_1 + (2*best_ask_product_2) + (4*best_ask_product_3))
            current_premium = product_4_order_multiplier * best_bid_product_4 - product_3_order_multiplier * best_ask_product_3 - product_2_order_multiplier * best_ask_product_2 - product_1_order_multiplier * best_ask_product_1 
            #z_score_actual = (current_ratio - mean_ratio) / stdev_ratio
            z_score_actual = (current_premium - mean_premium) / stdev_premium
            
            if z_score_actual > z_score_for_strategy_start :
                desired_position_product_1 = min((round((z_score_actual / (z_score_for_max_orders - z_score_for_strategy_start)) * trade_opportunities) * product_1_order_multiplier), product_1_max_orders)
                desired_position_product_2 = min((round((z_score_actual / (z_score_for_max_orders - z_score_for_strategy_start)) * trade_opportunities) * product_2_order_multiplier), product_2_max_orders)
                desired_position_product_3 = min((round((z_score_actual / (z_score_for_max_orders - z_score_for_strategy_start)) * trade_opportunities) * product_3_order_multiplier), product_3_max_orders)
                desired_position_product_4 = -min((round((z_score_actual / (z_score_for_max_orders - z_score_for_strategy_start)) * trade_opportunities) * product_4_order_multiplier), product_4_max_orders)
                
                if desired_position_product_1 >= product_1_inventory_limit:
                    desired_position_product_1 = product_1_inventory_limit
                    desired_position_product_2 = product_2_inventory_limit
                    desired_position_product_3 = product_3_inventory_limit
                    desired_position_product_4 = - product_4_inventory_limit
            else:
                desired_position_product_1 = 0
                desired_position_product_2 = 0
                desired_position_product_3 = 0
                desired_position_product_4 = 0
            
        elif z_score_mid < - z_score_for_strategy_start:
            #current_ratio = best_ask_product_4/(best_bid_product_1 + (2*best_bid_product_2) + (4*best_bid_product_3))
            current_premium = product_4_order_multiplier * best_bid_product_4 - product_3_order_multiplier * best_ask_product_3 - product_2_order_multiplier * best_ask_product_2 - product_1_order_multiplier * best_ask_product_1
            
            #z_score_actual = (current_ratio - mean_ratio) / stdev_ratio
            z_score_actual = (current_premium - mean_premium) / stdev_premium
            
            if z_score_actual < - z_score_for_strategy_start:
                desired_position_product_1 = min((round((z_score_actual / (z_score_for_max_orders - z_score_for_strategy_start)) * trade_opportunities) * product_1_order_multiplier), product_1_max_orders)
                desired_position_product_2 = min((round((z_score_actual / (z_score_for_max_orders - z_score_for_strategy_start)) * trade_opportunities) * product_2_order_multiplier), product_2_max_orders)
                desired_position_product_3 = min((round((z_score_actual / (z_score_for_max_orders - z_score_for_strategy_start)) * trade_opportunities) * product_3_order_multiplier), product_3_max_orders)
                desired_position_product_4 = -min((round((z_score_actual / (z_score_for_max_orders - z_score_for_strategy_start)) * trade_opportunities) * product_4_order_multiplier), product_4_max_orders)
                
                if desired_position_product_4 >= product_4_inventory_limit:
                    desired_position_product_1 = - product_1_inventory_limit
                    desired_position_product_2 = - product_2_inventory_limit
                    desired_position_product_3 = - product_3_inventory_limit
                    desired_position_product_4 = product_4_inventory_limit
                    

                #print(z_score_mid, z_score_actual)
            else:
                desired_position_product_1 = 0
                desired_position_product_2 = 0
                desired_position_product_3 = 0
                desired_position_product_4 = 0
        else:
            desired_position_product_1 = 0
            desired_position_product_2 = 0
            desired_position_product_3 = 0
            desired_position_product_4 = 0
        
        
        if initial_position_product_1 >= 0 and z_score_mid > 0: 
            if initial_position_product_1 < desired_position_product_1:
                # Add more, long 1,2,3 short 4
                available_units = min(best_ask_avail_units_product_1, best_ask_avail_units_product_2, best_ask_avail_units_product_3, best_bid_avail_units_product_4)
                product_1_orders.append(Order(product_1, best_ask_product_1, min((available_units * product_1_order_multiplier), abs(desired_position_product_1 - initial_position_product_1))))
                product_2_orders.append(Order(product_2, best_ask_product_2, min((available_units * product_2_order_multiplier), abs(desired_position_product_2 - initial_position_product_2))))
                product_3_orders.append(Order(product_3, best_ask_product_3, min((available_units * product_3_order_multiplier), abs(desired_position_product_3 - initial_position_product_3))))
                product_4_orders.append(Order(product_4, best_bid_product_4, -min((available_units * product_4_order_multiplier), abs(desired_position_product_4 - initial_position_product_4))))
                
            
        elif initial_position_product_1 > 0 and z_score_mid <= 0:
            # Neutralize everything we have
            available_units = min(best_bid_avail_units_product_1, best_bid_avail_units_product_2, best_bid_avail_units_product_3, best_ask_avail_units_product_4)
            product_1_orders.append(Order(product_1, best_bid_product_1, - min((available_units * product_1_order_multiplier), abs(desired_position_product_1 - initial_position_product_1))))
            product_2_orders.append(Order(product_2, best_bid_product_2, - min((available_units * product_2_order_multiplier), abs(desired_position_product_2 - initial_position_product_2))))
            product_3_orders.append(Order(product_3, best_bid_product_3, - min((available_units * product_3_order_multiplier), abs(desired_position_product_3 - initial_position_product_3))))
            product_4_orders.append(Order(product_4, best_ask_product_4, min((available_units * product_4_order_multiplier), abs(desired_position_product_4 - initial_position_product_4))))
        
        
        elif initial_position_product_1 < 0 and z_score_mid >= 0: 
            # Neutralize everything we have
            available_units = min(best_ask_avail_units_product_1, best_ask_avail_units_product_2, best_ask_avail_units_product_3, best_bid_avail_units_product_4)
            product_1_orders.append(Order(product_1, best_ask_product_1, min((available_units * product_1_order_multiplier), abs(desired_position_product_1 - initial_position_product_1))))
            product_2_orders.append(Order(product_2, best_ask_product_2, min((available_units * product_2_order_multiplier), abs(desired_position_product_2 - initial_position_product_2))))
            product_3_orders.append(Order(product_3, best_ask_product_3, min((available_units * product_3_order_multiplier), abs(desired_position_product_3 - initial_position_product_3))))
            product_4_orders.append(Order(product_4, best_bid_product_4, -min((available_units * product_4_order_multiplier), abs(desired_position_product_4 - initial_position_product_4))))
            
            
        elif initial_position_product_1 <= 0 and z_score_mid < 0:
            if initial_position_product_1 > desired_position_product_1:
                #add more, short 1,2,3 long 4
                available_units = min(best_bid_avail_units_product_1, best_bid_avail_units_product_2, best_bid_avail_units_product_3, best_ask_avail_units_product_4)
                product_1_orders.append(Order(product_1, best_bid_product_1, - min((available_units * product_1_order_multiplier), abs(desired_position_product_1 - initial_position_product_1))))
                product_2_orders.append(Order(product_2, best_bid_product_2, - min((available_units * product_2_order_multiplier), abs(desired_position_product_2 - initial_position_product_2))))
                product_3_orders.append(Order(product_3, best_bid_product_3, - min((available_units * product_3_order_multiplier), abs(desired_position_product_3 - initial_position_product_3))))
                product_4_orders.append(Order(product_4, best_ask_product_4, min((available_units * product_4_order_multiplier), abs(desired_position_product_4 - initial_position_product_4))))
        
        
        return product_1_orders, product_2_orders, product_3_orders, product_4_orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        
        # Initialize the method output dict as an empty dict
        orders_result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : [], 'CHOCOLATE' : [], 'STRAWBERRIES' : [], 'ROSES' : [], 'GIFT_BASKET' : []}
        conversions_result = 0
        
        # Decode traderData dict from previous iteration 
        decoded_dict = {}
        if state.traderData:
            decoded_dict = jsonpickle.decode(state.traderData)
        
        # Write starfruit cache from traderData
        new_starfruit_cache = []
        if "new_starfruit_cache" in decoded_dict.keys():
            new_starfruit_cache = decoded_dict["new_starfruit_cache"]
        
        # Iterate over all the keys (the available products) contained in the order dephts
        for key, val in state.position.items():
            self.position[key] = val
        
        
        #for product in ['AMETHYSTS', 'STARFRUIT', "ORCHIDS"]:
        #
        #    if product == "AMETHYSTS":
        #        orders = self.compute_orders_amethysts(product, state)
        #        orders_result[product] += orders
        #    
        #    elif product == "STARFRUIT":
        #        new_starfruit_cache, orders = self.compute_orders_starfruit(product, state, new_starfruit_cache)
        #        orders_result[product] += orders
        #    
        #    elif product == "ORCHIDS":
        #        orders, conversions = self.compute_orders_orchids(product, state)
        #        conversions_result += conversions
        #        orders_result[product] += orders
        
        if 'CHOCOLATE' in state.order_depths.keys() and 'STRAWBERRIES' in state.order_depths.keys() and 'ROSES' in state.order_depths.keys() and 'GIFT_BASKET' in state.order_depths.keys():
            strawberry_orders, chocolate_orders, roses_orders, gift_basket_orders = self.compute_orders_basket(state)
            
            orders_result['STRAWBERRIES'] += strawberry_orders
            orders_result['CHOCOLATE'] += chocolate_orders
            orders_result['ROSES'] += roses_orders
            orders_result['GIFT_BASKET'] += gift_basket_orders
        
        
        new_dict = {"new_starfruit_cache": new_starfruit_cache}
        trader_data = jsonpickle.encode(new_dict)
        
        
        logger.flush(state, orders_result, conversions_result, trader_data)
        
        return orders_result, conversions_result, trader_data
