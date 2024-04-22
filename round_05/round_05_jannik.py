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

empty_dict = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 'ORCHIDS' : 0, 'STRAWBERRIES' : 0, 'CHOCOLATE' : 0, 'ROSES' : 0, 'GIFT_BASKET' : 0, 'COCONUT' : 0, 'COCONUT_COUPON' : 0}


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
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS' : 100, 'STRAWBERRIES' : 350, 'CHOCOLATE' : 250, 'ROSES' : 60, 'GIFT_BASKET' : 60, 'COCONUT' : 300, 'COCONUT_COUPON' : 600}
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
        if cost > -1.3:
            margin = -1         # sign for us to switch to humidity-based one-directional trading
        elif cost > -2:
            margin = 0.3
        elif cost >= -2.4:
            margin = 0.4
        elif cost < -4.5:
            margin = 2.5
        else:
            margin = - cost - 2
        
        return margin


    def norm_cdf(self, x):
        """Cumulative distribution function for the standard normal distribution."""
        
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


    def black_scholes_call_price(self, S, X, T, r, sigma):
        '''Calculation of black scholes call price and delta'''
        d1 = (math.log(S / X) + (r + (sigma ** 2) / 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        N_d1 = self.norm_cdf(d1)
        N_d2 = self.norm_cdf(d2)
        call_value = S * N_d1 - X * math.exp(-r * T) * N_d2
        return call_value, N_d1


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


    def compute_orders_orchids(self, product, state: TradingState, new_humidity_cache, new_sunlight_cache):

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

        if len(new_humidity_cache) == 200:
            new_humidity_cache.pop(0)
            new_sunlight_cache.pop(0)

        new_humidity_cache.append(humidity_obs)
        new_sunlight_cache.append(sunlight_obs)

        margin = self.calc_orchids_profit_margin(import_tariff_obs, transport_fees_obs)

        if margin < 0 and (new_humidity_cache[1] < new_humidity_cache[0] or new_sunlight_cache[1] < new_sunlight_cache[0]):
            target_inventory = self.position[product] #so that no conversion request will be placed
            orders.append(Order(product, round((bid_price_obs+ask_price_obs)/2 - 2), -position_limit-self.position[product]))

        elif int(round(import_price + margin)) > import_price:
            orders.append(Order(product, int(round(import_price + margin)), - position_limit))
        
        else:
            orders.append(Order(product, int(round(import_price + 0.5)), - position_limit))
        
        orders.append(Order(product, int(round(export_price - 2)), position_limit))
        
        conversions += - self.position[product] + target_inventory

        return orders, conversions, new_humidity_cache, new_sunlight_cache


    def compute_orders_basket(self, state: TradingState):
        '''
        One Basket is made up of: 6 Strawberries, 4 Chocolates, 1 Rose
        '''
        
        #mean_premium = 379.4905
        #stdev_premium = 76.4243809265
        #median_ratio = 1.005414
        
        # calculated with mid prices
        mean_ratio_basket = 1.005397
        stdev_ratio_basket = 0.00109086        
        
        # Standard deviations of ratio difference when we start buying/selling for mean reversion        
        z_score_for_max_orders_product_4 = 0.71
        z_score_for_strategy_start_product_4 = 0.7
        
        product_1 = 'STRAWBERRIES'
        product_2 = 'CHOCOLATE'
        product_3 = 'ROSES'
        product_4 = 'GIFT_BASKET'
        
        product_4_orders: list[Order] = []
        
        product_1_order_multiplier = 6
        product_2_order_multiplier = 4
        product_3_order_multiplier = 1
        product_4_order_multiplier = 1
        
        product_4_inventory_limit = 58
        
        order_depth_product_1: OrderDepth = state.order_depths[product_1]
        order_depth_product_2: OrderDepth = state.order_depths[product_2]
        order_depth_product_3: OrderDepth = state.order_depths[product_3]
        order_depth_product_4: OrderDepth = state.order_depths[product_4]
        
        initial_position_product_4 = state.position.get(product_4, 0)
        
        best_bid_product_1 = max(order_depth_product_1.buy_orders.keys())
        best_ask_product_1 = min(order_depth_product_1.sell_orders.keys())
        mid_price_product_1 = stat.fmean([best_bid_product_1, best_ask_product_1])
        
        best_bid_product_2 = max(order_depth_product_2.buy_orders.keys())
        best_ask_product_2 = min(order_depth_product_2.sell_orders.keys())
        mid_price_product_2 = stat.fmean([best_bid_product_2, best_ask_product_2])
        
        best_bid_product_3 = max(order_depth_product_3.buy_orders.keys())
        best_ask_product_3 = min(order_depth_product_3.sell_orders.keys())
        mid_price_product_3 = stat.fmean([best_bid_product_3, best_ask_product_3])
        
        best_bid_product_4 = max(order_depth_product_4.buy_orders.keys())
        best_ask_product_4 = min(order_depth_product_4.sell_orders.keys())
        mid_price_product_4 = stat.fmean([best_bid_product_4, best_ask_product_4])
        
        
        
        current_mid_ratio = product_4_order_multiplier * mid_price_product_4 / ((product_1_order_multiplier * mid_price_product_1) + (product_2_order_multiplier * mid_price_product_2) + (product_3_order_multiplier * mid_price_product_3))
        z_score_mid = (current_mid_ratio - mean_ratio_basket) / stdev_ratio_basket
        
        if z_score_mid > z_score_for_strategy_start_product_4:
            desired_position_product_4 = -min(round(((z_score_mid - z_score_for_strategy_start_product_4) / (z_score_for_max_orders_product_4 - z_score_for_strategy_start_product_4)) * product_4_inventory_limit), product_4_inventory_limit)
            
        elif z_score_mid < - z_score_for_strategy_start_product_4:
            desired_position_product_4 = min(round(((- z_score_mid - z_score_for_strategy_start_product_4) / (z_score_for_max_orders_product_4 - z_score_for_strategy_start_product_4)) * product_4_inventory_limit), product_4_inventory_limit)
        else:
            desired_position_product_4 = 0
        
        if initial_position_product_4 <= 0 and z_score_mid > 0: 
            if initial_position_product_4 > desired_position_product_4:
                # Add more short 4
                product_4_orders.append(Order(product_4, best_bid_product_4, -abs(desired_position_product_4 - initial_position_product_4)))

        elif initial_position_product_4 < 0 and z_score_mid <= 0:
            # Neutralize everything we have
            product_4_orders.append(Order(product_4, best_ask_product_4, abs(desired_position_product_4 - initial_position_product_4)))

        elif initial_position_product_4 > 0 and z_score_mid >= 0: 
            # Neutralize everything we have
            product_4_orders.append(Order(product_4, best_bid_product_4, -abs(desired_position_product_4 - initial_position_product_4)))
            
        elif initial_position_product_4 >= 0 and z_score_mid < 0:
            if initial_position_product_4 < desired_position_product_4:
                #add more long 4
                product_4_orders.append(Order(product_4, best_ask_product_4, abs(desired_position_product_4 - initial_position_product_4)))
        
        
        return product_4_orders


    def compute_orders_coco(self, state:TradingState):
        
        product_1_orders: list[Order] = []
        product_2_orders: list[Order] = []
        
        product_1 = 'COCONUT'
        product_2 = 'COCONUT_COUPON'
        
        order_depth_product_1: OrderDepth = state.order_depths[product_1]
        order_depth_product_2: OrderDepth = state.order_depths[product_2]
        
        initial_position_product_1 = state.position.get(product_1, 0)
        initial_position_product_2 = state.position.get(product_2, 0)
        
        best_bid_product_1 = max(order_depth_product_1.buy_orders.keys(), default = 0)
        best_ask_product_1 = min(order_depth_product_1.sell_orders.keys(), default = 0)
        
        if best_bid_product_1 != 0 and best_ask_product_1 != 0:
            mid_price_product_1 = stat.fmean([best_bid_product_1, best_ask_product_1])
        else:
            return product_1_orders, product_2_orders
        
        
        best_ask_product_2 = min(order_depth_product_2.sell_orders.keys(), default = 0)
        best_bid_product_2 = max(order_depth_product_2.buy_orders.keys(), default = 0)
        
        if best_bid_product_2 != 0 and best_ask_product_2 != 0:
            mid_price_product_2 = stat.fmean([best_bid_product_2, best_ask_product_2])
        else:
            return product_1_orders, product_2_orders
        
        
        X = 10000                       # Exercise price
        T = 245/365                     # Time until expiration, in years
        r = 0.0                         # Risk-free interest rate
        sigma = 0.19314                 # Expected annualized volatility = given from premium at the first time step
        
        call_price_stdev = 13.468083665890335
        z_score_start = 0.7
        z_score_max_orders = 0.700001
        product_1_inventory_limit = 300
        product_2_inventory_limit = 600
        
        
        black_scholes_call_price, black_scholes_call_delta = self.black_scholes_call_price(mid_price_product_1, X, T, r, sigma)
        
        z_score_call_price = (mid_price_product_2 - black_scholes_call_price) / call_price_stdev
        # if z_score_call_price positive -> short coupon (product_2)
        
        if z_score_call_price > z_score_start:
            # short coupon, long coconut --> coupon is overpriced
            desired_position_product_2 = -min(round(((z_score_call_price - z_score_start) / (z_score_max_orders - z_score_start)) * product_2_inventory_limit), product_2_inventory_limit)
            
        elif z_score_call_price < - z_score_start:
            # long coupon, short coconut --> coupon is underpriced
            desired_position_product_2 = min(round(((- z_score_call_price - z_score_start) / (z_score_max_orders - z_score_start)) * product_2_inventory_limit), product_2_inventory_limit)
        else:
            desired_position_product_2 = 0
        
        desired_position_product_1 = round( - 0.5 * desired_position_product_2)
        
        if initial_position_product_2 <= 0 and z_score_call_price > 0: 
            if initial_position_product_2 > desired_position_product_2:
                # Add more short 2, long 1
                product_1_orders.append(Order(product_1, best_ask_product_1, abs(desired_position_product_1 - initial_position_product_1)))
                product_2_orders.append(Order(product_2, best_bid_product_2, -abs(desired_position_product_2 - initial_position_product_2)))

        elif initial_position_product_2 < 0 and z_score_call_price <= 0:
            # Neutralize everything we have
            product_1_orders.append(Order(product_1, best_bid_product_1, -abs(desired_position_product_1 - initial_position_product_1)))
            product_2_orders.append(Order(product_2, best_ask_product_2, abs(desired_position_product_2 - initial_position_product_2)))

        elif initial_position_product_2 > 0 and z_score_call_price >= 0: 
            # Neutralize everything we have
            product_1_orders.append(Order(product_1, best_ask_product_1, abs(desired_position_product_1 - initial_position_product_1)))
            product_2_orders.append(Order(product_2, best_bid_product_2, -abs(desired_position_product_2 - initial_position_product_2)))
            
        elif initial_position_product_2 >= 0 and z_score_call_price < 0:
            if initial_position_product_2 < desired_position_product_2:
                #add more long 2, short 1
                product_1_orders.append(Order(product_1, best_bid_product_1, -abs(desired_position_product_1 - initial_position_product_1)))
                product_2_orders.append(Order(product_2, best_ask_product_2, abs(desired_position_product_2 - initial_position_product_2)))
        
        
        return product_1_orders, product_2_orders


    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        
        # Initialize the method output dict as an empty dict
        orders_result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : [], 'CHOCOLATE' : [], 'STRAWBERRIES' : [], 'ROSES' : [], 'GIFT_BASKET' : [], 'COCONUT' : [], 'COCONUT_COUPON' : []}
        conversions_result = 0
        
        # Decode traderData dict from previous iteration 
        decoded_dict = {}
        if state.traderData:
            decoded_dict = jsonpickle.decode(state.traderData)
        
        # Write starfruit cache, humidity cache and sunlight cache from traderData
        new_starfruit_cache = []
        if "new_starfruit_cache" in decoded_dict.keys():
            new_starfruit_cache = decoded_dict["new_starfruit_cache"]

        new_humidity_cache = [0]
        if "new_humidity_cache" in decoded_dict.keys():
            new_humidity_cache = decoded_dict["new_humidity_cache"]

        new_sunlight_cache = [0]
        if "new_sunlight_cache" in decoded_dict.keys():
            new_sunlight_cache = decoded_dict["new_sunlight_cache"]
        
        # Iterate over all the keys (the available products) contained in the order dephts
        for key, val in state.position.items():
            self.position[key] = val
        
        
        
        for product in ['AMETHYSTS', 'STARFRUIT', "ORCHIDS", 'COCONUT']:
        
            if product == "AMETHYSTS":
                orders = self.compute_orders_amethysts(product, state)
                orders_result[product] += orders
        
            elif product == "STARFRUIT":
                new_starfruit_cache, orders = self.compute_orders_starfruit(product, state, new_starfruit_cache)
                orders_result[product] += orders
            
            if product == "ORCHIDS":
                orders, conversions, new_humidity_cache, new_sunlight_cache = self.compute_orders_orchids(product, state, new_humidity_cache, new_sunlight_cache)
                conversions_result += conversions
                orders_result[product] += orders
            
            elif product == 'COCONUT':
                coconut_orders, coupon_orders = self.compute_orders_coco(state)
                orders_result['COCONUT'] += coconut_orders
                orders_result['COCONUT_COUPON'] += coupon_orders
        
        if 'CHOCOLATE' in state.order_depths.keys() and 'STRAWBERRIES' in state.order_depths.keys() and 'ROSES' in state.order_depths.keys() and 'GIFT_BASKET' in state.order_depths.keys():
            gift_basket_orders = self.compute_orders_basket(state)
            orders_result['GIFT_BASKET'] += gift_basket_orders
        
        
        new_dict = {"new_starfruit_cache": new_starfruit_cache, "new_humidity_cache": new_humidity_cache, "new_sunlight_cache": new_sunlight_cache}
        trader_data = jsonpickle.encode(new_dict)
        
        
        logger.flush(state, orders_result, conversions_result, trader_data)
        
        return orders_result, conversions_result, trader_data
