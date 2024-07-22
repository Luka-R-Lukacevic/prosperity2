# Aruba Capital
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->


In this repo we present ideas and code for the second IMC Prosperity competition, hosted in 2024. Our team, Aruba Capital, finished 22nd globally out of more then 2800 active competitors, placing us in the top 1%. In this write up we will focus on the algorithmic coding rounds (and not the manual challenges).

## Team members

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/jaJann312">
          <img src="https://avatars.githubusercontent.com/u/49684458?v=4" width="100px;" alt="Jannik Proff"/>
          <br /><sub><b>Jannik Proff</b></sub></a>
        <br /><sub><a href="https://www.linkedin.com/in/jproff/" title="LinkedIn">🔗 LinkedIn</a></sub>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/JGGrosse">
          <img src="https://avatars.githubusercontent.com/u/142249387?v=4" width="100px;" alt="Janek Große"/>
          <br /><sub><b>Janek Große</b></sub></a>
        <br /><sub><a href="https://www.linkedin.com/in/janek-grosse/" title="LinkedIn">🔗 LinkedIn</a></sub>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/Luka-R-Lukacevic">
          <img src="https://avatars.githubusercontent.com/u/125273166?v=4" width="100px;" alt="Luka Lukačević"/>
          <br /><sub><b>Luka Lukačević</b></sub></a>
        <br /><sub><a href="https://www.linkedin.com/in/luka-lukačević/" title="LinkedIn">🔗 LinkedIn</a></sub>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/PaulHenrik">
          <img src="https://avatars.githubusercontent.com/u/19336571?v=4" width="100px;" alt="Paul Heilmann"/>
          <br /><sub><b>Paul Heilmann</b></sub></a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/tinotil">
          <img src="https://avatars.githubusercontent.com/u/35002593?v=4" width="100px;" alt="Constantin Schott"/>
          <br /><sub><b>Constantin Schott</b></sub></a>
      </td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## 🐚 What is prosperity?

"Prosperity is a 15-day long trading challenge happening somewhere in a near - utopian - future. You’re in control of an island in an archipelago and your goal is to bring your island to prosperity. You do so by earning as many SeaShells as possible; the main currency in the archipelago. The more SeaShells you earn, the more your island will prosper. 

During your 15 days on the island, your trading abilities will be tested through a variety of trading challenges. It’s up to you to develop a successful trading strategy. You will be working on a Python script to handle algorithmic trades on your behalf. Every round you will also be confronted with a manual trading challenge. Your success depends on both these algorithmic and manual trades." (IMC description)

There were over 2800 active teams, tasked with algorithmically trading various products, such as amethysts, starfruit, orchids, coconuts, and more, with the goal of maximizing seashells: the underlying currency of our island.

In round 1, we began trading amethysts and starfruit. With each subsequent round, additional products were introduced. Our trading algorithm was assessed at the end of each round by comparing its performance to that of bot participants in the marketplace. We could attempt to predict the bots' behavior using historical data. Our PNL from this evaluation was then compared to that of other teams.

Aside from the main focus on algorithmic trading, the competition included manual trading challenges in each round. These challenges varied significantly, but manual trading ultimately contributed only a small fraction to our overall PNL.

For more details on the algorithmic trading environment and additional context about the competition, please refer to the [Prosperity 2 Wiki](https://imc-prosperity.notion.site/Prosperity-2-Wiki-fe650c0292ae4cdb94714a3f5aa74c85).

## Organization

This repository contains all of our code–including internal tools, research notebooks, raw data and backtesting logs, and all versions of our algorithmic trader. The repository is organized by round.

<details>
<summary><h2>Tools </h2></summary>

Instead of building our tools in-house, we decided to leverage the open-source wizardry of  [Jasper van Merle](https://github.com/jmerle). His tools provided the foundation we needed, allowing us to tailor our focus on other areas of development. We utilized his two main tools: a backtester and a visualiser.

### Backtester

We realized we needed a comprehensive backtesting environment very early on. After going after that ourselves with not a lot of success, fortunately, Jasper van Merle's [backtester](https://github.com/jmerle/imc-prosperity-2-backtester) was released to take in historical data and a trading algorithm. With the historical data, it would construct all the necessary information (replicating the actual trading environment perfectly) that our trading algorithm needed, input it into our trading algorithm, and receive the orders that our algorithm would send. Then, it would match those orders to the orderbook to generate trades. After running, the backtester would create a log file in the exact same format as the Prosperity website, that the visualiser was then able to visualise.


![Backtested PNL](https://github.com/Luka-R-Lukacevic/prosperity2/blob/main/Images/Backtester%20Image.jpeg)


### Visualiser

Jasper van Merle's [visualizer](https://jmerle.github.io/imc-prosperity-2-visualizer/?/visualizer) visualiser was an immense tool for us that provided a powerful and flexible way to analyze our trading data, helping us to identify and rectify issues, and ultimately improve our trading strategies. 


![Visualiser in Action](https://github.com/Luka-R-Lukacevic/prosperity2/blob/main/Images/Visualiser%20Image.png)


</details>
<details>
<summary><h2>Round 1️⃣</h2></summary>

In round 1, we had access to two symbols to trade: amethysts and starfruit. 

### Amethysts
Amethysts were fairly simple, as the fair price clearly never deviated from 10,000. As such, we wrote our algorithm to trade against bids above 10,000 and asks below 10,000. Besides taking orders, our algorithm also would market-make, placing bids and asks below and above 10,000, respectively.

### Starfruit

Starfruits were an asset with an orderbook limit of 20 (as were amethysts). Here the price fluctuated much more though, usually up to hundreds of seashells. Also, notice that the spread is pretty wide (around 6-7 consistently, which is much more then for the other products).

![Starfruit](https://github.com/Luka-R-Lukacevic/prosperity2/blob/main/Images/Starfruit.jpeg)

This opened up the opportunity for market making, provided one had a good price estimate. After trying lots of things we concluded that there was no additional information in knowing the whole price history (in comparison to just the current orderbook). In mathematical terms you could say the prices followed a discrete-time Markov process. As a small digression, in the Black-Scholes (BS) model the assumed SDE that leads to the formula also necessitates the Markov property, we will see more of the BS formula later in round 4.

Still, while we concluded there was basically no point in taking complicated history into account for the fair price, we still had the problem that we could not just use the mid-price (average of highest bid and lowest ask) as our price estimate, since if there are good trades in the orderbook for us, then these will necessarily be either exceptionally high bids or exceptionally low asks. One could then use past history to get a better fair estimate. This works fine, but something else worked even better. Essentially one could see that in the orderbook there were usually bids and ask that had high volume and were around 6-7 apart and then some small deviant orders (in real markets these would be called micro-noise).

![Starfruit micro noise](https://github.com/Luka-R-Lukacevic/prosperity2/blob/main/Images/Starfruit%20orderbook.jpeg)

A nifty solution came from last years second place, the [Stanford Cardinals](https://github.com/ShubhamAnandJain/IMC-Prosperity-2023-Stanford-Cardinal/tree/main). The mid-price estimate is simply the mid-price by lowest-bid and highest-ask. In this market this eliminates the micro-noise, allowing us to pick off the bad orders.

After the first round we were in the 70s but with relatively small distance to the lead.


</details>

<details>
<summary><h2>Round 2️⃣</h2></summary>
  
### Orchids
In round 2, orchids were introduced. While they could be traded on our home market just like amethysts and starfruits, there was another market (the "south archipelago") where we were able to trade orchids. Trades executed on that foreign market however were subject to import/export tariffs as well as shipping costs, all of which changed over time. We were only allowed to do trades with the south archipelago however that would bring our position closer to zero (i.e. whenever we were long we could export, whenever we were short we could import). They also gave us historical graphs for sunlight and humidity levels which we were told would influence the orchids' growth/availability. They even gave us exact ranges of values of sunlight/humidity for which the growth would be perfect and how it deviates outside of these ranges. This was however very unclear as they for example failed to provide a unit for the sunlight and gave contradictory information on it in the discord channel.

We started by investigating whether there was any relation between the orchids' price and sunlight/humidity and found a loose correlation. Position-taking on that however gave us only minimal profits. This fact combined with the very vague definition led us to believe that the key in this round was not to be found in the weather (which turned out to be probably true as we have not heard of any successful team using sunlight/humidity at all).

Next up, we tried simple market making on our home island which failed miserably. Experimenting a bit made us realize however that - even though the order book was notoriously empty this round - there was a big buyer that would regularly take all our sell orders. Realizing that we were actually paid the tariffs when importing from the south archipelago, we saw that cross-exchange market making was the way to go: We would sell on our home market to the big buyer and immediately import the same quantity (often 100 which was the position limit) to cancel our position. That way, we were able to trade an incredible volume and our profits on the website skyrocketed to 80-100k. Now that we had the basic strategy (which ironically was implemented in two lines of code), we just had to do some fine-tuning of the only parameter in our code: What was the perfect price for our sell orders on the home market that would maximize volume*profit per share. We realized that this depended heavily on the import tariffs and the shipping costs and also that our strategy was only so profitable because of the incredibly high import tariffs (which we were paid) during the last day. In the end we came up with an algorithm that dynamically adapts the sell price to the current tariffs/shipping costs.

After this round we were ranked 48th. We probably could have gotten a higher rank had we done the optimization of the sell price at times of semi-low import tariffs more carefully. For low import tariffs we realized that there was no profit to be made from this strategy which led us to implement sunlight/humidity based position taking in that case in subsequent rounds.

</details>
<details>
<summary><h2>Round 3️⃣</h2></summary>
Gift baskets, chocolate, roses, and strawberries were introduced in round 3, where a gift basket consisted of 4 chocolate bars, 6 strawberries, and a single rose. This round, we mainly traded spreads, which we defined as `basket - synthetic`, with `synthetic` being the sum of the price of all products in a basket.

### Spread
In this round, we quickly converged on two hypotheses. The first hypothesis was that the synthetic would be leading baskets or vice versa, where changes in the price of one would lead to later changes in the price of the other.  Our second hypothesis was that the spread might simply just be mean reverting. We observed that the price of the spread–which theoretically should be 0–hovered around some fixed value, which we could trade around. We looked into leading/lagging relationships between the synthetic and the basket, but this wasn't very fruitful, so we then investigated the spread price. 

![newplot (1)](https://github.com/ericcccsliu/imc-prosperity-2/assets/62641231/6e56f911-8f7c-484c-8dab-32a1603ad2de)

Looking at the spread, we found that the price oscillated around ~370 across all three days of our historical data. Thus, we could profitably trade a mean-reverting strategy, buying spreads (going long baskets and short synthetic) when the spread price was below average, and selling spreads when the price was above. We tried various different ways to parameterize this trade. Due to our position limits, which were relatively small (about 2x the volume on the book at any instant), and the relatively small number of mean-reverting trading opportunities, we realized that timing the trade correctly was critical, and could result in a large amount of additional pnl. 

We tried various approaches in parameterizing this trade. A simple, first-pass strategy was just to set hardcoded prices at which to trade–for example, trading only when the spread deviated from the average value by a certain amount. We backtested to optimize these hardcoded thresholds, and our best parameters netted us ~120k in projected pnl. However, with this strategy, we noticed that we could lose out on a lot of pnl if the spread price reverted before touching our threshold. To remedy this, we could set our thresholds closer, but then we'd also lose pnl from trading before the spread price reached a local max/min. 

Therefore, we developed a more adaptive algorithm for spreads. We traded on a modified z-score, using a hardcoded mean and a rolling window standard deviation, with the window set relatively small. The idea behind this was that there should be a fundamental reason behind the mean of spread (think the price of the basket itself), but the volatility each day would be less predictable. Then, we thresholded the z-score, selling spreads when our z-score went above a certain value and buying when the z-score dropped below. By using a small window for our rolling standard deviation, we'd see our z-score spike when the standard deviation drastically dropped–and this would often happen right as the price started reverting, allowing us to trade closer to local minima/maxima. This idea bumped our backtest pnl up to ~135k. 


![newplot (2)](https://github.com/ericcccsliu/imc-prosperity-2/assets/62641231/0db11d51-8916-4ed5-83f6-82faeb846267)
<p align="center">
  <em>a plot of spread prices and our modified z-score, as well as z-score thresholds (in green) to trade at</em>
</p>

After results from this round were released, we found that our actual pnl had a significant amount of slippage compared to our backtests–we made only 111k seashells from our algo. Nevertheless, we got a bit lucky–all the teams ahead of us in this round seemed to overfit significantly more, as we were ranked #2 overall.

</details>
<details>
<summary><h2>Round 4️⃣</h2></summary>
  
### Coconuts/coconut coupon :coconut:
Coconuts and coconut coupons were introduced in round 4. Coconut coupons were the 10,000 strike call option on coconuts, with a time to expiry of 250 days. The price of coconuts hovered around 10,000, so this option was near-the-money. 

This round was fairly simple. Using Black-Scholes, we calculated the implied volatility of the option, and once we plotted this out, it became clear that the implied vol oscillated around a value of ~16%. We implemented a mean reverting strategy similar to round 3, and calculated the delta of the coconut coupons at each time in order to hedge with coconuts and gain pure exposure to vol. However, the delta was around 0.53 while the position limits for coconuts/coconut coupons were 300/600, respectively. This meant that we couldn't be fully hedged when holding 600 coupons (we would be holding 18 delta). Since the coupon was far away from expiry (thus, gamma didn't matter as much) and holding delta with vega was still positive ev (but higher var), we ran the variance in hopes of making more from our exposure to vol. 

![newplot (3)](https://github.com/ericcccsliu/imc-prosperity-2/assets/62641231/21fc47f7-727f-48a4-bf4e-b9b9c5fd25a1)

While holding this variance worked out in our backtests, we experienced a fair amount of slippage in our submission–we got unlucky and lost money from our delta exposure. In retrospect, not fully delta hedging might not have been  a smart move–we were already second place and thus should've went for lower var to try and keep the lead. Our algorithm in this round made only 145k, dropping us down to a terrifying 26th place. However, in the results of this round, we saw Puerto Vallarta leap ahead with a whopping profit of 1.2 *million* seashells. We knew we could catch up and end up well within the top 10 if only we could figure out what they did. 
</details>
<details>
<summary><h2>Round 5️⃣</h2></summary>

Our leading hypothesis in trying to replicate Puerto Vallarta's profits were that they must've found some way to predict the future–profits on the order of 1.2 million could reasonably match up with a successful stat. arb strategy across multiple symbols. So, we started blasting away with linear regressions on lagged and synchronous returns across all symbols and all days of our data, with the hypothesis that symbols from different days could have correlations that we'd previously missed. However, we didn't find anything particularly interesting here–starfruits seemed to have a bit of lagged predictive power in all other symbols, but this couldn't explain 1.2 million in additional profits.

As a last-ditch attempt in this front, we recalled that last year's competition (which we read about in [Stanford Cardinal's awesome writeup](https://github.com/ShubhamAnandJain/IMC-Prosperity-2023-Stanford-Cardinal)) had many similarities to this competition–especially in the first round, where the symbols we traded basically sounded the exact same. So, we went and sourced last year's data from public GitHub repositories, and performed a linear regression from returns in each of last year's symbols to returns in each symbol of this year. The results we found were surprising: diving gear returns from last year's competition, with a multiplier of ~3, was almost a perfect predictor of roses, with a $R^2$ of 0.99. Additionally, coconuts from last year was a perfect predictor of coconuts from this year, with a beta of 1.25 and an $R^2$ of 0.99.

![image](https://github.com/ericcccsliu/imc-prosperity-2/assets/62641231/64b2c041-b14d-47eb-9c25-df8cb6fcc290)

These discoveries were quite silly, but nonetheless, our goal was to maximize pnl, and as the data from last year was publically available on the internet, we felt like this was still fair game. The rest of our efforts in this competition centered around maximizing the value we could extract from the market with our new knowledge. We believed that many other teams might find these same relationships, and therefore optimization was key.

As a first pass, we simply bought/sold coconuts and roses when our predicted price rose/fell (beyond some threshold to account for spread costs) over a certain number of future iterations. While this worked spectacularly (in comparison to our pnl from literally all previous rounds), we thought we could do better. Indeed, with the data from last year, we had all local maxima/minima, and thus we could theoretically time our trades perfectly and extract max. value. 

To do this systematically across the three symbols we wanted to trade (roses, coconuts, and gift baskets, due to their natural correlation with roses), we developed a dynamic programming algorithm. Our algorithm took many factors into account–costs of crossing spread, the volume we could take at iteration (the volume on the orderbook), and our volume limits.

The motivation behind the complexity of our dp algorithm was the fact that, at each iteration, we couldn't necessarily achieve our full desired position–therefore, we needed a state for each potential position that we could feasibly achieve. A simple example of this is to imagine a product going through the following prices: 
$$8 \rightarrow 7 \rightarrow 12 \rightarrow 10$$
With a position limit of 2, and with sufficient volume on the orderbook, the optimal trades would be: sell 2 -> buy 4 -> sell 4, with a pnl of 16. Now imagine if you could only buy/sell 2 shares at each iteration. Then, the optimal solution would change–you'd want to buy 2 -> buy 2 -> sell 2, with an overall pnl of 14. 


</details>

For the open-source tools we want to again give credit to [Jasper van Merle](https://github.com/jmerle). For this write up we followed the outline of the excellent report by the second place finish of [linear utility](https://github.com/ericcccsliu/imc-prosperity-2). Some of the ideas were the ones featured from the [Cardinals](https://github.com/ShubhamAnandJain/IMC-Prosperity-2023-Stanford-Cardinal/tree/main).
