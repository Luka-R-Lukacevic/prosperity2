path_prices_neg_1 <-"prices_round_2_day_-1.csv"
path_prices_0 <-"prices_round_2_day_0.csv"
path_prices_pos_1 <-"prices_round_2_day_1.csv"

prices_neg_1 <-read.csv(path_prices_neg_1, sep=";")
prices_0 <-read.csv(path_prices_0, sep=";")
prices_pos_1 <-read.csv(path_prices_pos_1, sep=";")

#bind into 2 dfs 
prices <- rbind(prices_neg_1, prices_0, prices_pos_1)


y <- prices$ORCHIDS
sunl <- prices$SUNLIGHT
hum <- prices$HUMIDITY
transport <- prices$TRANSPORT_FEES
import <- prices$IMPORT_TARIFF
export <- prices$EXPORT_TARIFF
n <- length(y)

prod_sunl <- function(sl){
  if(sl < 2916){
    return(-0.04*(2916-sl)/69.4)
  }
  return(0)
}

prod_hum <- function(hm){
  if(hm > 80){
    return(-0.02*(hm-80)/5)
  }
  if(hm < 60){
    return(0.02*(hm-60)/5)
  }
  return(0)
}

for (i in 2:n) {
  production[i] = 1 + prod_sunl(sunl[i]) + prod_hum(hum[i])
}

plot(production, type = 'l')

n <- length(y)
offset <- as.numeric(10)
y_shift <- y[offset:n]
length(y_shift)
model0 <- lm(y_shift~production[0:(-offset+1)]+sunl[0:(-offset+1)]+hum[0:(-offset+1)]+transport[0:(-offset+1)]+export[0:(-offset+1)]+import[0:(-offset+1)])
summary(model0)