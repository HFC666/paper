library("miceadds")
library("rstan")
library("bayesplot")
library("bridgesampling")
library("matrixStats")
library("lubridate")


source.all("include/")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

RA.data <- read.csv("data/S16.csv")
NAs <- which(is.na(RA.data$activity))
for (i in 1:length(NAs)) {
  ind_lower <- NAs[i] - 1
  ind_upper <- NAs[i] + 1
  RA.data$activity[NAs[i]] <- mean(RA.data$activity[c(ind_lower, ind_upper)])
}

RA.data$activity.sq <- sqrt(RA.data$activity)
attach(RA.data)
obs <- activity.sq
N <- nrow(RA.data)

ticks <- seq(from = 3, to = N, by = 96)
labels <- paste(hour(as.character(time))[ticks], "00", sep = ":")
labels[labels == "4:00"] = "04:00"
par(mfrow = c(1, 1))
plot(activity, type = "p", pch = 20, cex = 0.4, xaxt = "n", 
     xlab = "", col = "gray48", xlim = c(-5, N+5), xaxs='i', 
     ylab = "Physical Activity")
axis(side = 1, ticks, labels, cex.axis = 0.7, las = 2, tck=-0.04)
i <- c(0, 2*144, 4*144, 6*144, 8*144)
rect(3 + i, -2, 147 + i, -1, col="black") 

ticks <- seq(from = 3, to = N, by = 96)
labels <- paste(hour(as.character(time))[ticks], "00", sep = ":")
labels[labels == "4:00"] = "04:00"
par(mfrow = c(1, 1))
plot(sqrt(activity), type = "p", pch = 20, cex = 0.4, xaxt = "n", 
     xlab = "", col = "gray48", xlim = c(-5, N+5), xaxs='i', 
     ylab = "Sqrt (Physical Activity)", ylim = c(-0.06, 7.2))
axis(side = 1, ticks, labels, cex.axis = 0.7, las = 2, tck=-0.04)
i <- c(0, 2*144, 4*144, 6*144, 8*144)
rect(3 + i, -0.3, 147 + i, -0.2, col="black") 


## Comparable Priors - Set Up
K <- 3
alpha_0 <- matrix(NA, K, K)
a_0 <- numeric(K)
b_0 <- numeric(K)

## Comparable Priors - State 1 : Inactive (IA)
state <- 1
mean.target <- 7.5*12
var.target <- (0.5*12)^2

dir.parms <- optim(par = rep(5, 2),
                   fn = function(par) {
                     geom.dwell.error(exp(par[1]), exp(par[2]),
                                      mean.target, var.target)
                   })

est <- exp(dir.parms$par)
alpha_0[state, ] <- c(est[1], est[2]/2, est[2]/2)

gamma.parms <- optim(par = rep(5, 2),
                     fn = function(par) {
                       nb.dwell.error(exp(par[1]), exp(par[2]),
                                      mean.target, var.target)
                     })
est <- exp(gamma.parms$par)
a_0[state] <- est[1]
b_0[state] <- est[2]

geom.dwell.info(alpha_0[state, state], sum(alpha_0[state, -state]))
nb.dwell.info(a_0[state], b_0[state])


## Comparable Priors - State 2 : Moderately Active (MA)
state <- 2
mean.target <- 2*12
var.target <- (1.5*12)^2

dir.parms <- optim(par = c(5, 2),
                   fn = function(par) {
                     geom.dwell.error(exp(par[1]), exp(par[2]),
                                      mean.target, var.target)
                   })

est <- exp(dir.parms$par)
alpha_0[state, ] <- c(est[2]/5, est[1], est[2]*(4/5))

gamma.parms <- optim(par = c(5, 5), fn = function(par) {
  nb.dwell.error(exp(par[1]), exp(par[2]), mean.target, var.target)
})
est <- exp(gamma.parms$par)
a_0[state] <- est[1]
b_0[state] <- est[2]

geom.dwell.info(alpha_0[state, state], sum(alpha_0[state, -state]))
nb.dwell.info(a_0[state], b_0[state])

## Comparable Priors -- State 3: Highly Active (HA)
state <- 3
mean.target <- 2*12
var.target <- (1.5*12)^2

dir.parms <- optim(par = rep(5, 2),
                   fn = function(par) {
                     geom.dwell.error(exp(par[1]), exp(par[2]),
                                      mean.target, var.target)
                   })

est <- exp(dir.parms$par)
alpha_0[state, ] <- c(est[2]/5, est[2]*(4/5),est[1])

gamma.parms <- optim(par = rep(5, 2), 
                     fn = function(par) {
                       nb.dwell.error(exp(par[1]), exp(par[2]), 
                                      mean.target, var.target)
                     })
est <- exp(gamma.parms$par)
a_0[state] <- est[1]; 
b_0[state] <- est[2];


geom.dwell.info(alpha_0[state, state], sum(alpha_0[state, -state]))
nb.dwell.info(a_0[state], b_0[state])

## Bayesian HSMM Approx - Negative Binomial Dwell
# prior
K <- 3  # 状态的数量
m <- c(150, 10, 10)  # dwell threshold

# hyperparms
sigma_0 <- 2
mu_0 <- rep(mean(obs), K)
a_0; b_0
v_0 <- matrix(c(alpha_0[1, 2:3], alpha_0[2, c(2,3)], alpha_0[3, 2:3]),
              nrow = K, ncol = K-1, byrow = TRUE)

# Testing for sparsity
if((K /sum(m)) < 0.1) {
  path.stan <- "stan/bayesHSMMapprox_GaussEmis_NegBinomDur_sparse.stan"
} else {
  path.stan <- "stan/bayesHSMMapprox_GaussEmis_NegBinomDur.stan"
}
HSMM.data <- list(N = length(obs), K = K, y = obs, m = m, mu_0 = mu_0,
                  sigma_0 = sigma_0, a_0_lambda = a_0, b_0_lambda = b_0,
                  a_0_phi = 2, b_0_phi = 2, alpha_0 = v_0)
HSMM.stan <- stan(file = path.stan, data = HSMM.data,
                  warmup = 1000, iter = (1+5)*1000, chains = 1, cores = 1)
