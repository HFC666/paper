library(rstan)
library("miceadds")
library("bayesplot")
library("bridgesampling")
library("matrixStats")
library("lubridate")


source.all("include/")
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())


## 产生数据
N <- 200  # 生成数据的长度
params <- list()  # 参数
params$K <- 3  # 状态的数量
params$lambda <- c(20, 30, 20)  # poisson duration rate
params$mu <- c(5, 14, 30)    # emission parameter - mean
params$sigma <- c(1, 1, 1)   # emission parameter - sd
params$gamma <- matrix(c(0, 0.3, 0.7,
                         0.2, 0, 0.8,
                         0.1, 0.9, 0), params$K, params$K, byrow = T)
params$delta <- c(1.0, 0.0, 0.0)  # 初始概率

set.seed(111)
simul <- gauss.HMM.generate_sample(N, params)
obs <- simul$x
state <- simul$state
plot(obs, col=state, cex=0.7, pch=20, type="o")


## HSMM approx - Expectation/Conditional Maximization(ECM)
K <- 3  # n states
m <- rep(5, K)  # dwell threshold
lambda.0 <- rep(5, K)  # lambda initial value
params.0 <- HSMM.init(obs, K, lambda.0)  # emission initial values
HSMM.ECM.fit <- HSMM.ECM(K=K ,m=m, obs = obs, parms_init = params.0,niter = 1e2)
cat("mllk:", HSMM.ECM.fit$mllk, "AIC:", HSMM.ECM.fit$AIC, "BIC:", HSMM.ECM.fit$BIC, "\n")


## Bayesian HSMM
K <- 3
m <- rep(5, K)  # dwell threshold
lambda.0 <- rep(10, K)  # lambda initial value MCMC
data.stan <- list(
  N = length(obs), K=K, y=obs,
  m = m, mu_0 = rep(mean(obs), K),
  sigma_0 = 2, a_0 = rep(0.01, K), b_0 = rep(0.01, K),
  alpha_0 = matrix(1, nrow = K, ncol = K-1)
)

if ((K / sum(m) < 0.1)) {
  stan_path <- "stan/bayesHSMMapprox_GaussEmis_PoissDur.stan"
} else {
  stan_path <- "stan/bayesHSMMapprox_GaussEmis_PoissDur_sparse.stan"
}

HSMM.stan <- stan(file = stan_path, data = data.stan,
                  init = function(){HSMM.init.stan(K, obs, lambda.0)},
                  warmup = 1000, chains = 1, iter = (1+5)*1000, cores = 1,
                  control = list(adapt_delta=0.99, stepsize=0.01, max_treedepth=20))

sims <- extract(HSMM.stan)
lambda_sims <- sims$lambda
mu_sims <- sims$mu
sigma_sims <- sqrt(sims$sigma2)
gamma_sims <- get.gamma_sims(sims$gamma)


mcmc_trace(as.array(HSMM.stan), regex_pars = "gamma")
mcmc_trace(as.array(HSMM.stan), regex_pars = "mu")
mcmc_trace(as.array(HSMM.stan), regex_pars = "sigma2")
mcmc_trace(as.array(HSMM.stan), regex_pars = "lambda")
HSMM.transitions.hist(gamma_sims)


HSMM.performance <- HSMM.stan.performance(sims, obs, m)


K <- 3
HMM.data <- list(N = length(obs), K = K, y = obs, 
                 mu_0 = rep(mean(obs), K), sigma_0 = 2, 
                 alpha_0 = matrix(c(rep(1, K*K)), nrow = K, ncol = K, byrow = TRUE))
HMM.stan <- stan(file = "stan/bayesHMM_GaussEmis.stan", data = HMM.data,
                 warmup = 1000, iter=(1+5)*1000, chains=1, cores=1)
print(HMM.stan, probs = c(0.05, 0.95))


HSMM.bridge <- bridge_sampler(HSMM.stan)
HMM.bridge <- bridge_sampler(HMM.stan)


HSMM.marg_llk <- HSMM.bridge$logml
HMM.marg_llk <- HMM.bridge$logml
exp(HSMM.marg_llk - HMM.marg_llk)
