#!/usr/bin/env Rscript
# Post-processing

library(sgd)
library(ggplot2)
library(gridExtra)

sgd.list <- list()
for (i in 1:2) {
  # Load all data to plot.
  load(sprintf("out/m-estimation-high-%i.RData", i))
  sgd.list[[i]] <- sgd.theta
}
names(sgd.list) <- c("sgd", "ai-sgd")

pdf("img/huber_high.pdf")
p1 <- plot(sgd.list, data$theta, type="mse-param") +
  geom_hline(yintercept=1.5, color="green")
p2 <- plot(sgd.list, data$theta, type="mse-param-runtime") +
  geom_hline(yintercept=1.5, color="green")
grid.arrange(p1, p2, ncol=2)
dev.off()
