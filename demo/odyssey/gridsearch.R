gridsearch <- function(X_train, y_train, X_test, y_test) {
  # Returns: best hyperparameter in grid for each method according to which has
  # the minimum error.
  methods <- c("sgd", "implicit", "asgd", "ai-sgd")
  out <- rep(NA, length(methods))
  names(out) <- methods
  for (i in methods) {
    # Outline grid of lr.control values for each method.
    methods <- list(i, i, i, i, i)
    lrs <- list("one-dim", "one-dim", "one-dim", "one-dim", "one-dim")
    lr.controls <- list(100, 10, 1, 0.1, 0.01)
    lambda2s <- list(1e-6, 1e-6, 1e-6, 1e-6, 1e-6)
    np <- list(2, 2, 2, 2, 2)
    names <- as.character(lr.controls)
    dataset <- NULL
    ylim <- NULL
    # Run experiment.
    temp <- run_exp(methods, names, lrs, lr.controls, lambda2s, np,
                   X_train, y_train, X_test, y_test,
                   plot=F)
    preds <- temp$preds
    ys <- y_test
    # Taken from plot.error.
    dat <- data.frame()
    count <- 1
    for (pred in preds) {
      error <- 1 - colSums(pred$pred == ys[[count]]) / nrow(pred$pred)
      pos <- colMeans(pred$pos)
      temp_dat <- data.frame(error=error, pos=pos)
      temp_dat[["npass"]] <- temp_dat$pos/max(temp_dat$pos) * np[[count]]
      temp_dat[["label"]] <- as.factor(names[[count]])
      dat <- rbind(dat, temp_dat)
      count <- count + 1
    }
    out[i] <- as.numeric(as.character(dat$label[which.min(dat$error)]))
  }
  return(out)
}
