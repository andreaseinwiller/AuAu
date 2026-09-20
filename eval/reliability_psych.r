rm(list = ls())

if (!requireNamespace("psych", quietly = TRUE)) install.packages("psych")
library(psych) # We used version 2.5.6


# Session > Set Working Directory (RStudio)
# e.g., eval/data/cfa/


# ------------------------------------------------------------------------
# Item-factor mapping
# ------------------------------------------------------------------------

model.map <- list(
  "F" = list(
    "F" = paste0("X", 1:44)
  ),
  "LAS" = list(
    "LAS" = paste0("X", 1:18)
  ),
  "D" = list(
    "D" = paste0("X", 1:40)
  ),
  "A" = list(
    "RIG" = paste0("X", 1:17),
    "DOG" = paste0("X", 18:41),
    "joint.factor" = paste0("X", 1:41)
  ),
  "AA" = list(
    "AA" = paste0("X", 1:22)
  ),
  "RWA" = list(
    # 1, 2, 24, and 34 were filtered during preprocessing.
    "RWA" = c("X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20","X21","X22","X23","X25","X26","X27","X28","X29","X30","X31","X32","X33")
  ),
  "RWA3D" = list(
    "AGR" = paste0("X", 1:4),
    "SUB" = paste0("X", 5:8),
    "CONV" = paste0("X", 9:12),
    "joint.factor" = paste0("X", 1:12)
  ),
  "KSA3" = list(
    "AGR" = paste0("X", 1:3),
    "SUB" = paste0("X", 4:6),
    "CONV" = paste0("X", 7:9),
    "joint.factor" = paste0("X", 1:9)
  ),
  "ACT" = list(
    "AUTH" = paste0("X", 1:12),
    "CONS" = paste0("X", 13:24),
    "TRAD" = paste0("X", 25:36),
    "joint.factor" = paste0("X", 1:36)
  ),
  "VSA" = list(
    "AUTH" = c("X5", "X6"),
    "CONS" = c("X1", "X2"),
    "TRAD" = c("X3", "X4"),
    "joint.factor" = paste0("X", 1:6)
  ),
  "ASC" = list(
    "AGR" = paste0("X", 13:18),
    "SUB" = paste0("X", 1:6),
    "CONV" = paste0("X", 7:12),
    "joint.factor" = paste0("X", 1:18)
  ),
  "APC" = list(
    "APC" = paste0("X", 1:46)
  ),
  "CSM" = list(
    "CSM" = paste0("X", 1:12)
  ),
  "DW" = list(
    "DW" = paste0("X", 1:10)
  ),
  "BDW" = list(
    "BDW" = paste0("X", 1:10)
  )
)





# "PI" = list(
#   "PI" = paste0("X", 1:6)
# ),
# "CW" = list(
#   "CW" = paste0("X", 1:10)
# ),
# "SDO7" = list(
#   "SDO-D" = paste0("X", 1:8),
#   "SDO-E" = paste0("X", 9:16),
#   "joint.factor" = paste0("X", 1:16)
# ),
# "PISD" = list(
#   "HARM" = paste0("X", 1:5),
#   "COOP" = paste0("X", 6:9),
#   "PROG" = paste0("X", 10:13),
#   "REG" = paste0("X", 14:17),
#   "STAB" = paste0("X", 18:21),
#   "JUST" = paste0("X", 22:24),
#   "PL" = paste0("X", 25:29),
#   "joint.factor" = paste0("X", 1:29)
# ),
# "BFI10" = list(
#   "EXTRA" = c("X1", "X6"),
#   "COMP" = c("X2", "X7"),
#   "CONS" = c("X3", "X8"),
#   "NEURO" = c("X4", "X9"),
#   "OPEN" = c("X5", "X10"),
#   "joint.factor" = paste0("X", 1:10)
# )




# ------------------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------------------

output.dir <- "eval/data/reliability/R1/"
dir.create(output.dir, recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------------------
# Loop over all datasets
# ------------------------------------------------------------------------

for (dataset.label in names(model.map)) {
  cat("\n========================================\n")
  cat("Processing dataset:", dataset.label, "\n")
  cat("========================================\n")
  
  closed.file <- paste0(dataset.label, "_en_closed_question.csv")
  open.file <- paste0(dataset.label, "_en_open_question.csv")
  
  if (!file.exists(closed.file) || !file.exists(open.file)) {
    cat("Warning: Files not found for", dataset.label, "- skipping\n")
    next
  }
  
  # Init factor.map
  if (dataset.label %in% names(model.map)) {
    factor.map <- model.map[[dataset.label]]
  } else {
    cat("Model for the selected dataset.label is not defined in model.map!")
    next
  }

  data.en.closed <- read.csv(paste0(dataset.label, "_en_closed_question.csv"))
  data.en.open <- read.csv(paste0(dataset.label, "_en_open_question.csv"))
  
  
  # ------------------------------------------------------------------------
  # Reliability comp helper function
  # ------------------------------------------------------------------------
  
  compute.reliability <- function(df, items) {
    df <- df[, items, drop = FALSE]
    
    # 1. Compute Cronbach's alpha
    ca <- suppressWarnings(alpha(df))
    # 2. McDonald’s omega
    mcdo <- tryCatch({
      suppressWarnings(omega(df, plot = FALSE, cor = "pearson"))
    }, error = function(e) {
      list(omega_h = NA, omega.tot = NA)
    })
    
    data.frame(
      alpha = ca$total$raw_alpha,
      a.l.ci95 = ca$feldt$lower.ci$raw_alpha, # lower 95% ci bound for alpha
      a.u.ci95 = ca$feldt$upper.ci$raw_alpha, # upper 95% ci bound for alpha
      omega_h = mcdo$omega_h, # Low values for subset scales (with low number of items) are expected
      omega.tot = mcdo$omega.tot
    )
  }
  
  # ------------------------------------------------------------------------
  # Compute reliability for CLOSED (English only)
  # ------------------------------------------------------------------------
  
  results.closed <- list()
  
  for (factor.name in names(factor.map)) {
    items <- factor.map[[factor.name]]
    
    rel <- compute.reliability(data.en.closed, items)
    if (!is.null(rel)) {
      results.closed[[length(results.closed) + 1]] <- cbind(
        factor = factor.name,
        rel
      )
    }
  }
  
  reliability.closed <- do.call(rbind, results.closed)
  row.names(reliability.closed) <- NULL
  
  # ------------------------------------------------------------------------
  # Compute reliability for OPEN (English only)
  # ------------------------------------------------------------------------
  
  results.open <- list()
  
  for (factor.name in names(factor.map)) {
    items <- factor.map[[factor.name]]
    
    rel <- compute.reliability(data.en.open, items)
    if (!is.null(rel)) {
      results.open[[length(results.open) + 1]] <- cbind(
        factor = factor.name,
        rel
      )
    }
  }
  
  reliability.open <- do.call(rbind, results.open)
  row.names(reliability.open) <- NULL
  
  
  # ------------------------------------------------------------------------
  # Display results
  # ------------------------------------------------------------------------
  
  cat("\n=== ENGLISH CLOSED ===\n")
  print(reliability.closed)
  
  cat("\n=== ENGLISH OPEN ===\n")
  print(reliability.open)
  
  
  # ------------------------------------------------------------------------
  # Save results to CSV
  # ------------------------------------------------------------------------
  
  closed.filename <- file.path(output.dir, paste0(dataset.label, "_closed_stats.csv"))
  write.csv(reliability.closed, closed.filename, row.names = FALSE)
  cat("\nClosed results saved to:", closed.filename, "\n")
  
  open.filename <- file.path(output.dir, paste0(dataset.label, "_open_stats.csv"))
  write.csv(reliability.open, open.filename, row.names = FALSE)
  cat("Open results saved to:", open.filename, "\n")
}

