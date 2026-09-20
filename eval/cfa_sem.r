rm(list = ls())

if (!requireNamespace("lavaan",   quietly = TRUE)) install.packages("lavaan")
if (!requireNamespace("semTools", quietly = TRUE)) install.packages("semTools")
library(lavaan) # We used version 0.6-21
library(semTools) # We used version 0.5-7

# Session > Set Working Directory (RStudio)
# e.g., eval/data/cfa/


# ------------------------------------------------------------------------
# Item-factor mapping
# ------------------------------------------------------------------------

# Note SDO-D -> SDOD and SDO-E -> SDOE due to lavaan constraints 
model.map <- list(
  "F" = '
    # Factor loadings for F
    F =~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + 
         X11 + X12 + X13 + X14 + X15 + X16 + X17 + X18 + X19 + X20 + 
         X21 + X22 + X23 + X24 + X25 + X26 + X27 + X28 + X29 + X30 +
         X31 + X32 + X33 + X34 + X35 + X36 + X37 + X38 + X39 + X40 + 
         X41 + X42 + X43 + X44
  ',
  "LAS" = '
    # Factor loadings for LAS
    LAS =~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + 
           X11 + X12 + X13 + X14 + X15 + X16 + X17 + X18
  ',
  "D" = '
    # Factor loadings for D
    D =~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + 
         X11 + X12 + X13 + X14 + X15 + X16 + X17 + X18 + X19 + X20 + 
         X21 + X22 + X23 + X24 + X25 + X26 + X27 + X28 + X29 + X30 +
         X31 + X32 + X33 + X34 + X35 + X36 + X37 + X38 + X39 + X40
  ',
  "A" = '
    # Factor loadings for A
    RIG =~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + 
           X11 + X12 + X13 + X14 + X15 + X16 + X17
    DOG =~ X18 + X19 + X20 + 
           X21 + X22 + X23 + X24 + X25 + X26 + X27 + X28 + X29 + X30 +
           X31 + X32 + X33 + X34 + X35 + X36 + X37 + X38 + X39 + X40 + 
           X41
  ',
  "AA" = '
    # Factor loadings for AA
    AA =~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + 
          X11 + X12 + X13 + X14 + X15 + X16 + X17 + X18 + X19 + X20 + 
          X21 + X22
  ',
  "RWA" = '
    # Factor loadings for RWA
    RWA =~ X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + 
           X11 + X12 + X13 + X14 + X15 + X16 + X17 + X18 + X19 + X20 + 
           X21 + X22 + X23 + X25 + X26 + X27 + X28 + X29 + X30 +
           X31 + X32 + X33
  ',
  "RWA3D" = '
    # Factor loadings for RWA3D
    AGR =~ X1 + X2 + X3 + X4
    SUB =~ X5 + X6 + X7 + X8
    CONV =~ X9 + X10 + X11 + X12
  ',
  "KSA3" = '
    # Factor loadings for KSA3
    AGR =~ X1 + X2 + X3
    SUB =~ X4 + X5 + X6
    CONV =~ X7 + X8 + X9
  ',
  "ACT" = '
    # Factor loadings for ACT
    AUTH =~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + 
            X11 + X12
    CONS =~ X13 + X14 + X15 + X16 + X17 + X18 + X19 + X20 + 
            X21 + X22 + X23 + X24
    TRAD =~ X25 + X26 + X27 + X28 + X29 + X30 +
            X31 + X32 + X33 + X34 + X35 + X36
  ',
  "VSA" = '
    # Factor loadings for VSA
    AUTH =~ X5 + X6
    CONS =~ X1 + X2
    TRAD =~ X3 + X4
  ',
  "ASC" = '
    # Factor loadings for ASC
    AGR =~ X13 + X14 + X15 + X16 + X17 + X18
    SUB =~ X1 + X2 + X3 + X4 + X5 + X6
    CONV =~ X7 + X8 + X9 + X10 + X11 + X12
  ',
  "APC" = '
    # Factor loadings for APC
    APC =~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + 
         X11 + X12 + X13 + X14 + X15 + X16 + X17 + X18 + X19 + X20 + 
         X21 + X22 + X23 + X24 + X25 + X26 + X27 + X28 + X29 + X30 +
         X31 + X32 + X33 + X34 + X35 + X36 + X37 + X38 + X39 + X40 + 
         X41 + X42 + X43 + X44 + X45 + X46
  ',
  "CSM" = '
    # Factor loadings for CSM
    CSM =~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + 
         X11 + X12
  ',
  "DW" = '
    # Factor loadings for DW
    DW =~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10
  ',
  "BDW" = '
    # Factor loadings for BDW
    BDW =~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10
  '
)


# "PI" = '
#     # Factor loadings for PI
#     PI =~ X1 + X2 + X3 + X4 + X5 + X6
#   ',
# "CW" = '
#     # Factor loadings for CW
#     CW =~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10
#   ',
# "SDO7" = '
#     # Factor loadings for SDO7
#     SDOD =~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8
#     SDOE =~ X9 + X10 + X11 + X12 + X13 + X14 + X15 + X16
#   ',
# "PISD" = '
#     # Factor loadings for PISD
#     HARM =~ X1 + X2 + X3 + X4 + X5
#     COOP =~ X6 + X7 + X8 + X9
#     PROG =~ X10 + X11 + X12 + X13
#     REG =~ X14 + X15 + X16 + X17
#     STAB =~ X18 + X19 + X20 + X21
#     JUST =~ X22 + X23 + X24
#     PL =~ X25 + X26 + X27 + X28 + X29
#   ',
# "BFI10" = '
#     # Factor loadings for BFI10
#     EXTRA =~ X1 + X6
#     COMP =~ X2 + X7
#     CONS =~ X3 + X8
#     NEURO =~ X4 + X9
#     OPEN =~ X5 + X10
#   '



# ------------------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------------------

output.dir <- "eval/data/reliability/R2"
dir.create(output.dir, recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------
# Remove zero-variance columns
remove_zero_var <- function(data) {
  zero_var_cols <- !apply(data, 2, function(x) var(x, na.rm = TRUE) > 0)
  if (any(zero_var_cols)) {
    removed_cols <- names(data)[zero_var_cols]
    cat("  Removing", length(removed_cols), "zero-variance columns:", 
        paste(removed_cols, collapse = ", "), "\n")
  }
  data[, !zero_var_cols, drop = FALSE]
}

# Prune model to match available data
prune_model <- function(model.syntax, data) {
  vars.in.data <- colnames(data)
  
  lines <- unlist(strsplit(model.syntax, "\n"))
  lines <- trimws(lines)
  lines <- lines[lines != "" & !grepl("^#", lines)]
  
  out <- c()
  removed_factors <- c()
  
  i <- 1
  while (i <= length(lines)) {
    line <- lines[i]
    
    # If not a factor definition, keep as is
    if (!grepl("=~", line)) {
      out <- c(out, line)
      i <- i + 1
      next
    }
    
    # Collect full factor block (multi-line)
    block <- line
    i <- i + 1
    while (i <= length(lines) && !grepl("=~", lines[i])) {
      block <- paste(block, lines[i])
      i <- i + 1
    }
    
    lhs <- trimws(sub("=~.*", "", block))
    rhs <- sub(".*=~", "", block)
    
    items <- trimws(unlist(strsplit(rhs, "\\+")))
    items_available <- items[items %in% vars.in.data]
    items_missing <- items[!items %in% vars.in.data]
    
    # Keep factor only if ≥2 indicators remain
    if (length(items_available) >= 2) {
      out <- c(out, paste0(lhs, " =~ ", paste(items_available, collapse = " + ")))
      if (length(items_missing) > 0) {
        cat("  Factor", lhs, "- removed items:", paste(items_missing, collapse = ", "), "\n")
      }
    } else {
      removed_factors <- c(removed_factors, lhs)
      cat("  WARNING: Factor", lhs, "dropped (insufficient indicators:", 
          length(items_available), ")\n")
    }
  }
  
  if (length(removed_factors) > 0) {
    cat("  Total factors removed:", paste(removed_factors, collapse = ", "), "\n")
  }
  
  paste(out, collapse = "\n")
}

# compute cfa helper
compute.cfa.sem <- function(model.fit, dataset.label = "", condition = "") {
  if (!lavInspect(model.fit, "converged")) {
    return(data.frame(
      chi.sq = NA,
      df = NA,
      chi.sq.df.ratio = NA,
      RMSEA = NA,
      RMSEA_CI_lower = NA,
      RMSEA_CI_upper = NA,
      SRMR = NA,
      CFI = NA,
      GFI = NA,
      CR_total = NA
    ))
  }
  # lavInspect(model.fit, "cov.lv")
  
  # Report stats such as RMSEA, CFI, and TLI indices.
  # CFI and TLI should be >= 0.95
  # Low RMSEA and SRMR is better
  #summary(model.fit, fit.measures = T, standardized = T, rsquare = T)
  
  chi.sq <- fitMeasures(model.fit, "chisq") # Chi-square
  df <- fitMeasures(model.fit, "df") # Degrees of freedom
  chi.sq.df.ratio <- if (!is.na(df) && df > 0) chi.sq / df else NA # Ratio
  
  rmsea <- fitMeasures(model.fit, "rmsea")
  rmsea.ci.lower <- fitMeasures(model.fit, "rmsea.ci.lower")
  rmsea.ci.upper <- fitMeasures(model.fit, "rmsea.ci.upper")
  srmr <- fitMeasures(model.fit, "srmr")
  cfi <- fitMeasures(model.fit, "cfi")
  gfi <- fitMeasures(model.fit, "gfi")
  
  res <- c(
    chi.sq = chi.sq,
    df = df,
    chi.sq.df.ratio = chi.sq.df.ratio,
    RMSEA = rmsea,
    RMSEA_CI_lower = rmsea.ci.lower,
    RMSEA_CI_upper = rmsea.ci.upper,
    SRMR = srmr,
    CFI = cfi,
    GFI = gfi
  )
  
  # modificationindices(model.fit)
  
  # parameterestimates(model.fit)
  
  # fitMeasures(model.fit)
  
  # Add to appendix
  #covMat <- cov(data)
  #covMat
  
  #corMat <- cor(data)
  #corMat
  
  # semTools, https://journals.sagepub.com/doi/full/10.1177/2515245920951747
  # The reliability() function was deprecated in 2022
  #reliability(model.fit)
  
  # Calculate **composite reliability** from estimated factor-model parameters
  # return.total=TRUE to add a column including a reliability coefficient for the total composite
  # compRel <- compRelSEM(model.fit, return.total=TRUE)
  # cr.values <- setNames(compRel, paste0("CR_", names(compRel)))
  
  # Calculate composite reliability
  cr.values <- tryCatch(
    {
      compRel <- compRelSEM(model.fit, return.total = TRUE)
      setNames(compRel, paste0("CR_", names(compRel)))
    },
    error = function(e) {
      cat("  WARNING: Could not compute CR for", dataset.label, condition, "\n")
      cat("  Error:", e$message, "\n")
      
      # Return NA for each factor + total
      factor_names <- names(lavInspect(model.fit, "cor.lv"))
      setNames(rep(NA, length(factor_names) + 1), 
               c(paste0("CR_", factor_names), "CR_total"))
    }
  )
  
  res <- c(res, cr.values)
  as.data.frame(as.list(res))
}


# ------------------------------------------------------------------------
# Loop over all datasets
# ------------------------------------------------------------------------

dataset.labels <- c("F","LAS","D","A","AA","RWA","RWA3D","KSA3","ACT","VSA", "ASC","APC","CSM","DW","BDW")

for (dataset.label in dataset.labels) {
  
  closed.file <- paste0(dataset.label, "_en_closed_question.csv")
  open.file <- paste0(dataset.label, "_en_open_question.csv")
  
  if (!file.exists(closed.file) || !file.exists(open.file)) {
    cat("Warning: Files not found for", dataset.label, "- skipping\n")
    next
  }
  
  
  data.en.closed <- read.csv(paste0(dataset.label, "_en_closed_question.csv"))
  data.en.open <- read.csv(paste0(dataset.label, "_en_open_question.csv"))
  
  
  data.en.closed <- remove_zero_var(data.en.closed)
  data.en.open   <- remove_zero_var(data.en.open)
  
  # Init model
  if (dataset.label %in% names(model.map)) {
    model <- model.map[[dataset.label]]
  } else {
    stop("Model for the selected dataset.label is not defined in model.map!")
  }
  
  
  model.closed <- prune_model(model, data.en.closed)
  model.open   <- prune_model(model, data.en.open)
  
  
  # ------------------------------------------------------------------------
  # Compute CFA and SEM stats for CLOSED (English only)
  # ------------------------------------------------------------------------
  cfa.sem.closed <- NULL
  
  if (nchar(trimws(model.closed)) > 0) {
    # Missing data already handled, thus drop missing='direct', estimator='MLR'
    # std.lv (standardize latent variables) = TRUE :: setting factor variance to 1, leaving all factor loadings as free parameters
    
    cat("\n--- Fitting CLOSED model ---\n")
    model.fit.closed <- tryCatch(
      {
        cfa(model.closed, data = data.en.closed, std.lv = TRUE)
      },
      error = function(e) {
        cat("ERROR fitting CLOSED model:", e$message, "\n")
        return(NULL)
      }
    )
    
    if (!is.null(model.fit.closed)) {
      cfa.sem.closed <- compute.cfa.sem(model.fit.closed, dataset.label, "CLOSED")
      row.names(cfa.sem.closed) <- NULL
    } else {
      cfa.sem.closed <- NULL
    }
  } else {
    cat("WARNING: No valid CLOSED model after pruning for", dataset.label, "\n")
  }
  
  
  # ------------------------------------------------------------------------
  # Compute reliability for OPEN (English only)
  # ------------------------------------------------------------------------
  cfa.sem.open   <- NULL
  
  if (nchar(trimws(model.open)) > 0) {
    # Missing data already handled, thus drop missing='direct', estimator='MLR'
    # std.lv (standardize latent variables) = TRUE :: setting factor variance to 1, leaving all factor loadings as free parameters
    
    cat("\n--- Fitting OPEN model ---\n")
    model.fit.open <- tryCatch(
      {
        cfa(model.open, data = data.en.open, std.lv = TRUE)
      },
      error = function(e) {
        cat("ERROR fitting OPEN model:", e$message, "\n")
        return(NULL)
      }
    )
    
    if (!is.null(model.fit.open)) {
      cfa.sem.open <- compute.cfa.sem(model.fit.open, dataset.label, "OPEN")
      row.names(cfa.sem.open) <- NULL
    } else {
      cfa.sem.open <- NULL
    }
  } else {
    cat("WARNING: No valid OPEN model after pruning for", dataset.label, "\n")
  }
  
  
  
  # ------------------------------------------------------------------------
  # Display results
  # ------------------------------------------------------------------------
  
  cat("\n=== RESULTS FOR", dataset.label, "===\n")
  
  if (!is.null(cfa.sem.closed)) {
    cat("\n--- ENGLISH CLOSED ---\n")
    print(cfa.sem.closed)
  }
  
  if (!is.null(cfa.sem.open)) {
    cat("\n--- ENGLISH OPEN ---\n")
    print(cfa.sem.open)
  }
  
  
  # ------------------------------------------------------------------------
  # Save results to CSV
  # ------------------------------------------------------------------------
  
  if (!is.null(cfa.sem.closed)) {
    closed.filename <- file.path(output.dir, paste0(dataset.label, "_closed_stats.csv"))
    write.csv(cfa.sem.closed, closed.filename, row.names = FALSE)
    cat("\nClosed results saved to:", closed.filename, "\n")
  }
  
  if (!is.null(cfa.sem.open)) {
    open.filename <- file.path(output.dir, paste0(dataset.label, "_open_stats.csv"))
    write.csv(cfa.sem.open, open.filename, row.names = FALSE)
    cat("Open results saved to:", open.filename, "\n")
  }
}
