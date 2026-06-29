# Databricks notebook source
# ============================================================
# AI MODEL DRIFT + HALLUCINATION MONITORING STARTER NOTEBOOK
# ============================================================
#
# PURPOSE
# - Computes:
#   1) feature/data drift
#   2) prediction drift
#   3) response quality / hallucination proxy metrics for LLMs or agents
# - Writes results to Delta tables
# - Creates SQL views that are easy to use in a Databricks AI/BI dashboard
#
# ------------------------------------------------------------
# VARIABLES YOU SHOULD CHANGE FOR YOUR SPECIFIC MODEL / AGENT
# ------------------------------------------------------------
#
# CHANGE THESE to point at your specific ML model or AI agent:
#
# CATALOG_NAME         -> Unity Catalog catalog where output tables live
# SCHEMA_NAME          -> schema/database for monitoring tables
# BASELINE_TABLE       -> baseline/reference dataset table for your model
# SCORING_TABLE        -> production inference log table for your model/agent
# OUTPUT_METRICS_TABLE -> output table for aggregate monitoring metrics
# OUTPUT_DETAIL_TABLE  -> output table for per-feature drift details
#
# MODEL_NAME           -> name of the MLflow registered model OR agent label
# MODEL_TYPE           -> "classification", "regression", or "llm_agent"
#
# ID_COL               -> unique request or prediction id
# TIMESTAMP_COL        -> event timestamp column in scoring table
# PREDICTION_COL       -> prediction column for ML models
# LABEL_COL            -> actual label column if available
# PROMPT_COL           -> prompt/user question column for LLM/agent systems
# RESPONSE_COL         -> model response column for LLM/agent systems
# CONTEXT_COL          -> retrieved context / grounding context column for RAG agents
# AGENT_NAME_COL       -> optional column if one table stores many agents
#
# NUMERIC_FEATURES     -> input features to monitor with PSI + KS
# CATEGORICAL_FEATURES -> input features to monitor with categorical PSI
#
# AGENT_FILTER_VALUE   -> set this if your scoring table contains many agents/models
#                         Example: "customer_support_agent_v2"
#                         Set to None to monitor all rows together
#
# TIME_GRAIN           -> "day" or "hour"
#
# ------------------------------------------------------------
# DEPLOYMENT STEPS IN DATABRICKS (READ THESE COMMENTS FIRST)
# ------------------------------------------------------------
#
# STEP 1:
# - Create or identify two Delta tables:
#   A) BASELINE_TABLE: historical "good" reference data
#   B) SCORING_TABLE: live or recent inference logs
#
# STEP 2:
# - Ensure SCORING_TABLE contains at least:
#   request id, timestamp, model/agent identifier, prediction or response
#
# STEP 3:
# - For LLM / agent monitoring, also log:
#   prompt, model response, retrieved context, optional ground truth / expected answer
#
# STEP 4:
# - Open Databricks Workspace -> New -> Notebook
# - Attach a cluster that can read Unity Catalog tables
#
# STEP 5:
# - Paste this code into the notebook
# - Update all variables in the CONFIG section below
#
# STEP 6:
# - Run the notebook once to create the metrics tables and SQL views
#
# STEP 7:
# - Create a Databricks Workflow job from this notebook
# - Schedule it every hour/day depending on your inference volume
#
# STEP 8:
# - Go to Databricks SQL / AI-BI Dashboards
# - Build dashboard visualizations from the created SQL views
#
# STEP 9:
# - Optionally add Databricks SQL alerts on high drift or hallucination rate
#
# STEP 10:
# - Publish the dashboard and add a refresh schedule
#
# NOTE:
# - Databricks dashboards are typically created in the UI from SQL queries/views.
# - This notebook prepares the data layer for that dashboard.
#
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql import types as T
import math

# ============================================================
# CONFIG SECTION -- CHANGE THESE VALUES
# ============================================================

CATALOG_NAME = "main"                     # CHANGE ME
SCHEMA_NAME = "ai_monitoring"            # CHANGE ME

BASELINE_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.baseline_inference_reference"   # CHANGE ME
SCORING_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.production_inference_logs"       # CHANGE ME

OUTPUT_METRICS_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.model_monitoring_metrics"
OUTPUT_DETAIL_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.model_monitoring_feature_detail"

MODEL_NAME = "my_model_or_agent"         # CHANGE ME
MODEL_TYPE = "llm_agent"                 # CHANGE ME: classification | regression | llm_agent

ID_COL = "request_id"                    # CHANGE ME
TIMESTAMP_COL = "event_ts"               # CHANGE ME
PREDICTION_COL = "prediction"            # CHANGE ME for classical ML
LABEL_COL = "label"                      # CHANGE ME if actual label exists

PROMPT_COL = "prompt"                    # CHANGE ME for LLM/agent
RESPONSE_COL = "response"                # CHANGE ME for LLM/agent
CONTEXT_COL = "retrieved_context"        # CHANGE ME for RAG/agent, or set None
AGENT_NAME_COL = "agent_name"            # CHANGE ME, or set None if not used
AGENT_FILTER_VALUE = None                # CHANGE ME e.g. "claims_agent_v4"

NUMERIC_FEATURES = [
    "feature_1",
    "feature_2",
    "feature_3"
]                                        # CHANGE ME

CATEGORICAL_FEATURES = [
    "feature_cat_1",
    "feature_cat_2"
]                                        # CHANGE ME

TIME_GRAIN = "day"                       # CHANGE ME: day | hour
BASELINE_LOOKBACK_DAYS = 30
SCORING_LOOKBACK_DAYS = 7

# Thresholds for dashboard coloring / alerting
PSI_WARN = 0.10
PSI_CRIT = 0.25
HALLUCINATION_WARN = 0.05
HALLUCINATION_CRIT = 0.10

# ============================================================
# SETUP
# ============================================================

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")

baseline_df = spark.table(BASELINE_TABLE)
scoring_df = spark.table(SCORING_TABLE)

if AGENT_NAME_COL and AGENT_FILTER_VALUE is not None:
    scoring_df = scoring_df.filter(F.col(AGENT_NAME_COL) == F.lit(AGENT_FILTER_VALUE))
    baseline_df = baseline_df.filter(F.col(AGENT_NAME_COL) == F.lit(AGENT_FILTER_VALUE))

if TIME_GRAIN == "hour":
    scoring_df = scoring_df.withColumn("metric_time", F.date_trunc("hour", F.col(TIMESTAMP_COL)))
else:
    scoring_df = scoring_df.withColumn("metric_time", F.to_date(F.col(TIMESTAMP_COL)))

# Optional recent windows
baseline_df = baseline_df.filter(F.col(TIMESTAMP_COL) >= F.date_sub(F.current_timestamp(), BASELINE_LOOKBACK_DAYS))
scoring_df = scoring_df.filter(F.col(TIMESTAMP_COL) >= F.date_sub(F.current_timestamp(), SCORING_LOOKBACK_DAYS))

# ============================================================
# HELPERS
# ============================================================

def safe_log(x):
    return math.log(x) if x > 0 else 0.0

def psi_from_distributions(expected, actual):
    eps = 1e-6
    keys = set(expected.keys()).union(actual.keys())
    total = 0.0
    for k in keys:
        e = max(expected.get(k, 0.0), eps)
        a = max(actual.get(k, 0.0), eps)
        total += (a - e) * safe_log(a / e)
    return float(total)

def approx_contains_context(response, context):
    if response is None or context is None:
        return 0
    response_tokens = set([t.strip(".,!?;:()[]{}\"'").lower() for t in response.split() if len(t) > 3])
    context_tokens = set([t.strip(".,!?;:()[]{}\"'").lower() for t in context.split() if len(t) > 3])
    if len(response_tokens) == 0:
        return 0
    overlap = len(response_tokens.intersection(context_tokens)) / max(len(response_tokens), 1)
    return 1 if overlap >= 0.20 else 0

approx_contains_context_udf = F.udf(approx_contains_context, T.IntegerType())

# ============================================================
# NUMERIC FEATURE DRIFT
# - Uses decile bins from baseline and computes PSI
# ============================================================

feature_detail_rows = []

for feature in NUMERIC_FEATURES:
    base_nonnull = baseline_df.select(F.col(feature).cast("double").alias(feature)).dropna()
    score_nonnull = scoring_df.select("metric_time", F.col(feature).cast("double").alias(feature)).dropna()

    quantiles = base_nonnull.approxQuantile(feature, [i / 10.0 for i in range(11)], 0.001)
    quantiles = sorted(list(set(quantiles)))

    if len(quantiles) < 3:
        continue

    splits = quantiles

    def bucket_expr(col_name, splits):
        expr = None
        for i in range(len(splits) - 1):
            lower = splits[i]
            upper = splits[i + 1]
            cond = ((F.col(col_name) >= F.lit(lower)) & (F.col(col_name) <= F.lit(upper))) if i == len(splits) - 2 else ((F.col(col_name) >= F.lit(lower)) & (F.col(col_name) < F.lit(upper)))
            if expr is None:
                expr = F.when(cond, F.lit(f"bin_{i+1}"))
            else:
                expr = expr.when(cond, F.lit(f"bin_{i+1}"))
        return expr.otherwise(F.lit("unknown"))

    base_binned = base_nonnull.withColumn("bucket", bucket_expr(feature, splits))
    score_binned = score_nonnull.withColumn("bucket", bucket_expr(feature, splits))

    base_dist = {
        row["bucket"]: row["pct"] for row in
        base_binned.groupBy("bucket").count()
        .withColumn("pct", F.col("count") / F.sum("count").over())
        .select("bucket", "pct").collect()
    }

    score_dist_df = (
        score_binned.groupBy("metric_time", "bucket").count()
        .withColumn("group_total", F.sum("count").over(Window.partitionBy("metric_time")))
        .withColumn("pct", F.col("count") / F.col("group_total"))
        .select("metric_time", "bucket", "pct")
    )

    times = [r["metric_time"] for r in score_dist_df.select("metric_time").distinct().collect()]

    for mt in times:
        dist = {
            r["bucket"]: r["pct"] for r in
            score_dist_df.filter(F.col("metric_time") == F.lit(mt)).select("bucket", "pct").collect()
        }
        psi_val = psi_from_distributions(base_dist, dist)
        feature_detail_rows.append((MODEL_NAME, str(mt), feature, "numeric", float(psi_val)))

# ============================================================
# CATEGORICAL FEATURE DRIFT
# - Computes PSI over category frequencies
# ============================================================

for feature in CATEGORICAL_FEATURES:
    base_cat = baseline_df.select(F.coalesce(F.col(feature).cast("string"), F.lit("NULL")).alias("bucket"))
    score_cat = scoring_df.select("metric_time", F.coalesce(F.col(feature).cast("string"), F.lit("NULL")).alias("bucket"))

    base_counts = base_cat.groupBy("bucket").count().cache()
    base_total = base_counts.agg(F.sum("count").alias("total")).first()["total"]
    base_dist = {r["bucket"]: r["count"] / base_total for r in base_counts.collect()}

    score_counts = score_cat.groupBy("metric_time", "bucket").count()
    totals = score_counts.groupBy("metric_time").agg(F.sum("count").alias("total"))
    score_pct = score_counts.join(totals, on="metric_time").withColumn("pct", F.col("count") / F.col("total"))

    times = [r["metric_time"] for r in score_pct.select("metric_time").distinct().collect()]

    for mt in times:
        dist = {
            r["bucket"]: r["pct"] for r in
            score_pct.filter(F.col("metric_time") == F.lit(mt)).select("bucket", "pct").collect()
        }
        psi_val = psi_from_distributions(base_dist, dist)
        feature_detail_rows.append((MODEL_NAME, str(mt), feature, "categorical", float(psi_val)))

# ============================================================
# QUALITY / HALLUCINATION PROXY METRICS FOR AGENTS / LLMs
# - This is a practical proxy, not a perfect truth detector
# - Best results come when you log retrieved context or have labels
# ============================================================

quality_df = scoring_df

if MODEL_TYPE == "llm_agent":
    quality_df = quality_df.withColumn(
        "response_length",
        F.length(F.col(RESPONSE_COL))
    )

    # Proxy 1: groundedness against retrieved context
    # CHANGE CONTEXT_COL if your RAG system stores grounding documents elsewhere
    if CONTEXT_COL is not None:
        quality_df = quality_df.withColumn(
            "is_grounded_proxy",
            approx_contains_context_udf(F.col(RESPONSE_COL), F.col(CONTEXT_COL))
        )
        quality_df = quality_df.withColumn(
            "hallucination_proxy",
            F.when(F.col("is_grounded_proxy") == 1, F.lit(0)).otherwise(F.lit(1))
        )
    else:
        quality_df = quality_df.withColumn("is_grounded_proxy", F.lit(None).cast("int"))
        quality_df = quality_df.withColumn("hallucination_proxy", F.lit(None).cast("int"))

    # Proxy 2: unsupported certainty language
    certainty_regex = r"(?i)\\b(always|never|guaranteed|definitely|certainly|proven|without a doubt)\\b"
    quality_df = quality_df.withColumn(
        "overconfident_language_flag",
        F.when(F.col(RESPONSE_COL).rlike(certainty_regex), F.lit(1)).otherwise(F.lit(0))
    )

    quality_metrics = (
        quality_df.groupBy("metric_time")
        .agg(
            F.count("*").alias("request_count"),
            F.avg("response_length").alias("avg_response_length"),
            F.avg("hallucination_proxy").alias("hallucination_rate_proxy"),
            F.avg("overconfident_language_flag").alias("overconfidence_rate")
        )
    )
else:
    quality_metrics = (
        quality_df.groupBy("metric_time")
        .agg(F.count("*").alias("request_count"))
        .withColumn("avg_response_length", F.lit(None).cast("double"))
        .withColumn("hallucination_rate_proxy", F.lit(None).cast("double"))
        .withColumn("overconfidence_rate", F.lit(None).cast("double"))
    )

# ============================================================
# PREDICTION DRIFT / PERFORMANCE METRICS
# ============================================================

if MODEL_TYPE == "classification":
    pred_metrics = (
        scoring_df.groupBy("metric_time")
        .agg(
            F.count("*").alias("prediction_count"),
            F.avg(F.when(F.col(PREDICTION_COL) == F.col(LABEL_COL), 1.0).otherwise(0.0)).alias("accuracy")
        )
    )
elif MODEL_TYPE == "regression":
    pred_metrics = (
        scoring_df.groupBy("metric_time")
        .agg(
            F.count("*").alias("prediction_count"),
            F.avg(F.abs(F.col(PREDICTION_COL) - F.col(LABEL_COL))).alias("mae")
        )
    )
else:
    pred_metrics = (
        scoring_df.groupBy("metric_time")
        .agg(F.count("*").alias("prediction_count"))
    )

# ============================================================
# FEATURE DETAIL TABLE
# ============================================================

feature_detail_schema = T.StructType([
    T.StructField("model_name", T.StringType(), True),
    T.StructField("metric_time", T.StringType(), True),
    T.StructField("feature_name", T.StringType(), True),
    T.StructField("feature_type", T.StringType(), True),
    T.StructField("psi", T.DoubleType(), True),
])

feature_detail_df = spark.createDataFrame(feature_detail_rows, schema=feature_detail_schema)

feature_detail_df = (
    feature_detail_df
    .withColumn("metric_time", F.to_timestamp("metric_time"))
    .withColumn(
        "drift_status",
        F.when(F.col("psi") >= PSI_CRIT, F.lit("critical"))
         .when(F.col("psi") >= PSI_WARN, F.lit("warning"))
         .otherwise(F.lit("normal"))
    )
)

feature_detail_df.write.format("delta").mode("overwrite").saveAsTable(OUTPUT_DETAIL_TABLE)

# ============================================================
# AGGREGATE METRICS TABLE
# ============================================================

drift_summary = (
    feature_detail_df.groupBy("metric_time")
    .agg(
        F.avg("psi").alias("avg_feature_psi"),
        F.max("psi").alias("max_feature_psi"),
        F.sum(F.when(F.col("drift_status") == "critical", 1).otherwise(0)).alias("critical_feature_count"),
        F.sum(F.when(F.col("drift_status") == "warning", 1).otherwise(0)).alias("warning_feature_count")
    )
)

metrics_df = (
    drift_summary.join(quality_metrics, on="metric_time", how="full")
                 .join(pred_metrics, on="metric_time", how="full")
                 .withColumn("model_name", F.lit(MODEL_NAME))
                 .withColumn(
                     "overall_status",
                     F.when((F.col("max_feature_psi") >= PSI_CRIT) | (F.col("hallucination_rate_proxy") >= HALLUCINATION_CRIT), F.lit("critical"))
                      .when((F.col("max_feature_psi") >= PSI_WARN) | (F.col("hallucination_rate_proxy") >= HALLUCINATION_WARN), F.lit("warning"))
                      .otherwise(F.lit("normal"))
                 )
                 .select(
                     "model_name",
                     "metric_time",
                     "overall_status",
                     "avg_feature_psi",
                     "max_feature_psi",
                     "critical_feature_count",
                     "warning_feature_count",
                     "request_count",
                     "prediction_count",
                     "avg_response_length",
                     "hallucination_rate_proxy",
                     "overconfidence_rate",
                     *([c for c in ["accuracy", "mae"] if c in pred_metrics.columns])
                 )
)

metrics_df.write.format("delta").mode("overwrite").saveAsTable(OUTPUT_METRICS_TABLE)

# ============================================================
# CREATE SQL VIEWS FOR DATABRICKS DASHBOARD
# ============================================================

# These views are what you should point your Databricks dashboard at.

spark.sql(f"""
CREATE OR REPLACE VIEW {CATALOG_NAME}.{SCHEMA_NAME}.vw_model_monitoring_summary AS
SELECT *
FROM {OUTPUT_METRICS_TABLE}
ORDER BY metric_time DESC
""")

spark.sql(f"""
CREATE OR REPLACE VIEW {CATALOG_NAME}.{SCHEMA_NAME}.vw_model_monitoring_feature_drift AS
SELECT *
FROM {OUTPUT_DETAIL_TABLE}
ORDER BY metric_time DESC, psi DESC
""")

# Optional compact latest status view
spark.sql(f"""
CREATE OR REPLACE VIEW {CATALOG_NAME}.{SCHEMA_NAME}.vw_model_monitoring_latest AS
SELECT *
FROM {OUTPUT_METRICS_TABLE}
QUALIFY ROW_NUMBER() OVER (PARTITION BY model_name ORDER BY metric_time DESC) = 1
""")

# ============================================================
# DASHBOARD BUILD NOTES
# ============================================================
#
# In Databricks SQL / AI-BI Dashboards, create these visuals:
#
# 1) KPI tiles from vw_model_monitoring_latest:
#    - overall_status
#    - max_feature_psi
#    - hallucination_rate_proxy
#    - accuracy or mae
#
# 2) Time series from vw_model_monitoring_summary:
#    - metric_time vs max_feature_psi
#    - metric_time vs hallucination_rate_proxy
#    - metric_time vs request_count
#
# 3) Bar chart from vw_model_monitoring_feature_drift:
#    - latest metric_time
#    - x = feature_name
#    - y = psi
#
# 4) Table:
#    - feature_name, psi, drift_status, metric_time
#
# ============================================================
# JOB DEPLOYMENT NOTES
# ============================================================
#
# Step-by-step deployment in Databricks:
#
# A. Notebook Deployment
# 1. Save this notebook in Workspace, for example:
#    /Shared/monitoring/ai_model_drift_hallucination_monitor
# 2. Attach a compute cluster and test run it.
#
# B. Production Scheduling
# 1. Go to Workflows -> Jobs -> Create Job
# 2. Add this notebook as a task
# 3. Set parameters by editing the config section in notebook
#    OR refactor config into Databricks widgets if you want runtime parameters
# 4. Set schedule, for example every 1 hour
# 5. Set alerts for task failure
#
# C. Dashboard Deployment
# 1. Go to SQL -> Dashboards
# 2. Create new AI/BI dashboard
# 3. Add datasets using:
#    - {CATALOG_NAME}.{SCHEMA_NAME}.vw_model_monitoring_summary
#    - {CATALOG_NAME}.{SCHEMA_NAME}.vw_model_monitoring_feature_drift
#    - {CATALOG_NAME}.{SCHEMA_NAME}.vw_model_monitoring_latest
# 4. Build KPI cards, line charts, bar charts, and tables
# 5. Publish dashboard
# 6. Add refresh schedule using a SQL warehouse
#
# D. Optional SQL Alerts
# 1. Create alert on hallucination_rate_proxy >= threshold
# 2. Create alert on max_feature_psi >= threshold
# 3. Route to email / webhook depending on your workspace setup
#
# ============================================================