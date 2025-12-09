"""
Simple analysis utilities for anchor quality prediction.
Uses pandas and scipy - no complex abstractions.
"""
import pandas as pd
import numpy as np
from scipy import stats as sp_stats


def compute_anchor_quality(graph, algorithm_func, start_vertex=None) -> pd.DataFrame:
    """
    Run anchor algorithm from each vertex and return quality scores.

    Args:
        graph: NetworkX graph
        algorithm_func: Function that takes (graph, start_vertex) and returns (tour, weight)
        start_vertex: If provided, only test this vertex. Otherwise test all.

    Returns:
        DataFrame with columns: vertex_id, tour_weight, rank, percentile
    """
    results = []
    vertices = [start_vertex] if start_vertex is not None else list(graph.nodes())

    for vertex in vertices:
        tour, weight = algorithm_func(graph, start_vertex=vertex)
        results.append({
            'vertex_id': vertex,
            'tour_weight': weight,
        })

    df = pd.DataFrame(results)
    df['rank'] = df['tour_weight'].rank()
    df['percentile'] = (df['tour_weight'].rank(pct=True) * 100).round(2)
    return df


def correlation_analysis(stats_df: pd.DataFrame, quality_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation between each statistic and anchor quality.

    Args:
        stats_df: DataFrame with vertex_id and edge statistics
        quality_df: DataFrame with vertex_id, tour_weight, percentile

    Returns:
        DataFrame with columns: feature, correlation, p_value, abs_correlation
    """
    merged = pd.merge(stats_df, quality_df, on='vertex_id')

    stat_cols = [c for c in stats_df.columns if c != 'vertex_id']
    results = []

    for col in stat_cols:
        try:
            r, p = sp_stats.pearsonr(merged[col], merged['percentile'])
            results.append({
                'feature': col,
                'correlation': float(r),
                'p_value': float(p),
                'abs_correlation': abs(r)
            })
        except Exception as e:
            print(f"Warning: Could not compute correlation for {col}: {e}")

    return pd.DataFrame(results).sort_values('abs_correlation', ascending=False)


def simple_regression(X: pd.DataFrame, y: pd.Series):
    """
    Train simple linear regression and return coefficients and metrics.

    Args:
        X: Feature matrix (DataFrame)
        y: Target variable (Series)

    Returns:
        Dictionary with model, coefficients, intercept, and performance metrics
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        'model': model,
        'coefficients': dict(zip(X.columns, model.coef_)),
        'intercept': float(model.intercept_),
        'r2_train': float(model.score(X_train, y_train)),
        'r2_test': float(r2_score(y_test, y_pred)),
        'rmse_test': float(np.sqrt(mean_squared_error(y_test, y_pred)))
    }
