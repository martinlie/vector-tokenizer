from pyecharts import options as opts
from pyecharts.charts import Line

def plot_actual_vs_pred(
    df_actual,
    df_pred,
    col_idx: int,
    title: str = "Predicted vs actual",
):
    """
    Plot actual vs predicted time series using pyecharts (static, no animation).

    Parameters
    ----------
    df_actual : pd.DataFrame
        DataFrame with actual values (index used as x-axis)
    df_pred : pd.DataFrame
        DataFrame with predicted values (index used as x-axis)
    col_idx : int
        Column index to plot
    title : str
        Chart title
    """
    col = df_actual.columns[col_idx]

    x_data = [str(x) for x in df_actual.index]
    y_actual = df_actual[col].tolist()
    y_pred = df_pred[col].tolist()

    chart = (
        Line(
            init_opts=opts.InitOpts(
                animation_opts=opts.AnimationOpts(animation=False)
            )
        )
        .add_xaxis(x_data)
        .add_yaxis(
            "Actual",
            y_actual,
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=2),
        )
        .add_yaxis(
            "Predicted",
            y_pred,
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=2),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            toolbox_opts=opts.ToolboxOpts(),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
    )

    return chart