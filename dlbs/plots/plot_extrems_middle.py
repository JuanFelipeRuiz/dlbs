from dlbs.plots.image_grid import plot_instances_grid
import pandas as pd


def plot_extremes_and_middle(df,column,group_titles, instances=False, yolo_annotation=True):
    """Plot the extremes and middle values  depending on the given column values
    except the test split images
    """
    

    subset = df.copy()

    # Optionally exclude a split (like test)
    subset = subset[subset['split'] != 'test']

    # Sort and calculate mean
    subset = subset.sort_values(column, ascending=True)
    mean_val = subset[column].mean()

    # Select samples
    low_df = subset.head(10).assign(group='low')
    mid_df = subset.iloc[(subset[column] - mean_val).abs().argsort()[:10]].assign(group='mid')
    high_df = subset.tail(10).assign(group='high')

    # Combine
    selected_df = pd.concat([low_df, mid_df, high_df], ignore_index=True)

    # order by the column
    selected_df = selected_df.sort_values(column, ascending=True)

    ncols = 5
    fig = plot_instances_grid(
        selected_df,
        show_instances=False,
        ncols=ncols,
        figsize=(20, 16)
    )

    for i, ax in enumerate(fig.axes):
    # Re-enable axis frame so labels can show

        # Determine if this axis is in the first column
        if i % ncols == 0:  
            # Figure out which group (low/mid/high) this row belongs to
            row_idx = i // ncols
            title_idx = row_idx // 2
            if title_idx < len(group_titles):
                ax.set_ylabel(
                    group_titles[title_idx],
                    rotation=90,
                    fontsize=12,
                    ha='center',
                    va='center',
                    labelpad=10
                )


    return fig