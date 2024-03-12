
import itertools
import streamlit
import pandas
import plotly_express as px
from plotly.subplots import make_subplots

__version__ = '1.5'

streamlit.set_page_config(layout="wide")

# User input - File upload - starting point
uploaded_file = streamlit.sidebar.file_uploader("Choose a file to start")

# Define colors to use for cluster colors
cluster_colors = ["#989794", "#813D86", "#F6EA4C", "#FD973C", "#FF3C2F", "#04ff00", "#997b1a", "#870e0e", "#9a05f7", "#509690", "#6f85f2", "#fc0303", "#fc03b6", "#8a3a01", "#fa6c07", "#c4a7a7", "#fcfc03", "#fa6161", "#025750", "#7a5e04", "#66024a", "#ad73d1", "#01188c", "#032dff", "#c5cefc", "#fab9b9", "#fc954c", "#1e5e1e", "#703d62", "#acfcf6", "#fcc203", "#a6f7a6", "#e6c2fc", "#00fcec", "#4d1570", "#694125", "#016b01"]

if uploaded_file is not None:
    data_frame = pandas.read_csv(uploaded_file, sep='\t')

    # Sort data frame relative to cluster names (in order to get legends sorted as they will be in the order they are added)
    data_frame = data_frame.sort_values(by='Cluster', key=lambda col: col.map(lambda x: int(x.split('_')[1])))

    column_names = data_frame.columns
    vaf_columns = [name for name in column_names if (name.startswith('VAF') and not name.startswith('VAF_CCF')) ]
    pyclone_ccf_columns = [name for name in column_names if name.startswith('pyclone')]
    vaf_ccf_columns = [name for name in column_names if name.startswith('CCF')]
    minor_cn_columns = [name for name in column_names if name.startswith('min')]
    major_cn_columns = [name for name in column_names if name.startswith('maj')]
    ref_counts_columns = [name for name in column_names if name.startswith('ref')]
    alt_counts_columns = [name for name in column_names if name.startswith('alt')]
    sample_names = [name.split('_')[1] for name in vaf_columns]

    # Combine sample names

    # Since we do not know the interrelation of samples - i.e. which is primary tumor and which is metastasis etc. we make
    # no attempt to organize combinations in terms of which sample name is first/last in combinations. As this, however,
    # will effect what axis each sample goes to, implement the ability to switch axes on the plots later in the script.

    pairwise_sample_combinations = itertools.combinations(sample_names, 2)

    display_combinations = {'{}_vs_{}'.format(combination[0], combination[1]): list(combination) for combination in pairwise_sample_combinations}
    number_of_plots = len(display_combinations)

    if number_of_plots > 1:

        # Add the choice of multiplot to the samples dropdown
        display_combinations['MultiPlot'] = 'MultiPlot'

        # User input - Which samples to plot
        sample_combination = display_combinations[streamlit.sidebar.selectbox('Please select which samples to compare', display_combinations.keys())]
    else:
        sample_combination = list(display_combinations.values())[0]

    # Data filtering
    data_filtering = streamlit.sidebar.expander('Data filters')

    # User input
    min_vaf = data_filtering.slider('Minimal MAF', min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    # User input - Which data type to plot
    data_type = streamlit.sidebar.radio('Select data type', ('VAF', 'pyclone_CCF', 'VAF_CCF'))

    visual_appearance = streamlit.sidebar.expander('Edit visual appearance')

    # User input - dot size
    dot_size = visual_appearance.selectbox('Dot size...', range(5, 21), index=3)

    # User input - display/hide dot periphery line
    display_dot_periphery_line = visual_appearance.checkbox('Toggle dot edge-lines', value=False)

    if display_dot_periphery_line:
        marker = {
            'size': dot_size,
            'line': {
                'width': 2,
                'color': 'DarkSlateGrey'
            }
        }
    else:
        marker = {
            'size': dot_size,
        }

    # Define ranges for axes (toggle between range for VAF plot or CCF plot)
    x_y_ranges = ([-0.05, 1.05], [-0.1, 2.1 ])
    range_x = x_y_ranges[0] if data_type == 'VAF' else x_y_ranges[1]
    range_y = x_y_ranges[0] if data_type == 'VAF' else x_y_ranges[1]

    # Multiplot
    if sample_combination == 'MultiPlot':

        # User input
        grid_columns = visual_appearance.slider('Number of grid columns', min_value=2, max_value=8, value=3, step=1)

        # User input
        inter_space = visual_appearance.slider('Space between plots', min_value=0.05, max_value=0.5, value=0.2, step=0.05)

        grid_rows = int(number_of_plots / grid_columns) + 1 if number_of_plots % grid_columns else int(number_of_plots / grid_columns)

        subplot_size = visual_appearance.slider('Sub-plot size', min_value=100, max_value=800, value=400, step=100)

        subplot_titles = list(display_combinations.keys())[:-1]

        # Define plot settings for axis flipping
        plot_settings = streamlit.sidebar.expander('Edit plots')
        plot_settings.write('Flip axes')
        flip_combination_axes = {}
        for title in subplot_titles:
            flip_combination_axes[title] = plot_settings.checkbox(title, value=False, key=title)

        for sample_combination in flip_combination_axes:
            if flip_combination_axes[sample_combination]:
                display_combinations[sample_combination] = display_combinations[sample_combination][::-1]
        h_space = inter_space / grid_columns
        v_space = inter_space / grid_rows

        figure = make_subplots(cols=grid_columns, rows=grid_rows, subplot_titles=subplot_titles, horizontal_spacing=h_space, vertical_spacing=v_space)

        row = 1
        col = 1
        plot_number = 1
        legend_groups = set()
        for sample_combination in list(display_combinations.values())[:-1]:

            # We currently have no structure to guarantee primary tumors on x axes and metastases on y since we
            x_y_axes = (data_type + '_' + sample_combination[0], data_type + '_' + sample_combination[1])

            px_figure = px.scatter(
                data_frame,
                x=x_y_axes[0],
                y=x_y_axes[1],
                range_x=range_x,
                range_y=range_y,
                color="Cluster",
                color_discrete_sequence=cluster_colors,
                facet_col="Cluster",
                hover_data={
                    # x_y_axes[0]: False,  # Displays VAF value
                    # x_y_axes[1]: False,
                    'ref_counts_' + sample_combination[0]: True,
                    'alt_counts_' + sample_combination[0]: True,
                    'ref_counts_' + sample_combination[1]: True,
                    'alt_counts_' + sample_combination[1]: True,
                    'major_cn_' + sample_combination[0]: True,
                    'minor_cn_' + sample_combination[0]: True,
                    'major_cn_' + sample_combination[1]: True,
                    'minor_cn_' + sample_combination[1]: True,
                    'Cluster': True,
                    "Mutation": True,
                    "Variant_Type": True,
                    'Impact': True,
                    'Gene': True,
                },
            )

            for trace in px_figure['data']:

                # Avoid having the legend duplicated for each added subplot
                if trace['legendgroup'] in legend_groups:
                    trace['showlegend'] = False
                else:
                    legend_groups.add(trace['legendgroup'])
                figure.add_trace(trace, row=row, col=col)

            plot_number += 1
            row = int(plot_number / grid_columns) + 1 if plot_number % grid_columns else int(plot_number / grid_columns)
            col = plot_number % grid_columns or grid_columns

        figure.update_traces(
            marker=marker,
            # selector=dict(mode='markers'),

        ).update_layout(
            hoverlabel_align='left',  # Necessary for streamlit to make text for all labels align left
            width=subplot_size*grid_columns + (grid_columns - 1) * h_space,
            height=subplot_size*grid_rows + (grid_rows - 1) * v_space,
        ).update_yaxes(
            range=range_y
        ).update_xaxes(
            range=range_x
        )

    # Single Plot
    else:
        # User input - plot width
        plot_width = visual_appearance.slider('Plot width', min_value=200, max_value=1000, value=800, step=100)

        # User input - Flip axes
        flip_axes = streamlit.checkbox('Flip axes', value=False)
        orig_sample_combination = sample_combination
        if flip_axes:
            sample_combination = sample_combination[::-1]
        else:
            sample_combination = orig_sample_combination

        # We currently have no structure to guarantee primary tumors on x axes and metastases on y. We could implement a manual axis flip
        x_y_axes = (data_type + '_' + sample_combination[0], data_type + '_' + sample_combination[1])

        figure = px.scatter(data_frame, x=x_y_axes[0], y=x_y_axes[1],
                            range_x=range_x,
                            range_y=range_y,
                            color="Cluster",
                            color_discrete_sequence=cluster_colors,
                            width=plot_width, height=plot_width,
                            # facet_col='Sample',
                            hover_data={
                                # x_y_axes[0]: False,
                                # x_y_axes[1]: False,
                                'ref_counts_' + sample_combination[0]: True,
                                'alt_counts_' + sample_combination[0]: True,
                                'ref_counts_' + sample_combination[1]: True,
                                'alt_counts_' + sample_combination[1]: True,
                                'major_cn_' + sample_combination[0]: True,
                                'minor_cn_' + sample_combination[0]: True,
                                'major_cn_' + sample_combination[1]: True,
                                'minor_cn_' + sample_combination[1]: True,
                                'Cluster': True,
                                "Mutation": True,
                                "Variant_Type": True,
                                'Impact': True,
                                'Gene': True,
                            },
                            )

        figure.update_traces(
            marker=marker,
            selector=dict(mode='markers'),
        ).update_layout(
            hoverlabel_align='left'  # Necessary for streamlit to make text for all labels align left
        )
    streamlit.plotly_chart(figure, theme=None)
