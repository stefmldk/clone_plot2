import itertools
import streamlit
import pandas
import os
import plotly_express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

__version__ = '1.8.0'

# Production candidate. Can do VAF/VAF plots and heatmaps similar to the one in Stephanies PhD: page 53 - fig. 3
# Todo: Make the cluster remove ability available for both plot types and fix color to clusters. Also, ability to filter minimal max VAF
# Cluster colors should be part of the input file


# New in version 1.7: Added a new heat-map kind of plot similar to the one in Stephanies PhD: page 53 - fig. 3


streamlit.set_page_config(layout="wide")

# User input - File upload - starting point
uploaded_file = streamlit.sidebar.file_uploader("Choose a file to start", key='file_uploader')

if uploaded_file is not None:
    # streamlit.write(streamlit.session_state)

    patient_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
    data_frame = pandas.read_csv(uploaded_file, sep='\t')

    # Sort data frame relative to cluster names (in order to get legends sorted as they will be in the order they are added)
    # data_frame = data_frame.sort_values(by='Cluster', key=lambda col: col.map(lambda x: int(x.split('_')[1])))

    column_names = data_frame.columns
    vaf_columns = [name for name in column_names if (name.startswith('VAF') and not name.startswith('VAF_CCF'))]
    # pyclone_ccf_columns = [name for name in column_names if name.startswith('pyclone')]
    # vaf_ccf_columns = [name for name in column_names if name.startswith('CCF')]
    # minor_cn_columns = [name for name in column_names if name.startswith('min')]
    # major_cn_columns = [name for name in column_names if name.startswith('maj')]
    # ref_counts_columns = [name for name in column_names if name.startswith('ref')]
    # alt_counts_columns = [name for name in column_names if name.startswith('alt')]
    sample_names = [name.split('_')[1] for name in vaf_columns]

    all_clusters = data_frame.Cluster.unique()
    clusters = all_clusters

    # Combine sample names

    # Since we do not know the interrelation of samples - i.e. which is primary tumor and which is metastasis etc. we make
    # no attempt to organize combinations in terms of which sample name is first/last in combinations. As this, however,
    # will effect what axis each sample goes to, we implement the ability to switch axes on the plots later in the script.

    #################################################################################
    #                           SELECT CLUSTERS EXPANDER                            #
    #################################################################################

    # Define UI

    def check_all():
        for cluster in all_clusters:
            streamlit.session_state[cluster] = True

    def uncheck_all():
        for cluster in all_clusters:
            streamlit.session_state[cluster] = False

    select_clusters = streamlit.sidebar.expander('Select clusters')
    check_all_button = select_clusters.button('Check all', 'check_all_button', on_click=check_all)
    uncheck_all_button = select_clusters.button('Uncheck all', 'uncheck_all_button', on_click=uncheck_all)

    show_cluster = {}
    for cluster in all_clusters:
        if cluster not in streamlit.session_state:
            streamlit.session_state[cluster] = True
        show_cluster[cluster] = select_clusters.checkbox(cluster, key=cluster)

    # Filter data frame based on show_cluster values
    clusters_to_include = [cluster for cluster in show_cluster if show_cluster[cluster]]
    data_frame = data_frame[data_frame['Cluster'].isin(clusters_to_include)]
    cluster_colors = data_frame.Color.unique()
    clusters = clusters_to_include

    #################################################################################
    #                              Select plot type                                 #
    #################################################################################

    plot_type_ui = streamlit.sidebar.expander('Plot type')
    plot_type = plot_type_ui.radio('Select plot type', ('Dot plot', 'Heat map', '3D line plot', '3D surface plot'), key='plot_type')

    if plot_type == 'Dot plot':

        pairwise_sample_combinations = itertools.combinations(sample_names, 2)

        display_combinations = {'{}_vs_{}'.format(combination[0], combination[1]): list(combination) for combination in
                                pairwise_sample_combinations}
        number_of_plots = len(display_combinations)

        if number_of_plots > 1:

            # Add the choice of multiplot to the samples dropdown
            display_combinations['MultiPlot'] = 'MultiPlot'

            # User input - Which samples to plot
            sample_combination = display_combinations[
                streamlit.sidebar.selectbox('Please select which samples to compare', display_combinations.keys())]
        else:
            sample_combination = list(display_combinations.values())[0]

        # Data filtering
        # data_filtering = streamlit.sidebar.expander('Data filters')
        #
        # # User input
        # min_vaf = data_filtering.slider('Minimal MAF', min_value=0.0, max_value=1.0, value=0.0, step=0.01)

        # User input - Which data type to plot
        data_type = streamlit.sidebar.radio('Select data type', ('VAF', 'cluster_CCF', 'VAF_CCF'))

        #################################################################################
        #                           VISUAL APPEARANCE EXPANDER                          #
        #################################################################################

        # Define Expander for visual appearance
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
        x_y_ranges = ([-0.05, 1.05], [-0.1, 2.1])
        range_x = x_y_ranges[0] if data_type == 'VAF' else x_y_ranges[1]
        range_y = x_y_ranges[0] if data_type == 'VAF' else x_y_ranges[1]

        # Multiplot
        if sample_combination == 'MultiPlot':

            # User input
            grid_columns = visual_appearance.slider('Number of grid columns', min_value=2, max_value=8, value=3, step=1)

            # User input
            inter_space = visual_appearance.slider('Space between plots', min_value=0.05, max_value=0.5, value=0.2,
                                                   step=0.05)

            grid_rows = int(number_of_plots / grid_columns) + 1 if number_of_plots % grid_columns else int(
                number_of_plots / grid_columns)

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

            figure = make_subplots(cols=grid_columns, rows=grid_rows, subplot_titles=subplot_titles,
                                   horizontal_spacing=h_space, vertical_spacing=v_space)

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
                width=subplot_size * grid_columns + (grid_columns - 1) * h_space,
                height=subplot_size * grid_rows + (grid_rows - 1) * v_space,
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


            # If for some reason variants are uninformative in some samples (var + ref_counts = 0), we may indicate that by
            # finding indexes for those cases and individually styling each corresponding marker in the trace, using:
            # https://stackoverflow.com/questions/70275607/how-to-highlight-a-single-data-point-on-a-scatter-plot-using-plotly-express
            #
            # # User input - indicate uninformative VAFs
            # indicate_uninformative_vafs = streamlit.checkbox('Indicate potentially uninformative VAFs', value=False)
            #
            # if indicate_uninformative_vafs:
            #     uninformative_vaf_indexes = data_frame.index[data_frame['ref_counts_' + sample_combination[0]] == 0].tolist()
            #     # uninformative_vaf_indexes = data_frame.index[
            #     #     (
            #     #         (data_frame['ref_counts_' + sample_combination[0]] == 0) &
            #     #         (data_frame['alt_counts_' + sample_combination[0]]) == 0
            #     #     ) |
            #     #     (
            #     #         (data_frame['ref_counts_' + sample_combination[1]] == 0) &
            #     #         (data_frame['alt_counts_' + sample_combination[1]] == 0)
            #     #     )
            #     # ].tolist()
            #
            #     streamlit.write('ref_counts_' + sample_combination[0])
            #     streamlit.write(uninformative_vaf_indexes)
            #
            # However, this does not seem to be a problem with current datasets, so not implemented

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
    elif plot_type == 'Heat map':
        # Add UI for manipulating the heatmap
        edit_plot_ui = streamlit.sidebar.expander('Edit Plot')

        #################################################################################
        #                               Set gap between tiles                           #
        #################################################################################

        tile_gap = edit_plot_ui.slider('Gap between tiles', min_value=0, max_value=4, value=1)

        #################################################################################
        #                              Set tile width                                 #
        #################################################################################

        user_set_tile_width = edit_plot_ui.slider('Tile width', min_value=50, max_value=400, value=150, key='user_set_tile_width')

        #################################################################################
        #                              Set plot height                                 #
        #################################################################################
        number_of_rows = len(data_frame)

        def change_plot_height():
            streamlit.session_state.user_set_plot_height = streamlit.session_state.user_set_tile_height * number_of_rows

        def change_row_height():
            streamlit.session_state.user_set_tile_height = round(streamlit.session_state.user_set_plot_height / number_of_rows)


        # Set default values useing the session-state API rather than the value argument
        if 'user_set_tile_height' not in streamlit.session_state:
            streamlit.session_state.user_set_tile_height = 18
        if 'user_set_plot_height' not in streamlit.session_state:
            streamlit.session_state.user_set_plot_height = 18 * number_of_rows

        user_set_tile_height = edit_plot_ui.slider('Tile height', min_value=1, max_value=25, key='user_set_tile_height', on_change=change_plot_height)

        user_set_plot_height = edit_plot_ui.number_input('Plot height', key='user_set_plot_height', on_change=change_row_height)

        plot_height = user_set_plot_height
        #
        # edit_plot_ui.write('Current plot height: ')
        # edit_plot_ui.write(plot_height)

        # Strategy: use Heatmap from plotly.graph_objects - one trace per clone and potentially zero space between plots

        #################################################################################
        #                              Space between color bars                         #
        #################################################################################

        color_bars_distance_factor = edit_plot_ui.number_input('Color bars distance factor', min_value=0.001, max_value=0.5, value=0.01, step=0.005)

        clusters = data_frame.Cluster.unique()

        genes_per_cluster = {}

        for cluster in clusters:
            genes_per_cluster[cluster] = len(data_frame[data_frame['Cluster'] == cluster])

        subplot_heights = [genes_per_cluster[cluster] / number_of_rows for cluster in clusters]  # In fractions of the whole plot

        # For vertical bars (bad if one or more clusters have few variants):
        # Placing color bars at approximately the middle of each subplot (via fractions of the whole plot)
        # New traces are added top down, so we need to reflect that on the factor - i.e. we start high and go to low in the y-position
        # color_bars_y_position = [sum(subplot_heights[i + 1:]) + subplot_heights[i] / 2 for i in range(len(subplot_heights))]

        color_bars_y_position = [1-i*color_bars_distance_factor for i in range(len(clusters))]

        #################################################################################
        #                               Remove gene names                               #
        #################################################################################

        remove_gene_names = edit_plot_ui.checkbox('Remove gene names', value=False)


        figure = make_subplots(
            len(clusters),
            1,
            shared_xaxes=True,
            row_heights=subplot_heights,
            vertical_spacing=0,
            x_title='Sample',
            y_title='Gene',
        )
        figure.layout.annotations[0].yshift = -30
        figure.layout.annotations[0].text = '<b>Sample</b>'
        figure.layout.annotations[1].xshift = -60
        figure.layout.annotations[1].text = '<b>Gene</b>'

        figure.update_yaxes(
            showgrid=False,
            # title_font_size=20,
        )
        figure.update_xaxes(
            showgrid=False,
            tickcolor='rgba(0,0,0,0)',
        )

        # The idea is to display rgba values using the cluster colors in turn as colors and the VAF as a transparency.

        def hex_2_rgb(hex_string):
            r_hex = hex_string[1:3]
            g_hex = hex_string[3:5]
            b_hex = hex_string[5:7]
            return [int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)]  # We return a list, so we can append


        color_values = []

        vaf_columns = []
        for sample in sample_names:
            vaf_columns.append('VAF_' + sample)

        # Because each subplot gets its own yaxis config (see: https://stackoverflow.com/questions/72464495/plotly-how-to-change-ticks-in-a-subplot), we update nticks for each (we set it to the row number of the full figure, it seems the actual number is set to the number for the particular cluster)
        yaxis_nticks_layout_dict = {}

        for i, cluster in enumerate(clusters):
            vaf_values = []

            # Store the rgb color for this cluster
            cluster_rgb = hex_2_rgb(cluster_colors[i])

            # Get dataframe for just this cluster
            cluster_df = data_frame[data_frame['Cluster'] == cluster]
            cluster_y_axis_genes = cluster_df.Gene.tolist()

            yaxis_nticks_layout_dict['yaxis{}_nticks'.format('' if i == 0 else str(i + 1))] = len(cluster_y_axis_genes)

            if remove_gene_names:
                yaxis_nticks_layout_dict['yaxis{}_showticklabels'.format('' if i == 0 else str(i + 1))] = False
                yaxis_nticks_layout_dict['yaxis{}_tickcolor'.format('' if i == 0 else str(i + 1))] = 'rgba(0,0,0,0)'

            for index, row in cluster_df.iterrows():
                row_vaf_values = []
                for vaf_column in vaf_columns:
                    row_vaf_values.append(row[vaf_column])
                vaf_values.append(row_vaf_values)

            color_bar = dict(
                title=dict(
                    text=cluster,
                    side='top'
                ),
                tickmode='auto',
                lenmode='pixels',
                thicknessmode='pixels',
                len=70,
                thickness=10,
                y=color_bars_y_position[i],
                x=1.2,
                orientation='h'
            )

            figure.add_trace(
                go.Heatmap(
                    x=sample_names,
                    y=cluster_y_axis_genes,
                    z=vaf_values,
                    colorscale=[
                        [0, 'rgba({},{},{},0)'.format(cluster_rgb[0], cluster_rgb[1], cluster_rgb[2])],
                        [1, 'rgba({},{},{},1)'.format(cluster_rgb[0], cluster_rgb[1], cluster_rgb[2])]
                    ],
                    colorbar=color_bar,
                    # showscale=False,
                    xgap=tile_gap,  # Gap between heatmap tiles
                    ygap=tile_gap
                ),
                row=(i + 1), col=1,
            )

        # height_factor = streamlit.slider('Plot height', value=18, min_value=1, max_value=25, step=1)

        figure.update_layout(
            plot_bgcolor='white',
            height=user_set_plot_height or number_of_rows * user_set_tile_height,
            width=len(sample_names) * user_set_tile_width + 100,
        )

        # Also updating the layout for the y axes ticks
        figure.update_layout(yaxis_nticks_layout_dict)

        # figure.layout.yaxis.showticklabels = False

        streamlit.plotly_chart(figure, use_container_width=False, sharing="streamlit", theme=None)
    elif plot_type == '3D line plot':
        # First we need to reformat the data. We create a new data frame with four columns: Sample, Gene, VAF, Cluster
        new_data = {
            'Sample': [],
            'Gene': [],
            'VAF': [],
            'Cluster': [],
        }

        gene_column = data_frame.Gene.tolist()
        for vaf_column_name in vaf_columns:
            sample_name = vaf_column_name.split('_')[1]
            vaf_column = data_frame[vaf_column_name].tolist()
            gene_column = list(range(len(gene_column)))
            cluster_column = data_frame.Cluster.tolist()
            sample_column = len(vaf_column) * [sample_name]
            new_data['Sample'] += sample_column
            new_data['Gene'] += gene_column
            new_data['VAF'] += vaf_column
            new_data['Cluster'] += cluster_column

        # We replace the gene names with numbers - to account for genes that are represented more than once
        # tick_text = gene_column

        new_df = pandas.DataFrame(new_data)

        # We sort relative to cluster
        sorted_new_df = new_df.sort_values(
            by='Cluster',
            kind='mergesort'
        )

        sample_frames = {}
        for sample in sample_names:
            sample_frames[sample] = sorted_new_df[sorted_new_df['Sample'].isin([sample])]

        # Add the first figure in order to include the trace
        # plotly_figures = [
        #     px.line_3d(sample_frames[sample], x='Sample', y='Gene', z='VAF', color='Cluster', template='plotly')
        # ]
        plotly_figures = []
        for sample in list(sample_frames.keys()):
            plotly_figures.append(px.line_3d(sample_frames[sample], x='Sample', y='Gene', z='VAF', color='Cluster', color_discrete_sequence=cluster_colors))

        go_data = plotly_figures[0].data

        for plotly_figure in plotly_figures[1:]:
            go_data += plotly_figure.data

        figure = go.Figure(data=go_data)

        # Avoid having cluster legends repeated for each trace
        legend_groups = set()
        for trace in figure['data']:

            # Avoid having the legend duplicated for each added subplot
            if trace['legendgroup'] in legend_groups:
                trace['showlegend'] = False
            else:
                legend_groups.add(trace['legendgroup'])

        figure.update_layout(
            width=1000,
            height=1000,
            scene=dict(
                xaxis=dict(
                    title_text='Sample'
                ),
                yaxis=dict(
                    ticktext=gene_column,
                    title_text='"Variant"'
                    # nticks=len(gene_column)
                ),
                zaxis=dict(
                    title_text='VAF'
                ),
            ),
            template='plotly'
        )
        streamlit.plotly_chart(figure, theme='streamlit')
    elif plot_type == '3D surface plot':
        from scipy.signal import savgol_filter

        #################################################################################
        #                                 Edit plot UI                                  #
        #################################################################################
        edit_plot_ui = streamlit.sidebar.expander('Edit plot')

        #################################################################################
        #                             Manipulate plot UI                                #
        #################################################################################

        manipulate_plot = edit_plot_ui.radio('Manipulate plot', ['Original', 'Savitzky-Golay smoothing', 'Peak sorting'])

        # Separate dataframe into dataframes for each cluster. Since we want each cluster to take up a separate space at
        # the X/Y dimensions, we must for each cluster fill the other "clusters" spaces with NaN values.
        if manipulate_plot == 'Savitzky-Golay smoothing':

            #################################################################################
            #                  Set window size for Savitzky-Golay smoothing                 #
            #################################################################################

            window_size = edit_plot_ui.slider('Window size', min_value=2, max_value=150, value=5)

        elif manipulate_plot == 'Peak sorting':
            sort_type = edit_plot_ui.radio('Sort by sample having the', ['highest VAF peak', 'highest overall VAF'])

        cluster_dataframes = {}
        for i, cluster in enumerate(clusters):

            # get VAF columns data for current cluster only
            cluster_data = data_frame[data_frame['Cluster'].isin([cluster])].loc[:, vaf_columns]
            if manipulate_plot == 'Savitzky-Golay smoothing':
                if len(cluster_data) > 1:
                    cluster_data = cluster_data.apply(lambda x: savgol_filter(x, min(window_size, len(cluster_data)),1), axis=0)
            elif manipulate_plot == 'Peak sorting':
                if sort_type == 'highest VAF peak':
                    max_vaf_sample = cluster_data.max().sort_values(ascending=False).index[0]
                elif sort_type == 'highest overall VAF':
                    max_vaf_sample = cluster_data.sum().sort_values(ascending=False).index[0]

                cluster_indexes = cluster_data.index.tolist()
                max_vaf_sample_index = cluster_data.columns.get_loc(max_vaf_sample)
                vaf_array = cluster_data.values.tolist()
                # streamlit.write(vaf_array)
                descending_vaf_array = sorted(vaf_array, key=lambda x: x[max_vaf_sample_index], reverse=True)
                # streamlit.write(descending_vaf_array)
                peak_sorted_vaf_array = []
                prepend = False
                for row in descending_vaf_array:
                    if prepend:
                        peak_sorted_vaf_array = [row] + peak_sorted_vaf_array
                        prepend = False
                    else:
                        peak_sorted_vaf_array = peak_sorted_vaf_array + [row]
                        prepend = True
                    # streamlit.write(peak_sorted_vaf_array)
                # streamlit.write(peak_sorted_vaf_array)
                cluster_data = pandas.DataFrame(peak_sorted_vaf_array, columns=vaf_columns, index=cluster_indexes)
                # streamlit.write(peak_sorted_vaf_array)


            before_df = pandas.DataFrame(index=list(range(data_frame.first_valid_index(), cluster_data.first_valid_index())), columns=vaf_columns)
            after_df = pandas.DataFrame(index=list(range(cluster_data.last_valid_index() + 1, data_frame.last_valid_index() + 1)), columns=vaf_columns)
            cluster_dataframes[cluster_colors[i]] = pandas.concat([before_df, cluster_data, after_df], axis=0)
            # streamlit.write(cluster_dataframes[cluster_colors[i]])


        figure = go.Figure(data=[
            go.Surface(
                z=cluster_dataframes[color],
                x=sample_names,
                y=list(range(len(cluster_dataframes[color]))),
                colorscale=[[0, '#d9d9d9'], [1, color]],
                showscale=False
            ) for color in cluster_dataframes
        ])

        # To keep layout (rotate, zoom, etc. between plot edits) - see: https://stackoverflow.com/questions/68798315/how-to-update-plotly-plot-and-keep-ui-settings
        figure['layout']['uirevision'] = 'some string'
        
        figure.update_layout(
            width=1500,
            height=1500,
            scene=dict(
                yaxis=dict(
                    title='Variants',
                    showticklabels=False
                    # ticktext=list(range(len(data_frame))),
                    # nticks=len(data_frame)
                ),
                xaxis=dict(
                    title='Sample'
                ),
                zaxis=dict(
                    title='VAF'
                )
            )
        )

        streamlit.plotly_chart(figure, theme='streamlit')
