from pycirclize import Circos
from pycirclize.parser import Matrix

from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import pandas as pd
import os



# Define the link_kws_handler function
def custom_link_kws_handler(from_label, to_label):
    """
    Custom handler for link keyword arguments.
    Makes links identified in transparent_links set (p_value > 0.05) transparent.
    """
    if (from_label, to_label) in transparent_links:
        return {"alpha": 0.0}  # Make link transparent
    return None # Use default link_kws (or global link_kws) for other links


# Remove the short channels
# Function to extract the numeric part of the 'D' component
def extract_d_value(channel):
    try:
        return int(channel.split('-')[1][1:])  # Extract the numeric part after 'D'
    except (IndexError, ValueError):
        return None  

def grouping_channels(all_seeds):
    # #### This part of the program is for grouping the channels by brain areas
    input_file_grouping = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\MULPA fOLD areas\Channels to Brain Areas using fOLD.xlsx"
    # r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\Dataset_recategorized\Source-Detector Positions.xlsx"

    # in Sheet1 we have the source and detector numbers and their brain labes
    channel_positions = pd.read_excel(input_file_grouping, sheet_name="CLEAN")
    # Create a new column with the paired format "S<source_number>-D<detector_number>"
    channel_positions["Channel_Pair"] = channel_positions.apply(
        lambda row: f"S{row['source']}-D{row['detector']}", axis=1
    )

    # Group the channel pairs by their corresponding labels (e.g., "prefrontal cortex")
    grouped_channels = channel_positions.groupby("Category")["Channel_Pair"].apply(list).to_dict()

    ## Maybe I should not delete the seed if I am going to import the data from the other seeds
    # Delete the seed channel from the grouped channels
    for seed in all_seeds:
        for brain_area, channels in grouped_channels.items():
            if seed in channels:
                channels.remove(seed)
    # Add a new label for seeds
    # grouped_channels["seeds"] = all_seeds
    # This is for Seeds Outside ROI
    # grouped_channels = {"Seeds Outside ROI": all_seeds, **grouped_channels}  # Ensure "seeds" is the first group
    
    #! this is for HbO motor seed
    grouped_channels = {"HbR Motor Seeds": all_seeds, **grouped_channels}  # Ensure "seeds" is the first group


    return grouped_channels
    #### end of the grouping part
    # 

# ! change the figure name depending on the HbO/HbR
# fig_name = "Functional Connectivity SBA t-test for HbO Motor Seeds"
fig_name = "Functional Connectivity SBA t-test for HbR Motor Seeds"

## Trying to generalize it to input folder as seed_files

# # ! HbO
# seed_files = [
#     r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\Dataset_recategorized\01_RestingState\1_SBA\2a_Grouped_Betas_ttest\Group_GLM_Betas_RestingState_S10_D7_hbo_ttest.xlsx",
#     r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\Dataset_recategorized\01_RestingState\1_SBA\2a_Grouped_Betas_ttest\Group_GLM_Betas_RestingState_S25_D23_hbo_ttest.xlsx"
# ]
# # ! HbR/Outside Motor
seed_files = [
    r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\Dataset_recategorized\01_RestingState\1_SBA\2a_Grouped_Betas_ttest\Group_GLM_Betas_RestingState_S10_D6_hbr_ttest.xlsx",
    r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\Dataset_recategorized\01_RestingState\1_SBA\2a_Grouped_Betas_ttest\Group_GLM_Betas_RestingState_S25_D24_hbr_ttest.xlsx"
    ]
# channels outside ROI

# From to structure -> Value
# Initialize an empty DataFrame to store all data
matrix = pd.DataFrame(
    {
        "Seed": [],
        "Channel": [],
        "T_Value": [],
        "P_Value": [],
    }
)


all_seeds = []
for file in seed_files:
    # define the seed
    filename = os.path.basename(file)

    # # ! replace depending on HbO/HbR
    # HbR
    seed_name = filename.replace("Group_GLM_Betas_RestingState_", "").replace("_hbr_ttest", "").replace(".xlsx","").replace("_","-")
    
    # # HbO
    # seed_name = filename.replace("Group_GLM_Betas_RestingState_", "").replace("_hbo_ttest.xlsx", "").replace("_","-")
    
    
    all_seeds.append(seed_name)
    print(seed_name)
    data = pd.read_excel(file)
    # Data of seed name; t values and p values
    channels_tvalues = data.iloc[:, :2]
    channels_pvalues = data.iloc[:,[0,3]]
    # Remove the row where the first column matches "S10-D7" which is the seed
    channels_tvalues = channels_tvalues[channels_tvalues.iloc[:, 0] != seed_name]
    channels_pvalues = channels_pvalues[channels_pvalues.iloc[:, 0] != seed_name]
    # Filter the channels based on the threshold: D <= 28 above that we have short channels
    channels_tvalues = channels_tvalues[channels_tvalues.iloc[:, 0].apply(lambda x: extract_d_value(x) <= 28 if extract_d_value(x) is not None else False)]
    channels_pvalues = channels_pvalues[channels_pvalues.iloc[:, 0].apply(lambda x: extract_d_value(x) <= 28 if extract_d_value(x) is not None else False)]
    # # Prepare the matrix for the circos plot
    # From to structure 
    new_matrix = pd.DataFrame(
        {
            "Seed": [seed_name] * len(channels_tvalues), # multiply by number of rows
            "Channel": channels_tvalues.iloc[:, 0].tolist(), # this are the channel names
            "T_Value": channels_tvalues.iloc[:, 1].tolist(), # these are the t values
            "P_Value": channels_pvalues.iloc[:, 1].tolist(),
        }
    )
    # Append the new data to the existing matrix
    # print(new_matrix.to_string())
    matrix = pd.concat([matrix, new_matrix], ignore_index=True)

# print(matrix.to_string())


# Create a set of links that should be transparent (p_value > 0.05)
transparent_links = set()
for index, row in matrix.iterrows():
    if row["P_Value"] > 0.05:
        transparent_links.add((row["Seed"], row["Channel"]))

link_colors = []
node_colors = []

# Link colors & Node colors
# Create a node color dictionary and set everything to white and after we are going to insert colors based on significance
for channel in matrix["Channel"]:
    node_colors.append((channel, to_rgba("white", alpha = 0.8)))  # Set all nodes to white

# Here I will set the t values that are significant to be darkslateblue and those not significant to be white
# also the t values that are not significant will be set to near 0
# Create a color list of the links based on the p-values
for seed, channel, p_value, t_value in zip(matrix["Seed"], matrix["Channel"], matrix["P_Value"], matrix["T_Value"]):
    if p_value <= 0.05 and p_value > 0.01:
        link_colors.append((seed, channel, "deepskyblue"))
        node_colors.append((channel, "deepskyblue"))
    elif p_value <= 0.01:
        link_colors.append((seed, channel, "darkslateblue"))
        node_colors.append((channel, "darkslateblue"))


# Get the first color from tab20c colormap for the seeds
tab20c_cmap = colormaps.get_cmap("tab20c")
first_tab20c_color_rgba = tab20c_cmap(0.0)  # Get the first color (at position 0.0)
first_tab20c_color_hex = to_hex(first_tab20c_color_rgba) # Convert to hex

# make the seeds indigo or the same as the first color of the tab20c colormap which is the same as the sector category above it
for seed in all_seeds:
    node_colors.append((seed, first_tab20c_color_hex))  # Set all nodes to white
    # link_colors.append((seed, seed, "indigo"))

# now for the remaining nodes that are white, dont have a significant connection to any node
# because they remained white so I will make their t-values near 0 to not clutter our graph

for i, (channel, t_value, p_value) in enumerate(zip(matrix["Channel"], matrix["T_Value"], matrix["P_Value"])):
    if p_value > 0.05:
        matrix.loc[i, "T_Value"] = 1  # Set the t-value to a small value
print(matrix.to_string())


"""This part is irrelevant for now"""
# for i, (seed, channel) in enumerate(zip(matrix["Seed"], matrix["Channel"])):
#### T values code snipet
# This is the part where we set the color map for the t-values
# I decided not to use this part of the code
# matrix = matrix.drop(columns=["P_Value"])
# We have this problem now.
# Adding more seeds in effect dublicates the t-values assigned to a channel
# for example we have seed x paired with Ch1 and seed y paired with Ch1
# resulting in a t value for x-ch1 and another for y-ch2
# so how should we color the nodes? based on which t-value
# a good thing is that the links show the proportion of the t-value 
# assigned at each pair
# Below my strategy of solving this problem is to use the average t-value 
# of the x-ch1 and y-ch2
# maximum t value of the 2 maybe??
# Initialize a dictionary to store the largest t-values from the two to color based on this strategy
# maximum_t_values = {}
# # Iterate through all unique channels
# for channel in matrix["Channel"].unique():
#     # Find all rows in the matrix where the channel appears
#     channel_rows = matrix[matrix["Channel"] == channel]
# #     # Calculate the max t-value for the channel
#     max_t_value = channel_rows["T_Value"].max()
#     print(f"{channel_rows['T_Value']} with max {max_t_value}")
    
#     # Store the max t-value in the dictionary
#     # average_t_values[channel] = avg_t_value
#     maximum_t_values[channel] = max_t_value
# standardize the average t-values and center them around the median
# we do this because otherwise the map will show the colors with some offset
# ## Using TwoSlopeNorm to center the colormap around the median value is not meaningful 
# # anymore becaause the negative t-values are 0 now after what we did above and in our data 
# # we have mostly max t values that are significant and the non significant pairs are set to 0 
# # so the central point 0 is not necessary anymore
# norm = TwoSlopeNorm(vmin=min(maximum_t_values.values()), 
#                     vmax=max(maximum_t_values.values()), 
#                     vcenter=np.median(list(maximum_t_values.values())))
# norm = Normalize(vmin=min(maximum_t_values.values()), vmax=max(maximum_t_values.values()))
# print(f"min: {min(average_t_values.values())} max: {max(average_t_values.values())} median: {np.median(list(average_t_values.values()))}")
# #  Print average t-values in a readable format
# print("Average t-values for each channel:")
# for channel, avg_t_value in average_t_values.items():
#     print(f"Channel: {channel}, Average t-value: {avg_t_value:.4f}")
# # Seismic colormap for the T_Values
# colormap = colormaps.get_cmap("seismic")
# color_dict = {}
# for channel, max_t in maximum_t_values.items():
#     normalized_value = norm(max_t)  # Normalize the average t-value
#     rgb_color = colormap(normalized_value)  # Get the RGBA color from the colormap
#     color_dict[channel] = to_hex(rgb_color)  # Convert RGBA to HEX for compatibility
# Convert t-values to z-scores because otherwise the nodes will be very small for the lowest 
# t value and larger for t values near the meadian, where in fact they should be the smallest
# print(f"AFTER z score min= {matrix.min()} max= {matrix.max()} \n {matrix.to_string()}")
# matrix["T_Value"] = zscore(matrix["T_Value"])

# before parsing it to circo we need to convert the data to absolute numbers
# because the t-values can be negative and Circos doesnt know what to do with them
matrix["T_Value"] = matrix["T_Value"].abs()

matrix = matrix.drop(columns=["P_Value"])

grouped_channels = grouping_channels(all_seeds)
# print(grouped_channels)

#### Before we parse the data to circos we need to sort the channels based on the brain areas
# Create a new column in the matrix to store the brain area labels
matrix["Brain_Area"] = matrix["Channel"].map(
    {channel: label for label, channels in grouped_channels.items() for channel in channels}
)

brain_area_order = []
for area in grouped_channels.keys():
    brain_area_order.append(area)

# print(f"Brain area order: {brain_area_order}")
matrix["Brain_Area"] = pd.Categorical(matrix["Brain_Area"], categories=brain_area_order, ordered=True)

# Now we can sort the matrix based on the brain areas
matrix = matrix.sort_values(by="Brain_Area")
# Reset the index
matrix = matrix.reset_index(drop=True)
# print(matrix.to_string())

mymatrix = Matrix.parse_fromto_table(matrix)



# print(mymatrix)
# Initialize Circos instance for chord diagram plot
circos = Circos.chord_diagram(
    mymatrix,
    space=2, ## Space between sectors
    # cmap = 'viridis', # This is the color of the nodes
    cmap=node_colors, # This is the color of the nodes
    # label_name -> color dict (e.g. `dict(A="red", B="blue", C="green", ...)`)
    link_cmap=link_colors, # this is the color of the links
    # link_cmap=link_colors, # Color dictionary for links (e.g. `dict(A="red", B="blue", C="green", ...)`)
    # ticks_interval=5,
    label_kws={
        "orientation": "vertical",  # Rotate labels: "vertical" or "horizontal"
        "r": 78,        # Adjust radial position of labels
        "size": 4.5,      # Set label size
        "ha": "center",  # Horizontal alignment
        "va": "center",  # Vertical alignment
        "alpha": 0.8, # Transparency of the label
    },
    link_kws=dict(alpha=0.6, direction = 1, arrow_length_ratio =0.05, ec='black', lw=0.07), # ec="black", lw=0.1
        # alpha = transparency of the link; ????
        # direction = -1, 0, 1, 2 # direction of the link
        # arrow_length_ratio = float: how pointy the arrow is (0.05 is default)
    link_kws_handler=custom_link_kws_handler, # Add the custom handler here

    r_lim = [65,75], # Adjust the radial limits of the plot, distance of nodes from the center

)



# Define a color map for brain areas
# Use a predefined colormap (e.g., "tab20" for distinct colors)
brain_area_colors = colormaps.get_cmap("tab20c")  # Use a colormap with distinct colors

# Assign colors to brain areas
brain_area_color_dict = {
    brain_area: to_hex(brain_area_colors(i / len(grouped_channels)))
    for i, brain_area in enumerate(grouped_channels.keys())
}

for brain_area, group_channels in grouped_channels.items():
    # Get the degree limits for the group
    group_deg_lim = circos.get_group_sectors_deg_lim(group_channels)
    print(f"brain area: {brain_area} in {group_channels}: {len(group_channels)} + space: {group_deg_lim}\n")


    # Draw the outer layer for the group
    circos.rect(r_lim=(90, 93), deg_lim=group_deg_lim, fc=brain_area_color_dict[brain_area], ec="black", lw=0.5)
    
    # Calculate the center of the group for labeling
    group_center_deg = sum(group_deg_lim) / 2

    print(f"Group center degree: {group_center_deg}\n\n")
    if group_center_deg < 180:
        myha="left"
    else:
        myha="right"
    circos.text(brain_area, r=97, deg=group_center_deg, size=10, color="black", rotation=0, ha=myha, va="center")

# Add the seed groups separately for all seeds

# ### for debugging purposes when sectors are very large than they should
# for channel in grouped_channels["primary motor & somatosensory"]:
#     sector = circos.get_sector(channel)
#     print(f"Channel: {channel}, Degree Limits: {sector.deg_lim}")

fig = circos.plotfig()
fig.suptitle(fig_name, fontsize=16, fontweight = "bold",y=0.95)  # Adjust `y` to position it above the figure
fig.set_size_inches(14.5, 6, forward=True)  # Set a fixed figure size
# Here we add a color bar as a legend for the t-values
# Maximize the figure window


# cax = fig.add_axes([0.84, 0.15, 0.02, 0.5], frameon=True)  # Create a new axis for the color bar
# cbar = fig.colorbar(
#     ScalarMappable(norm=norm, cmap=colormap),  # Use the seismic colormap and normalized t-values
#     cax = cax, 
#     orientation="vertical",  # Vertical color bar
#     # extend='both',  # This shows arrows at ends for out-of-range values
# #     pad=0.05,       # padding between colorbar and main plot
# #     shrink=0.95,    # slightly smaller than default
# )
# cbar.set_label("Average t-value", fontsize=10)  # Add a label for the color bar


# Add the legend for links
link_handles = [
    Line2D([0], [0], color="deepskyblue", lw=1.5, alpha=0.6, label="p ≤ 0.05*"),
    Line2D([0], [0], color="darkslateblue", lw=1.5, alpha=0.6, label="p ≤ 0.01**"),
]
# Add legend to the figure
fig.legend(handles=link_handles, 
        #   loc='upper right',  # Position on the right side
        bbox_to_anchor=(0.75, 0.99),  # Fine-tune position
        fontsize=10,
        frameon=True,
        fancybox=True,
        title="Significance (FDR corrected)",
        )


plt.show()

# Save the plot to a file
output_dir = r"C:\Users\foivo\Documents\Satori\SampleData\MULPA Dataset\Dataset_recategorized\01_RestingState\1_SBA\2c_Grouped_Betas_ttest_circos"
output_path = os.path.join(output_dir, f"{fig_name}.png")
fig.savefig(output_path, dpi=300)

