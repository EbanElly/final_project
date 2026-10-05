import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data/tanzania_individual_health_data.csv')

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')  # Hide axes
table = ax.table(cellText=df.head().values, colLabels=df.columns, cellLoc='left', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Save the table as an image
plt.savefig('./images/image3.png', bbox_inches='tight', dpi=300)
plt.close()