# %%
#pip install matplotlib > /dev/null
#pip install pandas > /dev/null
#pip install numpy > /dev/null
#pip install seaborn > /dev/null

# %%
# ## Lego challenge
# 
# Zadaniem na dzisiejsze laboratoria jest rozwiazanie problemu klasyfikacji koloru klocka na podstawie wartosci koloru z jego zdjęcia. Do instrukcji dołączony jest zbiór danych.
# Warto zapoznać się z zawartością zbioru danych i samodzielnie sprawdzić zawartośc plików.
# 
# Zadaniem jest zaproponowanie rozwiązania wykorzystujące techniki uczenia maszynowego w taki sposób, aby osiągnąć jak najwyższą skuteczność klasyfikacji modelu. Oprócz miary 'Accuracy' warto spojrzeć na inne metryki określające poprawność klasyfikacji, jak 'Precision' czy 'Recall' aby dostosować kolejne kroki w poprawie proponowanego rozwiązania.
# Wybór technik i metod jest dowolny. Prosze pamiętać aby na wstępnie `odpowiednio` podzielić zbiór danych na podzbiory Trenujący i Testujący, inaczej określenie poprawności działania zaproponowanej techniki będzie niemożliwe.
# 
# Poniżej przedstawiono sposób wczytania danych do programu oraz wyświetlono dostępne dane w przestrzeni trójwymiarowej (kolor punktu oznacza kolor klocka według producenta).
# 
# Powodzenia !

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px


#from google.colab import drive
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

%matplotlib inline
sns.set_theme(style="darkgrid")

# %%
# Next, all libraries containing helper functions are imported.

# %%
# pobierz zbiory danych 
df_data = pd.read_csv('archive/legocolor-basic.csv', delimiter=";")
df_color = pd.read_csv('archive/colors.csv', delimiter=",")

# %%
print(df_data['Color'].unique())

# %%
color_dict = dict(zip(df_color["name"], df_color["rgb"]))

df_datacolor=df_data["Color"]

df_data['Color'] = df_data['Color'].replace(color_dict)
df_data['Color'] = '#' + df_data['Color'].astype(str)
print(df_data['Color'].unique())

# %%
my_cmap = df_data['Color'].unique()

my_cmap = [x[1:] for x in my_cmap]

fig = px.scatter_3d(df_data, x = 'R', y = 'G', z = 'B', color = 'Color', color_discrete_sequence = my_cmap)
fig.update_traces(marker=dict(size = 2.5))
fig.show()
