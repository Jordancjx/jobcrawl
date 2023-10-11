import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\jiaen\Desktop\job_data.csv")
#pd.set_option('display.max_columns', None)

data['Job Title'] = data['Job Title'].str.replace('Chef de partie', 'Chef De Partie')
data['Job Title'] = data['Job Title'].str.replace('Chef de Partie', 'Chef De Partie')
data['Location'] = 'Singapore'
data['Location'] = data['Location'].str.replace('Orchard', 'Singapore')

data.to_csv(r"C:\Users\jiaen\Desktop\output_job_data.csv", index=False)
print(data)
