from datetime import date
import os
import model as md

# x = str(date.today())+'('+'JAPANESE OCHA'+')'+'.pkl'
# print(x)

models = os.listdir('models/')
print(models[0][11:-5])

# print(len(os.listdir('models/')) <= 0)

pathData = 'data/dataset.csv'
df = md.proccessData(pathData)
print(df.columns[0])