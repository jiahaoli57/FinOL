import pandas
pandas.options.display.max_rows = None
pandas.options.display.max_columns = None

good_csv = 'TRP.TO.xlsx'
promblem_csv_1 = 'BBD-B.TO.xlsx'
promblem_csv_2 = 'FVI.TO.xlsx'
promblem_csv_3 = 'IFP.TO.xlsx'
promblem_csv_4 = 'LB.TO.xlsx'
promblem_csv_5 = 'MATR.TO.xlsx'
promblem_csv_6 = 'POU.TO.xlsx'
promblem_csv_7 = 'TECK-B.TO.xlsx'
promblem_csv_8 = 'WFG.TO.xlsx'

promblem_csv = promblem_csv_8

df = pandas.read_excel(promblem_csv)
good_df = pandas.read_excel(good_csv)
#
df1 = df.sort_values(by='DATE')
df2 = good_df.sort_values(by='DATE')
df = pandas.merge(df1, df2, on='DATE', how='outer', indicator=True)
diff_df = df[df['_merge'] != 'both']
# print(diff_df)
drop_index = diff_df.index
# print(drop_index)

# # # #
df = pandas.read_excel(promblem_csv)
good_df = pandas.read_excel(good_csv)

# promblem_csv_1
# print(df.shape)
# drop_index = [66, 92, 122, 166, 191, 246, 247, 251, 320, 351, 381, 406, 426, 456, 508, 509, ]
# # promblem_csv_1
# print(df.shape)
# drop_index = pandas.Index(drop_index, dtype='int64')
# print(drop_index)

df.drop(index=drop_index, inplace=True)
print(df)
df.to_excel(promblem_csv, index=False)

print(f'good_df.shape: {good_df.shape}')
print(f'df.shape: {df.shape}')
print(good_df.iloc[:, 0].equals(df.iloc[:, 0]))

for i in range(len(good_df)):
    if good_df.iloc[i, 0] != df.iloc[i, 0]:
        print(f'First difference at index {i}')
# # df = df[['Date']]
# # good_df = good_df[['Date']]
# # df1 = pandas.concat([df, good_df],  axis=1, join='outer')
# print(df1)
