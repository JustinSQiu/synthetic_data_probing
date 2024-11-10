import pandas as pd
from io import StringIO

data = """
<<<<<<< HEAD
| Metric                          |   Wegmann Embeddings |   StyleDistance Embeddings |   StyleDistance Synthetic Only Embeddings |   bert-base-cased Embeddings |   roberta-base Embeddings |   xlm-roberta-base Embeddings |   lisa Embeddings |   luar Embeddings |
|:--------------------------------|---------------------:|---------------------------:|------------------------------------------:|-----------------------------:|--------------------------:|------------------------------:|------------------:|------------------:|
| simplicity hindi                |             0.509798 |                   0.514646 |                                  0.511212 |                     0.565657 |                  0.552626 |                      0.668182 |          0.512828 |          0.513232 |
| simplicity telugu               |             0.501414 |                   0.520606 |                                  0.497778 |                     0.526162 |                  0.511515 |                      0.536162 |          0.505051 |          0.527273 |
| simplicity urdu                 |             0.486263 |                   0.525455 |                                  0.529091 |                     0.531313 |                  0.527677 |                      0.546061 |          0.498182 |          0.511111 |
| simplicity magahi               |             0.497374 |                   0.526465 |                                  0.521818 |                     0.50404  |                  0.562828 |                      0.611919 |          0.523939 |          0.546465 |
| simplicity marathi              |             0.498182 |                   0.550707 |                                  0.528687 |                     0.538687 |                  0.541818 |                      0.606667 |          0.5      |          0.504646 |
| simplicity refined-english      |             0.593535 |                   0.663434 |                                  0.59899  |                     0.901616 |                  0.859394 |                      0.749293 |          0.974141 |          0.842222 |
| simplicity malayalam            |             0.491111 |                   0.513939 |                                  0.516162 |                     0.511515 |                  0.51798  |                      0.530909 |          0.506061 |          0.513131 |
| simplicity punjabi              |             0.516162 |                   0.532323 |                                  0.513939 |                     0.508889 |                  0.547677 |                      0.610707 |          0.51     |          0.571313 |
| simplicity odia                 |             0.494747 |                   0.513939 |                                  0.510909 |                     0.503737 |                  0.496364 |                      0.52404  |          0.496667 |          0.510101 |
| simplicity bengali              |             0.496566 |                   0.533131 |                                  0.502626 |                     0.524949 |                  0.52404  |                      0.568283 |          0.507879 |          0.507071 |
| formal italian                  |             0.714949 |                   0.74     |                                  0.730707 |                     0.649697 |                  0.639192 |                      0.742222 |          0.636768 |          0.704444 |
| formal brazilian_portuguese     |             0.731919 |                   0.816162 |                                  0.846869 |                     0.64     |                  0.641414 |                      0.67697  |          0.627273 |          0.705051 |
| formal french                   |             0.81798  |                   0.91899  |                                  0.864444 |                     0.69798  |                  0.667879 |                      0.787071 |          0.612121 |          0.836566 |
| simplicity Slovene              |             0.706869 |                   0.739394 |                                  0.610303 |                     0.745657 |                  0.797172 |                      0.739394 |          0.631717 |          0.737778 |
| simplicity Japanese             |             0.501919 |                   0.507576 |                                  0.499293 |                     0.503131 |                  0.505758 |                      0.50596  |          0.500303 |          0.502525 |
| simplicity English              |             0.586465 |                   0.605455 |                                  0.588485 |                     0.747475 |                  0.779394 |                      0.674747 |          0.546263 |          0.66404  |
| simplicity German               |             0.749495 |                   0.686869 |                                  0.682626 |                     0.755152 |                  0.746667 |                      0.711515 |          0.549697 |          0.833535 |
| simplicity Russian              |             0.55202  |                   0.597879 |                                  0.521717 |                     0.509192 |                  0.515859 |                      0.563535 |          0.505354 |          0.610202 |
| simplicity Italian              |             0.502424 |                   0.510707 |                                  0.503434 |                     0.545253 |                  0.529495 |                      0.517778 |          0.504848 |          0.512121 |
| simplicity French               |             0.496768 |                   0.497172 |                                  0.498182 |                     0.489899 |                  0.487273 |                      0.495758 |          0.508889 |          0.502222 |
| simplicity Brazilian Portuguese |             0.52     |                   0.514343 |                                  0.511313 |                     0.508283 |                  0.51697  |                      0.509899 |          0.506061 |          0.516162 |
| toxic am                        |             0.553939 |                   0.584848 |                                  0.610303 |                     0.596869 |                  0.607071 |                      0.750505 |          0.625758 |          0.578586 |
| toxic ru                        |             0.571515 |                   0.593535 |                                  0.621414 |                     0.629495 |                  0.669899 |                      0.789697 |          0.594949 |          0.627071 |
| toxic hi                        |             0.566465 |                   0.598788 |                                  0.554545 |                     0.542727 |                  0.699798 |                      0.777172 |          0.633232 |          0.580808 |
| toxic en                        |             0.689495 |                   0.688687 |                                  0.676768 |                     0.839596 |                  0.85798  |                      0.754747 |          0.906263 |          0.885657 |
| toxic uk                        |             0.535859 |                   0.535455 |                                  0.573636 |                     0.576061 |                  0.582525 |                      0.629192 |          0.56697  |          0.512222 |
| toxic ar                        |             0.544242 |                   0.530101 |                                  0.515354 |                     0.537172 |                  0.585657 |                      0.640202 |          0.596364 |          0.538384 |
| toxic es                        |             0.546061 |                   0.527879 |                                  0.522222 |                     0.585657 |                  0.664444 |                      0.675354 |          0.583434 |          0.539596 |
| toxic zh                        |             0.709495 |                   0.651111 |                                  0.662626 |                     0.711818 |                  0.816162 |                      0.858788 |          0.499495 |          0.667677 |
| toxic de                        |             0.519798 |                   0.518788 |                                  0.531111 |                     0.547475 |                  0.615556 |                      0.696768 |          0.621616 |          0.545657 |
| average                         |             0.573428 |                   0.591946 |                                  0.578552 |                     0.599172 |                  0.618936 |                      0.648316 |          0.576404 |          0.604896 |
"""

data = data.replace("|:--------------------------------|---------------------:|---------------------------:|------------------------------------------:|-----------------------------:|--------------------------:|------------------------------:|------------------:|------------------:|\n", "")
=======
| Metric           |   Wegmann Embeddings |   StyleDistance Embeddings |   StyleDistance Synthetic Only Embeddings |   bert-base-cased Embeddings |   roberta-base Embeddings |   xlm-roberta-base Embeddings |   lisa Embeddings |   luar Embeddings |
| positivity ur    |            0.240808  |                  0.203434  |                                 0.18      |                  0.10404     |                0.102626   |                    0.0755556  |       0.280101    |         0.204444  |
| positivity bn    |            0.270101  |                  0.125455  |                                 0.141212  |                  0.0456566   |                0.147273   |                    0.0434343  |       0.226667    |         0.130101  |
| positivity mr    |            0.193333  |                  0.178384  |                                 0.120202  |                  0.11899     |                0.059596   |                    0.030303   |       0.174141    |         0.143636  |
| positivity te    |            0.391717  |                  0.337374  |                                 0.277172  |                  0.297273    |                0.213535   |                    0.197778   |       0.289798    |         0.346465  |
| positivity mag   |            0.0949495 |                  0.0810101 |                                 0.137374  |                  0.0620202   |                0.0418182  |                    0.0761616  |       0.130505    |         0.0858586 |
| positivity hi    |            0.109697  |                  0.0953535 |                                 0.0882828 |                  0.116364    |                0.020404   |                    0.0373737  |       0.14202     |         0.124444  |
| positivity en    |            0.206465  |                  0.19798   |                                 0.0848485 |                  0.00565657  |                0.00929293 |                    0.0349495  |       0.185657    |         0.0294949 |
| positivity pa    |            0.179394  |                  0.153737  |                                 0.134545  |                  0.161212    |                0.0529293  |                    0.0632323  |       0.16798     |         0.135758  |
| positivity or    |            0.273131  |                  0.185051  |                                 0.185455  |                  0.239596    |                0.187475   |                    0.0840404  |       0.242323    |         0.195152  |
| positivity ml    |            0.324848  |                  0.278788  |                                 0.223232  |                  0.233636    |                0.230505   |                    0.0967677  |       0.269192    |         0.237374  |
| formality fr     |            0.698788  |                  0.805253  |                                 0.71596   |                  0.16        |                0.16101    |                    0.160202   |       0.0620202   |         0.404242  |
| formality it     |            0.63798   |                  0.633535  |                                 0.545051  |                  0.225455    |                0.172121   |                    0.178586   |       0.0967677   |         0.450909  |
| formality pt-br  |            0.559596  |                  0.572323  |                                 0.50303   |                  0.21596     |                0.192323   |                    0.153131   |       0.107677    |         0.320808  |
| simplicity ru    |            0.263434  |                  0.244242  |                                 0.28      |                  0.0763636   |                0.243232   |                    0.0658586  |       0.152121    |         0.120606  |
| simplicity fr    |            0.289293  |                  0.328687  |                                 0.332525  |                  0.223232    |                0.234141   |                    0.222222   |       0.119192    |         0.143434  |
| simplicity ja    |            0.0923232 |                  0.0505051 |                                 0.113333  |                  0.0792929   |                0.0260606  |                    0.0147475  |       0.476364    |         0.0822222 |
| simplicity en    |            0.255758  |                  0.317172  |                                 0.269697  |                  0.0010101   |                0.0129293  |                    0.0488889  |       0.00222222  |         0.0125253 |
| simplicity it    |            0.209091  |                  0.145051  |                                 0.257576  |                  0.0315152   |                0.0339394  |                    0.0769697  |       0.0254545   |         0.0981818 |
| simplicity de    |            0.227071  |                  0.0581818 |                                 0.0648485 |                  0.00262626  |                0.00484848 |                    0.00161616 |       0.00242424  |         0.0658586 |
| simplicity sl    |            0.434141  |                  0.425859  |                                 0.431717  |                  0.510101    |                0.429899   |                    0.460808   |       0.391919    |         0.36404   |
| simplicity pt-br |            0.100606  |                  0.0731313 |                                 0.0612121 |                  0.0549495   |                0.0474747  |                    0.0430303  |       0.0327273   |         0.0466667 |
| toxicity zh      |            0.0505051 |                  0.0161616 |                                 0.0793939 |                  0.0416162   |                0.0218182  |                    0.0444444  |       0.0692929   |         0.0212121 |
| toxicity ar      |            0.0545455 |                  0.0369697 |                                 0.0620202 |                  0.0242424   |                0.0169697  |                    0.0179798  |       0.100404    |         0.0349495 |
| toxicity ru      |            0.178182  |                  0.157576  |                                 0.17899   |                  0.0355556   |                0.0864646  |                    0.134141   |       0.0913131   |         0.170909  |
| toxicity hi      |            0.150505  |                  0.0945455 |                                 0.120202  |                  0.0420202   |                0.107677   |                    0.0917172  |       0.154646    |         0.101212  |
| toxicity en      |            0.560606  |                  0.478384  |                                 0.615152  |                  0.0284848   |                0.0290909  |                    0.0852525  |       0.0806061   |         0.230505  |
| toxicity de      |            0.0135354 |                  0.0242424 |                                 0.0739394 |                  0.000606061 |                0.00020202 |                    0.00868687 |       0.000606061 |         0.0010101 |
| toxicity es      |            0.25697   |                  0.197778  |                                 0.213939  |                  0.0884848   |                0.0822222  |                    0.129293   |       0.0456566   |         0.04      |
| toxicity uk      |            0.0745455 |                  0.0452525 |                                 0.0490909 |                  0.00262626  |                0.0179798  |                    0.0373737  |       0.0242424   |         0.0274747 |
| toxicity am      |            0.349899  |                  0.290505  |                                 0.285859  |                  0.238485    |                0.242626   |                    0.24303    |       0.208889    |         0.225253  |
| average          |            0.258061  |                  0.227731  |                                 0.227529  |                  0.115569    |                0.107616   |                    0.0985859  |       0.145098    |         0.153158  |
"""

>>>>>>> 167d839 (crosslingual dataset)
df = pd.read_csv(StringIO(data), sep="|", engine="python").iloc[:, 1:-1].apply(lambda x: x.str.strip() if x.dtype == "object" else x)

for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col])
<<<<<<< HEAD
output_file = 'stel.xlsx'
=======
output_file = 'stel-or-content_processed.xlsx'
>>>>>>> 167d839 (crosslingual dataset)
df.to_excel(output_file, index=False)

print(f"Data has been saved to {output_file}")