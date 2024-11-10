from io import StringIO

import pandas as pd

data = """
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

df = pd.read_csv(StringIO(data), sep="|", engine="python").iloc[:, 1:-1].apply(lambda x: x.str.strip() if x.dtype == "object" else x)

for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col])
output_file = 'stel-or-content_processed.xlsx'
df.to_excel(output_file, index=False)

print(f"Data has been saved to {output_file}")
