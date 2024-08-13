和baseline的对比结果有有意差。

```
a = [4,3,4,7,4,2,4,4,7,0,4,4,4,7,0,5,4,5,7,1,4,5,5,7,2]
b = [6,5,6,7,4,7,6,3,7,3,7,6,6,7,2,7,6,5,7,1,6,6,6,7,4]
from scipy import stats
stats.wilcoxon(a, b)
### >>> WilcoxonResult(statistic=2.0, pvalue=0.0002690922274892695)
```
