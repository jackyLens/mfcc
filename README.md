# mfcc
This part is programed to detect the snoring event.
设有m个用户组,第i组有ni个用户。

对第i组用户,模型输出为:

$y_{i1},...,y_{ini}$

组内用户相似度打分项为:

$L_{sim_i} = \sum\limits_{j=1}^{n_i}\sum\limits_{k=1,k\neq j}^{n_i}\frac{1}{n_i(n_i-1)}s(y_{ij},y_{ik})$

其中s(.)为相似度函数,可以为余弦相似度等。

组间差异性约束项为:

$L_{diff} = \sum\limits_{i=1}^{m-1}\sum\limits_{j=i+1}^{m}d(y_i,y_j)$

其中d(.)用于评价两组用户输出的差异性。

则最终损失函数为:

$L = \sum\limits_{i=1}^{m}\lambda_iL_{sim_i} + \lambda_dL_{diff}$

通过调节比重超参$\lambda$,控制个体差异的建模。
