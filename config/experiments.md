# 耦合训练
0. exp0 全参数回归，float， double, finished

# 可学习参数范围
1. exp1 角度为30度，float，sliced,
2. exp2 角度为45度，float，sliced,
3. exp3 角度为60度，float，unsliced,

# 透视变换参数

5. exp4 不分割样本透视参数, finished, 15
6. exp5 分割样本透视参数, finished, 13
7. exp9 dropout 0.25, finished, 16

# 角度分类
6. exp6 共享参数, finished, 35
7. exp7 不共享参数, finished, 34
8. exp8 角度回归, finished, fail
9. exp10 共享参数分割样本, finished, 38
10. exp11, 加入与不加入度量矩阵, finished, 33
11. exp12(5), exp13(13): 不同核大小影响
12. exp14: 初始化影响
13. exp15 global和全连接性能, 全连接方式, finished, overfit
14. exp16 最后一层角度卷积核不变, finished, 34

# 整体效果展示
13. exp6 分割样本整体性能 finished
14. exp10 不分割样本整体性能 finished

# 数据整理
1,2,3 表格图表结果
4,5,9 图，fill_with
6,7,10,11,15,16 表格包括分类正确率，残差均值方差，残差分布，top3正确率，top5正确率
7 10 11 15 16