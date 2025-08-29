## 现在你已经完全理解到了当前的任务的数据结构
帮我重新写可视化的代码，我希望是一个可视化界面，一个是可以选择数据集的下拉框，这里是train路径的下面的路径名，另一个是可以选择变量名的下拉框，变量的名字是excel的第一行的名字。然后纵轴是值，横轴是随着TIME的值的变化。并且可以增加多个变量显示在同一个表，支持多条曲线同时显示。增加变量相当于增加一行可选项，就是选择数据集，然后选择变量名。

## 你已经基本完成了上面的任务，但是有下面的bug：
我发现加载分钟级别的变量的时候会有下面的问题`use_container_width` will be removed after             │
│   2025-12-31.                                                                                         │
│                                                                                                       │
│   For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use         │
│   `width='content'`.                                                                                  │
│   2025-08-29 10:47:25.486 Please replace `use_container_width` with `width`.                          │
│                                                                                                       │
│   `use_container_width` will be removed after 2025-12-31.                                             │
│                                                                                                       │
│   For `use_container_width=True`, use `width='stretch'`. For `use_container_width=False`, use         │
│   `width='content'`. 

