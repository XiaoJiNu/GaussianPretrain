# 继承基础配置
_base_ = ['./gp_0.075_convnext.py']

# 覆盖数据配置 - 使用unified格式
data_root = 'data/nuscenes/'

# 修改数据集为 mini，使用unified格式的pkl文件
data = dict(
    samples_per_gpu=1,  # 单卡批次大小
    workers_per_gpu=2,  # 数据加载线程
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_unified_infos_train.pkl',  # 使用unified格式
    ),
    val=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_unified_infos_val.pkl',  # 使用unified格式
    ),
    test=dict(
        data_root=data_root,
        ann_file=data_root + 'nuscenes_unified_infos_val.pkl',  # 使用unified格式
    )
)

# 减少训练轮数（mini 数据集）
total_epochs = 6
evaluation = dict(interval=6)

# 日志配置
log_config = dict(
    interval=10,  # 每10个iter打印一次
    hooks=[
        dict(type='TextLoggerHook'),
    ]
)

# 工作目录
work_dir = './work_dirs/gp_0.075_convnext_mini'
