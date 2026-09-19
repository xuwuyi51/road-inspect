"""M3 训练闭环：微调 runner、评估与门禁、ONNX 导出包。

模块分工（都不在导入期引入 torch/ultralytics，缺依赖时由上层转 501）：

- :mod:`rdinspect.train.matching`   纯函数检测匹配指标（IoU/混淆矩阵/大小桶召回/AP），无第三方依赖
- :mod:`rdinspect.train.runner`     ultralytics 微调（日志流、断点、flipud 硬约束、run_key 幂等）
- :mod:`rdinspect.train.evaluate`   冻结数据集上的评估与失败样例导出
- :mod:`rdinspect.train.gate`       模型门禁与状态机 candidate→validated→production
- :mod:`rdinspect.train.export_onnx` ONNX 导出与导出包（labels/preprocess/manifest/parity）
- :mod:`rdinspect.train.service`    进程内后台训练任务与运行详情
- :mod:`rdinspect.train.compat`     受限容器兼容（/dev/shm 不可用时退化为串行扫描）

CLI/API 入口分别是 ``rdinspect train|runs|model`` 与 ``/api/train/*``、``/api/models/*``（见 docs/11-training-guide.md）。
"""

from __future__ import annotations

__all__ = ["matching"]
