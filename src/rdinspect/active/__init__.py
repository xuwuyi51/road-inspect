"""M5 主动学习与类别扩展。

- :mod:`rdinspect.active.scoring`  纯函数选样打分（不确定性/错误驱动/多样性，ADR-0007）
- :mod:`rdinspect.active.service`  编排：信号收集、可选 ONNX margin 打分、写优先级与明细、门禁告警

不改代码扩类别：类别注册表在 M1 就是数据驱动的（`rdinspect classes add`），
M5 负责把"新类别的全链路"（注册→标注→冻结→训练→导出）自动化验证并记录下来。
"""

from __future__ import annotations

__all__ = ["scoring", "service"]
