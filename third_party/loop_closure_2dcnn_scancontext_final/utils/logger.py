"""
日志工具 - 支持按模型分类的日志管理
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

class Logger:
    """简单的日志器类"""

    def __init__(self, log_file=None, level=logging.INFO):
        """
        初始化日志器

        参数:
            log_file (str): 日志文件路径
            level: 日志级别
        """
        self.logger = logging.getLogger('TemporalSystem')
        self.logger.setLevel(level)

        # 清除已有的处理器
        self.logger.handlers.clear()

        # 创建格式器
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 文件处理器
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message):
        """记录信息日志"""
        self.logger.info(message)

    def warning(self, message):
        """记录警告日志"""
        self.logger.warning(message)

    def error(self, message):
        """记录错误日志"""
        self.logger.error(message)

    def debug(self, message):
        """记录调试日志"""
        self.logger.debug(message)

def setup_logger(name, log_file=None, level=logging.INFO):
    """设置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有的处理器
    logger.handlers.clear()

    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def setup_model_logger(model_type, script_type, timestamp=None, project_root=None):
    """
    为特定模型设置日志器

    参数:
        model_type (str): 模型类型 (sc_ring_cnn, sc_standard_cnn, simple_cnn, simple_cnn_lite)
        script_type (str): 脚本类型 (training, evaluation, prediction)
        timestamp (str): 时间戳，如果为None则自动生成
        project_root (Path): 项目根目录，如果为None则自动推断

    返回:
        logger: 配置好的日志器
        log_file: 日志文件路径
    """
    if timestamp is None:
        timestamp = get_timestamp()

    if project_root is None:
        # 自动推断项目根目录
        current_file = Path(__file__)
        project_root = current_file.parent.parent

    # 创建按模型分类的日志目录结构
    log_base_dir = project_root / "outputs" / "logs"
    model_log_dir = log_base_dir / model_type / script_type
    model_log_dir.mkdir(parents=True, exist_ok=True)

    # 生成日志文件名
    log_filename = f"{script_type}_{model_type}_{timestamp}.log"
    log_file = model_log_dir / log_filename

    # 创建日志器名称
    logger_name = f"{script_type}_{model_type}"

    # 设置日志器
    logger = setup_logger(logger_name, log_file)

    return logger, log_file

def get_timestamp():
    """获取时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_model_type_from_script(script_path):
    """
    从脚本路径推断模型类型

    参数:
        script_path (str or Path): 脚本文件路径

    返回:
        str: 模型类型
    """
    script_path = Path(script_path)
    script_name = script_path.stem

    # 模型类型映射
    model_mapping = {
        'train_sc_ring_cnn': 'sc_ring_cnn',
        'train_sc_standard_cnn': 'sc_standard_cnn',
        'train_simple_cnn': 'simple_cnn',
        'train_simple_cnn_lite': 'simple_cnn_lite',
        'evaluate_model': 'general',
        'loop_closure_detector': 'general'
    }

    return model_mapping.get(script_name, 'general')

def get_script_type_from_path(script_path):
    """
    从脚本路径推断脚本类型

    参数:
        script_path (str or Path): 脚本文件路径

    返回:
        str: 脚本类型
    """
    script_path = Path(script_path)

    # 从路径中推断脚本类型
    if 'training' in script_path.parts:
        return 'training'
    elif 'evaluation' in script_path.parts:
        return 'evaluation'
    elif 'prediction' in script_path.parts:
        return 'prediction'
    else:
        return 'general'

def create_log_structure_info():
    """
    创建日志结构说明

    返回:
        str: 日志结构说明
    """
    return """
日志目录结构:
outputs/logs/
├── sc_ring_cnn/
│   ├── training/
│   │   └── training_sc_ring_cnn_20250802_151234.log
│   ├── evaluation/
│   │   └── evaluation_sc_ring_cnn_20250802_151234.log
│   └── prediction/
│       └── prediction_sc_ring_cnn_20250802_151234.log
├── sc_standard_cnn/
│   ├── training/
│   ├── evaluation/
│   └── prediction/
├── simple_cnn/
│   ├── training/
│   ├── evaluation/
│   └── prediction/
├── simple_cnn_lite/
│   ├── training/
│   ├── evaluation/
│   └── prediction/
└── general/
    ├── training/
    ├── evaluation/
    └── prediction/
"""
