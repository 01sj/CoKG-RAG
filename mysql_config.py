"""
MySQL 数据库配置文件
请根据你的实际 MySQL 服务器配置修改以下参数
"""

# MySQL 连接配置
MYSQL_CONFIG = {
    'host': '10.61.2.49',        # MySQL 服务器地址（远程服务器）
    'port': 3306,              # MySQL 端口（默认 3306）
    'user': 'root',            # MySQL 用户名
    'password': '123',         # MySQL 密码
    'charset': 'utf8mb4',      # 字符集
    'connect_timeout': 10,     # 连接超时时间（秒）
    'read_timeout': 30,        # 读取超时时间（秒）
    'write_timeout': 30,       # 写入超时时间（秒）
    'autocommit': True,        # 自动提交
    'sql_mode': 'TRADITIONAL'  # SQL模式
}

# 如果你使用远程 MySQL 服务器，请修改为：
# MYSQL_CONFIG = {
#     'host': 'your_mysql_server_ip',  # 例如: '192.168.1.100'
#     'port': 3306,
#     'user': 'your_username',
#     'password': 'your_password',
#     'charset': 'utf8mb4'
# }
