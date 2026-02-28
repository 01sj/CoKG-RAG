"""
MySQL 调试工具
用于诊断和解决MySQL连接和数据库创建问题
"""

import pymysql
import time
import sys
from mysql_config import MYSQL_CONFIG

def test_mysql_connection():
    """测试MySQL连接，提供详细的诊断信息"""
    print("=== MySQL连接诊断 ===")
    
    # 1. 测试基本连接
    print("1. 测试基本连接...")
    try:
        connection = pymysql.connect(**MYSQL_CONFIG)
        print("✓ 基本连接成功")
        connection.close()
    except Exception as e:
        print(f"✗ 基本连接失败: {e}")
        return False
    
    # 2. 测试连接超时设置
    print("2. 测试连接超时设置...")
    try:
        config_with_timeout = MYSQL_CONFIG.copy()
        config_with_timeout['connect_timeout'] = 5
        start_time = time.time()
        connection = pymysql.connect(**config_with_timeout)
        end_time = time.time()
        print(f"✓ 连接超时测试成功，耗时: {end_time - start_time:.2f}秒")
        connection.close()
    except Exception as e:
        print(f"✗ 连接超时测试失败: {e}")
    
    # 3. 测试数据库权限
    print("3. 测试数据库权限...")
    try:
        connection = pymysql.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()
        
        # 检查当前用户权限
        cursor.execute("SELECT USER(), VERSION()")
        user_info = cursor.fetchone()
        print(f"✓ 当前用户: {user_info[0]}, MySQL版本: {user_info[1]}")
        
        # 检查创建数据库权限
        cursor.execute("SHOW GRANTS FOR CURRENT_USER()")
        grants = cursor.fetchall()
        print("✓ 用户权限:")
        for grant in grants:
            print(f"  - {grant[0]}")
        
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"✗ 权限检查失败: {e}")
    
    return True

def create_database_with_retry(db_name, max_retries=3):
    """带重试机制的数据库创建函数"""
    print(f"=== 创建数据库: {db_name} ===")
    
    for attempt in range(max_retries):
        print(f"尝试 {attempt + 1}/{max_retries}...")
        
        try:
            # 建立连接
            connection = pymysql.connect(**MYSQL_CONFIG)
            cursor = connection.cursor()
            
            # 设置会话超时
            cursor.execute("SET SESSION innodb_lock_wait_timeout = 10")
            cursor.execute("SET SESSION net_read_timeout = 30")
            cursor.execute("SET SESSION net_write_timeout = 30")
            
            # 检查数据库是否已存在
            cursor.execute("SHOW DATABASES LIKE %s", (db_name,))
            if cursor.fetchone():
                print(f"✓ 数据库 '{db_name}' 已存在")
                cursor.close()
                connection.close()
                return True
            
            # 创建数据库
            print(f"正在创建数据库 '{db_name}'...")
            start_time = time.time()
            
            # 使用更安全的创建语句
            create_sql = f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            cursor.execute(create_sql)
            
            end_time = time.time()
            print(f"✓ 数据库创建成功，耗时: {end_time - start_time:.2f}秒")
            
            # 验证创建结果
            cursor.execute("SHOW DATABASES LIKE %s", (db_name,))
            if cursor.fetchone():
                print(f"✓ 数据库 '{db_name}' 创建验证成功")
            
            cursor.close()
            connection.close()
            return True
            
        except pymysql.Error as e:
            print(f"✗ MySQL错误 (尝试 {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print("等待3秒后重试...")
                time.sleep(3)
            else:
                print("所有重试都失败了")
                return False
        except Exception as e:
            print(f"✗ 未知错误 (尝试 {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print("等待3秒后重试...")
                time.sleep(3)
            else:
                print("所有重试都失败了")
                return False
    
    return False

def check_mysql_server_status():
    """检查MySQL服务器状态"""
    print("=== MySQL服务器状态检查 ===")
    
    try:
        connection = pymysql.connect(**MYSQL_CONFIG)
        cursor = connection.cursor()
        
        # 检查服务器状态
        cursor.execute("SHOW STATUS LIKE 'Threads_connected'")
        threads_connected = cursor.fetchone()
        print(f"当前连接数: {threads_connected[1]}")
        
        cursor.execute("SHOW STATUS LIKE 'Threads_running'")
        threads_running = cursor.fetchone()
        print(f"运行中的线程数: {threads_running[1]}")
        
        cursor.execute("SHOW STATUS LIKE 'Max_used_connections'")
        max_used = cursor.fetchone()
        print(f"历史最大连接数: {max_used[1]}")
        
        # 检查进程列表
        cursor.execute("SHOW PROCESSLIST")
        processes = cursor.fetchall()
        print(f"当前进程数: {len(processes)}")
        
        if len(processes) > 10:
            print("⚠️  警告: 进程数较多，可能影响性能")
        
        cursor.close()
        connection.close()
        return True
        
    except Exception as e:
        print(f"✗ 服务器状态检查失败: {e}")
        return False

def main():
    """主函数 - 运行完整的MySQL诊断"""
    print("开始MySQL诊断...")
    print("=" * 50)
    
    # 1. 连接测试
    if not test_mysql_connection():
        print("连接测试失败，请检查网络和MySQL服务")
        return
    
    # 2. 服务器状态检查
    check_mysql_server_status()
    
    # 3. 尝试创建测试数据库
    test_db_name = "test_leanrag_db"
    if create_database_with_retry(test_db_name):
        print(f"✓ 测试数据库 '{test_db_name}' 创建成功")
        
        # 清理测试数据库
        try:
            connection = pymysql.connect(**MYSQL_CONFIG)
            cursor = connection.cursor()
            cursor.execute(f"DROP DATABASE IF EXISTS `{test_db_name}`")
            print(f"✓ 测试数据库 '{test_db_name}' 已清理")
            cursor.close()
            connection.close()
        except Exception as e:
            print(f"⚠️  清理测试数据库失败: {e}")
    else:
        print(f"✗ 测试数据库 '{test_db_name}' 创建失败")
    
    print("=" * 50)
    print("MySQL诊断完成")

if __name__ == "__main__":
    main()
