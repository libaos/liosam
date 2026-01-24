#!/usr/bin/env python3
"""
日志管理工具
用于查看、清理和分析日志文件
"""
import argparse
from pathlib import Path
import json
from datetime import datetime
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import create_log_structure_info

def list_logs(log_dir, model_type=None, script_type=None):
    """列出日志文件"""
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"日志目录不存在: {log_dir}")
        return
    
    print("📋 日志文件列表")
    print("=" * 80)
    
    # 遍历日志目录
    for model_dir in sorted(log_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        
        # 过滤模型类型
        if model_type and model_type != model_dir.name:
            continue
        
        print(f"\n📁 {model_dir.name}/")
        
        for script_dir in sorted(model_dir.iterdir()):
            if not script_dir.is_dir():
                continue
            
            # 过滤脚本类型
            if script_type and script_type != script_dir.name:
                continue
            
            log_files = list(script_dir.glob("*.log"))
            if log_files:
                print(f"  📂 {script_dir.name}/ ({len(log_files)} 个日志)")
                
                for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True):
                    # 获取文件信息
                    stat = log_file.stat()
                    size_mb = stat.st_size / (1024 * 1024)
                    mtime = datetime.fromtimestamp(stat.st_mtime)
                    
                    print(f"    📄 {log_file.name}")
                    print(f"       大小: {size_mb:.2f} MB, 修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

def clean_logs(log_dir, days=7, dry_run=True):
    """清理旧日志文件"""
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"日志目录不存在: {log_dir}")
        return
    
    from datetime import timedelta
    cutoff_time = datetime.now() - timedelta(days=days)
    
    print(f"🧹 清理 {days} 天前的日志文件")
    print(f"截止时间: {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    deleted_count = 0
    total_size = 0
    
    # 遍历所有日志文件
    for log_file in log_dir.rglob("*.log"):
        stat = log_file.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)
        
        if mtime < cutoff_time:
            size_mb = stat.st_size / (1024 * 1024)
            total_size += stat.st_size
            
            print(f"{'[DRY RUN] ' if dry_run else ''}删除: {log_file.relative_to(log_dir)}")
            print(f"  大小: {size_mb:.2f} MB, 修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if not dry_run:
                log_file.unlink()
            
            deleted_count += 1
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"\n{'预计' if dry_run else '实际'}删除 {deleted_count} 个文件，释放 {total_size_mb:.2f} MB 空间")
    
    if dry_run:
        print("\n💡 使用 --no-dry-run 参数实际执行删除操作")

def analyze_logs(log_dir, model_type=None):
    """分析日志文件"""
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        print(f"日志目录不存在: {log_dir}")
        return
    
    print("📊 日志分析报告")
    print("=" * 80)
    
    stats = {}
    
    # 遍历日志目录
    for model_dir in log_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        # 过滤模型类型
        if model_type and model_type != model_dir.name:
            continue
        
        model_name = model_dir.name
        stats[model_name] = {
            'training': {'count': 0, 'size': 0},
            'evaluation': {'count': 0, 'size': 0},
            'prediction': {'count': 0, 'size': 0}
        }
        
        for script_dir in model_dir.iterdir():
            if not script_dir.is_dir():
                continue
            
            script_type = script_dir.name
            if script_type not in stats[model_name]:
                stats[model_name][script_type] = {'count': 0, 'size': 0}
            
            for log_file in script_dir.glob("*.log"):
                stats[model_name][script_type]['count'] += 1
                stats[model_name][script_type]['size'] += log_file.stat().st_size
    
    # 显示统计结果
    for model_name, model_stats in stats.items():
        print(f"\n📁 {model_name}")
        print("-" * 40)
        
        total_count = 0
        total_size = 0
        
        for script_type, script_stats in model_stats.items():
            count = script_stats['count']
            size_mb = script_stats['size'] / (1024 * 1024)
            
            if count > 0:
                print(f"  {script_type:12}: {count:3d} 个文件, {size_mb:6.2f} MB")
                total_count += count
                total_size += script_stats['size']
        
        total_size_mb = total_size / (1024 * 1024)
        print(f"  {'总计':12}: {total_count:3d} 个文件, {total_size_mb:6.2f} MB")

def show_structure():
    """显示日志目录结构"""
    print("📁 日志目录结构说明")
    print("=" * 80)
    print(create_log_structure_info())

def main():
    parser = argparse.ArgumentParser(description='日志管理工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 列出日志
    list_parser = subparsers.add_parser('list', help='列出日志文件')
    list_parser.add_argument('--model', type=str, help='过滤模型类型')
    list_parser.add_argument('--script', type=str, help='过滤脚本类型')
    list_parser.add_argument('--log-dir', type=str, default='outputs/logs', help='日志目录')
    
    # 清理日志
    clean_parser = subparsers.add_parser('clean', help='清理旧日志文件')
    clean_parser.add_argument('--days', type=int, default=7, help='保留天数')
    clean_parser.add_argument('--no-dry-run', action='store_true', help='实际执行删除')
    clean_parser.add_argument('--log-dir', type=str, default='outputs/logs', help='日志目录')
    
    # 分析日志
    analyze_parser = subparsers.add_parser('analyze', help='分析日志文件')
    analyze_parser.add_argument('--model', type=str, help='过滤模型类型')
    analyze_parser.add_argument('--log-dir', type=str, default='outputs/logs', help='日志目录')
    
    # 显示结构
    subparsers.add_parser('structure', help='显示日志目录结构')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    
    if args.command == 'list':
        log_dir = project_root / args.log_dir
        list_logs(log_dir, args.model, args.script)
    
    elif args.command == 'clean':
        log_dir = project_root / args.log_dir
        clean_logs(log_dir, args.days, not args.no_dry_run)
    
    elif args.command == 'analyze':
        log_dir = project_root / args.log_dir
        analyze_logs(log_dir, args.model)
    
    elif args.command == 'structure':
        show_structure()

if __name__ == "__main__":
    main()
