#!/bin/bash

# setup_vcan.sh - 自动检测ACM串口并设置为vcan0
# 使用: sudo ./setup_vcan.sh

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印彩色信息
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查root权限
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "请使用sudo运行此脚本: sudo $0"
        exit 1
    fi
}

# 检查必要的工具
check_dependencies() {
    local missing=()
    
    if ! command -v slcand &> /dev/null; then
        missing+=("can-utils")
    fi
    
    if ! command -v ip &> /dev/null; then
        missing+=("iproute2")
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "缺少必要的依赖: ${missing[*]}"
        log_info "请安装: sudo apt-get install can-utils iproute2"
        exit 1
    fi
}

# 检测ACM设备
detect_acm_devices() {
    local devices=()
    
    # 查找所有ACM设备
    for dev in /dev/ttyACM*; do
        if [ -c "$dev" ]; then
            devices+=("$dev")
        fi
    done
    
    echo "${devices[@]}"
}

# 清理现有的vcan0接口
cleanup_existing() {
    # 检查vcan0是否已存在
    if ip link show vcan0 &> /dev/null; then
        log_info "检测到已存在的vcan0接口，正在清理..."
        
        # 尝试关闭slcand进程
        sudo pkill -f "slcand.*vcan0" 2>/dev/null || true
        
        # 关闭并删除vcan0接口
        sudo ip link set down vcan0 2>/dev/null || true
        sudo ip link delete vcan0 2>/dev/null || true
        
        # 等待一段时间确保清理完成
        sleep 1
        log_success "已清理现有的vcan0接口"
    fi
}

# 主函数
main() {
    log_info "开始设置vcan0接口..."
    
    # 检查权限
    check_root
    
    # 检查依赖
    check_dependencies
    
    # 清理现有接口
    cleanup_existing
    
    # 检测ACM设备
    log_info "正在检测/dev/ttyACM*设备..."
    local devices=($(detect_acm_devices))
    local device_count=${#devices[@]}
    
    if [ $device_count -eq 0 ]; then
        log_error "未找到任何/dev/ttyACM*设备"
        log_info "请检查:"
        log_info "1. USB转CAN设备是否已连接"
        log_info "2. 设备驱动是否正确安装"
        log_info "3. 使用 'lsusb' 命令查看USB设备"
        exit 1
    elif [ $device_count -eq 1 ]; then
        # 只有一个设备的情况
        local device="${devices[0]}"
        log_info "找到设备: $device"
        log_info "正在设置vcan0接口..."
        
        # 设置CAN接口
        log_info "执行: slcand -o -c -s8 $device vcan0"
        if sudo slcand -o -c -s8 "$device" vcan0; then
            log_success "slcand命令执行成功"
        else
            log_error "slcand命令执行失败"
            exit 1
        fi
        
        # 等待slcand初始化
        sleep 2
        
        # 启动接口
        log_info "执行: ip link set up vcan0"
        if sudo ip link set up vcan0; then
            log_success "vcan0接口已启动"
        else
            log_error "启动vcan0接口失败"
            exit 1
        fi
        
        # 显示接口状态
        log_info "\n当前vcan0接口状态:"
        echo "========================================"
        sudo ip -details link show vcan0
        echo "========================================"
        
        log_success "设置完成！vcan0接口已就绪"
        
    else
        # 多个设备的情况
        log_warn "找到多个ACM设备:"
        for i in "${!devices[@]}"; do
            echo "  [$((i+1))] ${devices[$i]}"
        done
        
        log_info "请选择要使用的设备 (1-${#devices[@]})，或输入 'q' 退出:"
        read -r choice
        
        if [[ "$choice" =~ ^[1-9][0-9]*$ ]] && [ "$choice" -le "${#devices[@]}" ]; then
            local device="${devices[$((choice-1))]}"
            log_info "使用设备: $device"
            
            # 询问用户接口名称
            log_info "请输入接口名称 (默认: vcan0):"
            read -r ifname
            ifname=${ifname:-vcan0}
            
            # 清理可能存在的同名接口
            if ip link show "$ifname" &> /dev/null; then
                sudo pkill -f "slcand.*$ifname" 2>/dev/null || true
                sudo ip link set down "$ifname" 2>/dev/null || true
                sudo ip link delete "$ifname" 2>/dev/null || true
                sleep 1
            fi
            
            # 设置接口
            log_info "执行: slcand -o -c -s5 $device $ifname"
            if sudo slcand -o -c -s5 "$device" "$ifname"; then
                log_success "slcand命令执行成功"
            else
                log_error "slcand命令执行失败"
                exit 1
            fi
            
            sleep 2
            
            log_info "执行: ip link set up $ifname"
            if sudo ip link set up "$ifname"; then
                log_success "$ifname接口已启动"
                
                log_info "\n当前$ifname接口状态:"
                echo "========================================"
                sudo ip -details link show "$ifname"
                echo "========================================"
            else
                log_error "启动$ifname接口失败"
                exit 1
            fi
            
        else
            log_info "已取消操作"
            exit 0
        fi
    fi
    
    # 提供使用示例
    log_info "\n使用示例:"
    echo "1. 发送CAN帧: candump vcan0"
    echo "2. 接收CAN帧: cansend vcan0 123#667788"
    echo "3. 查看统计: ip -s link show vcan0"
}

# 捕获Ctrl+C信号
trap 'echo -e "\n${YELLOW}[INFO]${NC} 用户中断"; exit 1' INT

# 运行主函数
main "$@"
