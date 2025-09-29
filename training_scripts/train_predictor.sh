#!/bin/bash

# 预测器训练启动脚本
# 使用BurstGPT数据集训练多任务预测器，支持wandb日志记录

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查环境
check_environment() {
    log_info "检查运行环境..."

    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi

    # 检查CUDA
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "nvidia-smi 未找到，将使用CPU训练"
        DEVICE="cpu"
    else
        GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)
        if [ "$GPU_COUNT" -gt 0 ]; then
            log_success "检测到 $GPU_COUNT 个GPU"
            DEVICE="cuda"
        else
            log_warning "未检测到GPU，将使用CPU训练"
            DEVICE="cpu"
        fi
    fi

    # 检查wandb
    if ! python3 -c "import wandb" &> /dev/null; then
        log_warning "wandb 未安装，将跳过wandb日志记录"
        USE_WANDB=false
    else
        USE_WANDB=true
        log_success "wandb 已安装"
    fi

    log_info "设备: $DEVICE"
    log_info "Wandb: $USE_WANDB"
}

# 设置路径
setup_paths() {
    # 获取脚本所在目录
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

    # 设置数据路径（使用分钟级时间序列数据）
    TRAIN_DATA="dataset/split/burstgpt_predictor_train_minute.csv"
    VAL_DATA="dataset/split/burstgpt_predictor_val_minute.csv"
    CONFIG_FILE="$SCRIPT_DIR/configs/predictor_config.json"
    OUTPUT_DIR="$PROJECT_ROOT/training_output"

    # 检查数据文件是否存在
    if [ ! -f "$TRAIN_DATA" ]; then
        log_error "训练数据不存在: $TRAIN_DATA"
        log_info "请先运行以下脚本:"
        log_info "1. python3 training_scripts/convert_to_dense_timeseries.py --convert-original"
        log_info "2. python3 training_scripts/compress_timeseries.py"
        log_info "3. python3 training_scripts/split_minute_dataset.py"
        exit 1
    fi

    if [ ! -f "$VAL_DATA" ]; then
        log_error "验证数据不存在: $VAL_DATA"
        exit 1
    fi

    log_success "数据文件检查通过"
    log_info "训练数据: $TRAIN_DATA"
    log_info "验证数据: $VAL_DATA"
    log_info "配置文件: $CONFIG_FILE"
}

# 创建输出目录
create_output_dir() {
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/logs"
    mkdir -p "$OUTPUT_DIR/models"
    mkdir -p "$OUTPUT_DIR/plots"

    log_success "输出目录创建完成: $OUTPUT_DIR"
}

# 设置wandb
setup_wandb() {
    if [ "$USE_WANDB" = true ]; then
        log_info "设置wandb..."

        # 检查wandb是否已登录
        if ! python3 -c "import wandb; wandb.login()" 2>/dev/null; then
            log_warning "wandb 未登录，请先运行: wandb login"
            read -p "是否跳过wandb登录？(y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                USE_WANDB=false
            else
                log_info "请运行以下命令登录wandb:"
                log_info "wandb login"
                exit 1
            fi
        fi
    fi
}

# 训练函数
run_training() {
    log_info "开始训练预测器..."

    # 设置Python路径
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

    # 设置wandb项目（如果使用）
    if [ "$USE_WANDB" = true ]; then
        WANDB_ARGS="--use_wandb --wandb_project smartllm-serve --wandb_run_name predictor_$(date +%Y%m%d_%H%M%S)"
    else
        WANDB_ARGS=""
    fi

    # 运行训练
    cd "$PROJECT_ROOT"

    python3 training_scripts/train_predictor.py \
        --train_data "$TRAIN_DATA" \
        --val_data "$VAL_DATA" \
        --config "$CONFIG_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE" \
        --epochs 150 \
        --batch_size 128 \
        --sequence_length 120 \
        --prediction_horizon 3 \
        $WANDB_ARGS \
        2>&1 | tee "$OUTPUT_DIR/logs/training.log"

    # 检查训练是否成功
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "训练完成！"
    else
        log_error "训练失败，请检查日志: $OUTPUT_DIR/logs/training.log"
        exit 1
    fi
}

# 生成训练报告
generate_report() {
    log_info "生成训练报告..."

    REPORT_FILE="$OUTPUT_DIR/training_report.txt"

    {
        echo "预测器训练报告"
        echo "=================="
        echo "训练时间: $(date)"
        echo "设备: $DEVICE"
        echo "Wandb: $USE_WANDB"
        echo ""
        echo "数据集信息:"
        echo "训练数据: $(wc -l < "$TRAIN_DATA") 行"
        echo "验证数据: $(wc -l < "$VAL_DATA") 行"
        echo ""
        echo "输出目录:"
        echo "模型文件: $OUTPUT_DIR/models/"
        echo "日志文件: $OUTPUT_DIR/logs/"
        echo "图表文件: $OUTPUT_DIR/plots/"
        echo ""
        echo "训练日志:"
        echo "-----------"
        tail -50 "$OUTPUT_DIR/logs/training.log"
    } > "$REPORT_FILE"

    log_success "训练报告生成完成: $REPORT_FILE"
}

# 主函数
main() {
    echo "=============================================="
    echo "      SmartLLMServe 预测器训练脚本"
    echo "=============================================="
    echo ""

    # 检查环境
    check_environment
    echo ""

    # 设置路径
    setup_paths
    echo ""

    # 创建输出目录
    create_output_dir
    echo ""

    # 设置wandb
    setup_wandb
    echo ""

    # 运行训练
    run_training
    echo ""

    # 生成报告
    generate_report

    echo ""
    log_success "所有任务完成！"
    echo ""
    echo "训练结果位置:"
    echo "- 输出目录: $OUTPUT_DIR"
    echo "- 训练日志: $OUTPUT_DIR/logs/training.log"
    echo "- 训练报告: $OUTPUT_DIR/training_report.txt"
    echo ""
    echo "下一步:"
    echo "1. 查看训练结果: ls -la $OUTPUT_DIR/"
    echo "2. 如果使用了wandb，访问项目页面查看详细日志"
    echo "3. 评估模型性能: python3 training_scripts/evaluate_predictor.py"
}

# 捕获Ctrl+C
trap 'log_warning "训练被中断"; exit 130' INT

# 运行主函数
main "$@"