#!/bin/bash
# 对比所有 naive attention 版本的正确性和性能（表格输出）
cd "$(dirname "$0")"

VERSIONS="${@:-3 4 5}"
VALID_VERSIONS=()

# 编译并收集结果
declare -A RESULTS  # key: "v,N" -> time
declare -A CORRECT  # key: "v" -> pass count
BENCHMARKS=("256,64" "512,64" "1024,64" "16384,64")

for v in $VERSIONS; do
    src="kernels/naive_attention"
    [ "$v" -ne 0 ] && src="${src}_${v}"
    if [ ! -f "${src}.cu" ]; then
        continue
    fi
    make clean -s 2>/dev/null && make test_naive V=$v -s 2>&1
    if [ $? -ne 0 ]; then
        echo "V=$v 编译失败！"
        continue
    fi
    VALID_VERSIONS+=($v)
    output=$(./test_naive 2>&1)

    # 统计 PASS 数
    pass=$(echo "$output" | grep -c "✓ PASS")
    fail=$(echo "$output" | grep -c "✗ FAIL")
    CORRECT[$v]="${pass}/$((pass+fail))"

    # 提取性能数据
    while IFS= read -r line; do
        n=$(echo "$line" | grep -oP 'N=\K[0-9]+')
        ms=$(echo "$line" | grep -oP '[0-9]+\.[0-9]+ ms' | grep -oP '[0-9]+\.[0-9]+')
        if [ -n "$n" ] && [ -n "$ms" ]; then
            RESULTS["$v,$n"]=$ms
        fi
    done <<< "$(echo "$output" | grep '朴素')"
done

if [ ${#VALID_VERSIONS[@]} -eq 0 ]; then
    echo "没有可用的版本"
    exit 1
fi

# 输出正确性
echo "=============================="
echo "  正确性"
echo "=============================="
for v in "${VALID_VERSIONS[@]}"; do
    echo "  V=$v: ${CORRECT[$v]}"
done

# 输出性能表格
echo ""
echo "=============================="
echo "  性能对比 (ms)"
echo "=============================="

# 表头
header=$(printf "%-10s" "N")
for v in "${VALID_VERSIONS[@]}"; do
    header+=$(printf "%-14s" "V=$v")
done
echo "$header"
echo "--------------------------------------------------------------"

# 每行
for bench in "${BENCHMARKS[@]}"; do
    n=${bench%%,*}
    row=$(printf "%-10s" "$n")
    for v in "${VALID_VERSIONS[@]}"; do
        val=${RESULTS["$v,$n"]:-"-"}
        row+=$(printf "%-14s" "${val} ms")
    done
    echo "$row"
done
echo ""
