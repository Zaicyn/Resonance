#!/bin/bash
# run_comparison.sh - Simple, reliable timing for Arch/KDE

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  V16 vs Beta Resonance Comparison     ${NC}"
echo -e "${BLUE}========================================${NC}\n"

RESULTS_DIR="resonance_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR
echo -e "${YELLOW}Results will be saved to: ${RESULTS_DIR}/${NC}\n"

# Simple timing function - just use date +%s%N directly
time_command() {
    local start=$(date +%s%N)
    "$@"
    local end=$(date +%s%N)
    echo $(( ($end - $start) / 1000000 ))
}

echo -e "${GREEN}Compiling configurations...${NC}"

echo "  Baseline 18..."
start=$(date +%s%N)
nvcc -O2 -std=c++17 -arch=sm_75 -o $RESULTS_DIR/compare_baseline compare_strategies_resonance.cu -lm 2>/dev/null
end=$(date +%s%N)
t1=$(( ($end - $start) / 1000000 ))
echo "    ${CYAN}${t1}ms${NC}"

echo "  Baseline 24..."
start=$(date +%s%N)
nvcc -O2 -std=c++17 -arch=sm_75 -DV16_SLAB_SBS_PER_WARP=24 -o $RESULTS_DIR/compare_24 compare_strategies_resonance.cu -lm 2>/dev/null
end=$(date +%s%N)
t2=$(( ($end - $start) / 1000000 ))
echo "    ${CYAN}${t2}ms${NC}"

echo "  Resonance 27..."
start=$(date +%s%N)
nvcc -O2 -std=c++17 -arch=sm_75 -DUSE_RESONANCE -o $RESULTS_DIR/compare_resonance compare_strategies_resonance.cu -lm 2>/dev/null
end=$(date +%s%N)
t3=$(( ($end - $start) / 1000000 ))
echo "    ${CYAN}${t3}ms${NC}"

echo -e "${GREEN}Compilation complete.\n${NC}"

echo -e "${YELLOW}Running Baseline 18 (SBS_PER_WARP=18)...${NC}"
start=$(date +%s%N)
$RESULTS_DIR/compare_baseline > $RESULTS_DIR/baseline_18.txt
end=$(date +%s%N)
t18=$(( ($end - $start) / 1000000 ))
echo -e "  ${CYAN}${t18}ms${NC}"

echo -e "${YELLOW}Running Baseline 24 (SBS_PER_WARP=24)...${NC}"
start=$(date +%s%N)
$RESULTS_DIR/compare_24 > $RESULTS_DIR/baseline_24.txt
end=$(date +%s%N)
t24=$(( ($end - $start) / 1000000 ))
echo -e "  ${CYAN}${t24}ms${NC}"

echo -e "${YELLOW}Running Resonance 27 (SBS_PER_WARP=27)...${NC}"
start=$(date +%s%N)
$RESULTS_DIR/compare_resonance > $RESULTS_DIR/resonance_27.txt
end=$(date +%s%N)
t27=$(( ($end - $start) / 1000000 ))
echo -e "  ${CYAN}${t27}ms${NC}\n"

# ============================================================================
# Generate summary
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}              SUMMARY                   ${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Function to extract key data
extract_resonance() {
    local file=$1
    local label=$2
    local run_time=$3

    echo -e "${YELLOW}$label (Total: ${run_time}ms)${NC}"
    echo "Depth | Succ% | Rungs  | Fallbacks | Cycles | Imbal"
    echo "------|-------|--------|-----------|--------|------"
    grep -E "^[[:space:]]*[0-9]{2,4}" "$file" | while read line; do
        depth=$(echo "$line" | awk '{print $1}')
        beta_pct=$(echo "$line" | awk '{print $5}' | tr -d '%')
        imbalance=$(echo "$line" | awk '{print $7}' | tr -d '%')
        rungs=$(echo "$line" | awk '{print $9}')
        fallbacks=$(echo "$line" | awk '{print $11}')
        cycles=$(echo "$line" | awk '{print $13}')

        printf "  %3s  | %5s%% | %6s | %9s | %6s | %5s%%\n" \
               "$depth" "$beta_pct" "$rungs" "$fallbacks" "$cycles" "$imbalance"
    done
    echo ""
}

extract_resonance "$RESULTS_DIR/baseline_18.txt" "BASELINE 18" "$t18"
extract_resonance "$RESULTS_DIR/baseline_24.txt" "BASELINE 24" "$t24"
extract_resonance "$RESULTS_DIR/resonance_27.txt" "RESONANCE 27" "$t27"

# ============================================================================
# 54Hz Analysis
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}        54-Hz Resonance Analysis       ${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${YELLOW}At depth = 54 (2×27 harmonic):${NC}"
echo "Config       | Success | Rungs    | Fallbacks | Cycles | Imbalance"
echo "-------------|---------|----------|-----------|--------|----------"

grep -E "^[[:space:]]*54" "$RESULTS_DIR/resonance_27.txt" | while read line; do
    beta_pct=$(echo "$line" | awk '{print $5}' | tr -d '%')
    rungs=$(echo "$line" | awk '{print $9}')
    fallbacks=$(echo "$line" | awk '{print $11}')
    cycles=$(echo "$line" | awk '{print $13}')
    imbalance=$(echo "$line" | awk '{print $7}' | tr -d '%')
    printf "Resonance 27 |  %5s%% | %7s | %9s | %6s |   %5s%%\n" \
           "$beta_pct" "$rungs" "$fallbacks" "$cycles" "$imbalance"
done

grep -E "^[[:space:]]*36" "$RESULTS_DIR/baseline_18.txt" | head -1 | while read line; do
    beta_pct=$(echo "$line" | awk '{print $5}' | tr -d '%')
    rungs=$(echo "$line" | awk '{print $9}')
    fallbacks=$(echo "$line" | awk '{print $11}')
    cycles=$(echo "$line" | awk '{print $13}')
    imbalance=$(echo "$line" | awk '{print $7}' | tr -d '%')
    printf "Baseline 18  |  %5s%% | %7s | %9s | %6s |   %5s%%\n" \
           "$beta_pct" "$rungs" "$fallbacks" "$cycles" "$imbalance"
done

grep -E "^[[:space:]]*144" "$RESULTS_DIR/baseline_24.txt" | head -1 | while read line; do
    beta_pct=$(echo "$line" | awk '{print $5}' | tr -d '%')
    rungs=$(echo "$line" | awk '{print $9}')
    fallbacks=$(echo "$line" | awk '{print $11}')
    cycles=$(echo "$line" | awk '{print $13}')
    imbalance=$(echo "$line" | awk '{print $7}' | tr -d '%')
    printf "Baseline 24  |  %5s%% | %7s | %9s | %6s |   %5s%%\n" \
           "$beta_pct" "$rungs" "$fallbacks" "$cycles" "$imbalance"
done

echo ""

# ============================================================================
# Performance Summary with Timing
# ============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}         Performance Summary            ${NC}"
echo -e "${BLUE}========================================${NC}\n"

total_rungs_18=$(grep -E "^[[:space:]]*[0-9]" "$RESULTS_DIR/baseline_18.txt" | awk '{sum += $9} END {print sum}')
total_rungs_24=$(grep -E "^[[:space:]]*[0-9]" "$RESULTS_DIR/baseline_24.txt" | awk '{sum += $9} END {print sum}')
total_rungs_27=$(grep -E "^[[:space:]]*[0-9]" "$RESULTS_DIR/resonance_27.txt" | awk '{sum += $9} END {print sum}')

avg_cycles_18=$(grep -E "^[[:space:]]*[0-9]" "$RESULTS_DIR/baseline_18.txt" | awk '{sum += $13; count++} END {if(count>0) printf "%.0f", sum/count; else print "0"}')
avg_cycles_24=$(grep -E "^[[:space:]]*[0-9]" "$RESULTS_DIR/baseline_24.txt" | awk '{sum += $13; count++} END {if(count>0) printf "%.0f", sum/count; else print "0"}')
avg_cycles_27=$(grep -E "^[[:space:]]*[0-9]" "$RESULTS_DIR/resonance_27.txt" | awk '{sum += $13; count++} END {if(count>0) printf "%.0f", sum/count; else print "0"}')

total_ops_estimate=320000

# Use awk for floating point (more reliable than bc)
ms_per_op_18=$(awk "BEGIN {printf \"%.3f\", $t18 / $total_ops_estimate}")
ms_per_op_24=$(awk "BEGIN {printf \"%.3f\", $t24 / $total_ops_estimate}")
ms_per_op_27=$(awk "BEGIN {printf \"%.3f\", $t27 / $total_ops_estimate}")

us_per_op_18=$(awk "BEGIN {printf \"%.1f\", ($t18 / $total_ops_estimate) * 1000}")
us_per_op_24=$(awk "BEGIN {printf \"%.1f\", ($t24 / $total_ops_estimate) * 1000}")
us_per_op_27=$(awk "BEGIN {printf \"%.1f\", ($t27 / $total_ops_estimate) * 1000}")

printf "%-15s | %10s | %10s | %12s | %12s | %10s | %10s\n" "Configuration" "Total Rungs" "Avg Cycles" "Total Time" "Est. Ops" "ms/op" "μs/op"
printf "%-15s | %10s | %10s | %12s | %12s | %10s | %10s\n" "---------------" "-----------" "----------" "------------" "-------------" "----------" "----------"
printf "%-15s | %10s | %10s | %12s ms | %12s | %8s | %8s\n" \
       "Baseline 18" "$total_rungs_18" "$avg_cycles_18" "$t18" "$total_ops_estimate" "${ms_per_op_18}ms" "${us_per_op_18}μs"
printf "%-15s | %10s | %10s | %12s ms | %12s | %8s | %8s\n" \
       "Baseline 24" "$total_rungs_24" "$avg_cycles_24" "$t24" "$total_ops_estimate" "${ms_per_op_24}ms" "${us_per_op_24}μs"
printf "%-15s | %10s | %10s | %12s ms | %12s | %8s | %8s\n" \
       "Resonance 27" "$total_rungs_27" "$avg_cycles_27" "$t27" "$total_ops_estimate" "${ms_per_op_27}ms" "${us_per_op_27}μs"

echo ""
echo -e "${CYAN}Peak 54Hz Timing (16 blocks × 256 threads):${NC}"
peak_line=$(grep -E "^[[:space:]]*54" "$RESULTS_DIR/resonance_27.txt" | tail -1)
if [ -n "$peak_line" ]; then
    cycles=$(echo "$peak_line" | awk '{print $13}')
    total_cycles_all=$(grep -E "^[[:space:]]*[0-9]" "$RESULTS_DIR/resonance_27.txt" | awk '{sum += $13} END {print sum}')
    if [ "$total_cycles_all" -gt 0 ]; then
        ratio=$(awk "BEGIN {printf \"%.6f\", $cycles / $total_cycles_all}")
        ms_54=$(awk "BEGIN {printf \"%.1f\", $t27 * $ratio}")
        us_54=$(awk "BEGIN {printf \"%.1f\", ($t27 * $ratio) * 1000}")
        echo "  At 54Hz depth: ~${ms_54}ms total, ~${us_54}μs per operation (based on ${cycles} cycles)"
    fi
fi

echo -e "\n${GREEN}All results saved to: ${RESULTS_DIR}/${NC}"
echo -e "${YELLOW}To view raw data: less ${RESULTS_DIR}/resonance_27.txt${NC}"
echo -e "${YELLOW}To extract just the 54Hz data: grep '^[[:space:]]*54' ${RESULTS_DIR}/resonance_27.txt${NC}\n"
