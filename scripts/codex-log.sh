#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# codex-log.sh —— 把「自上次归档以来的全部代码变更」归档为一次可回看的记录
#
# 用法：
#   bash scripts/codex-log.sh "给登录页加验证码"
#   bash scripts/codex-log.sh "重构订单状态机" --base HEAD~3   # 手动指定基线
#
# 产出（默认目录 codex-logs/）：
#   codex-logs/2026-09-01-给登录页加验证码/
#     ├── README.md       # 四段式说明骨架，交给 Codex / 自己补全
#     ├── changes.patch   # 完整 diff（含新增文件），可 git apply 回放
#     └── files.txt       # 变更文件清单 + 增删行数统计
#   codex-logs/INDEX.md   # 总索引，每次自动追加一行
#
# 基线机制：
#   用 refs/codex-logs/baseline 这个 ref 记住「上次归档时的工作区快照」。
#   快照通过 `git stash create` 生成，不会改动你的工作区，也不会提交任何东西。
#   因此无论 Codex 的改动是「已提交」还是「还躺在工作区」，下次都能完整抓到，且不重复。
# ------------------------------------------------------------------------------
set -euo pipefail

LOG_DIR="${CODEX_LOG_DIR:-codex-logs}"
BASELINE_REF="refs/codex-logs/baseline"
MANUAL_BASE=""

# ---------- 参数 ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) MANUAL_BASE="${2:?--base 需要一个 commit / ref}"; shift 2 ;;
    --dir)  LOG_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) break ;;
  esac
done

TITLE="${*:-}"
if [[ -z "$TITLE" ]]; then
  echo "❌ 请给这次变更起个标题，例如： bash scripts/codex-log.sh \"给登录页加验证码\"" >&2
  exit 1
fi

# ---------- 前置检查 ----------
command -v git >/dev/null || { echo "❌ 未找到 git" >&2; exit 1; }
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$REPO_ROOT" ]] || { echo "❌ 当前不在 git 仓库内" >&2; exit 1; }
cd "$REPO_ROOT"

if ! git rev-parse --verify -q HEAD >/dev/null; then
  echo "❌ 仓库还没有任何 commit，请先做一次初始提交再归档" >&2
  exit 1
fi

# ---------- 标题转 slug（保留中文/英文/数字，空格转连字符，其余标点剔除） ----------
slug="$(printf '%s' "$TITLE" | tr '[:upper:]' '[:lower:]')"
slug="${slug// /-}"
slug="${slug//[,\.\/\!\?\(\)\[\]\{\}\<\>:\;\'\"\`~@\#\$%\^\&\*+=\|，。、；：！？（）【】《》“”‘’—]}"
slug="$(printf '%s' "$slug" | sed -e 's/--*/-/g' -e 's/^-*//' -e 's/-*$//' | cut -c1-40)"
[[ -n "$slug" ]] || slug="untitled"

DATE="$(date +%Y-%m-%d)"
ENTRY_DIR="$LOG_DIR/${DATE}-${slug}"
if [[ -d "$ENTRY_DIR" ]]; then
  n=2
  while [[ -d "${ENTRY_DIR}-${n}" ]]; do n=$((n+1)); done
  ENTRY_DIR="${ENTRY_DIR}-${n}"
fi
mkdir -p "$ENTRY_DIR"

# ---------- 计算基线 ----------
if [[ -n "$MANUAL_BASE" ]]; then
  BASE="$(git rev-parse --verify "$MANUAL_BASE^{commit}")"
else
  BASE="$(git rev-parse --verify -q "$BASELINE_REF" 2>/dev/null || true)"
  if [[ -z "$BASE" ]]; then
    BASE="$(git rev-parse HEAD)"
    echo "ℹ️  首次归档，基线取当前 HEAD（$BASE）"
  fi
fi

# ---------- 生成 diff ----------
# 关键：用一份临时索引副本执行 add -N，让「新增但未跟踪的文件」也进入 diff，
# 同时完全不污染用户真实的 git index / 暂存区。
TMP_INDEX="$(mktemp "${TMPDIR:-/tmp}/codex-log-index.XXXXXX")"
cleanup() { rm -f "$TMP_INDEX"; }
trap cleanup EXIT

REAL_INDEX="$(git rev-parse --git-path index)"
[[ -f "$REAL_INDEX" ]] && cp "$REAL_INDEX" "$TMP_INDEX"
export GIT_INDEX_FILE="$TMP_INDEX"

# 排除归档目录自身，避免日志自我污染
EXCLUDE=":(exclude,top)$LOG_DIR"
git add -A -- . "$EXCLUDE" >/dev/null 2>&1 || true
git update-index --intent-to-add -- $(git ls-files --others --exclude-standard -- . "$EXCLUDE") 2>/dev/null || true

set +e
git diff --binary --find-renames "$BASE" -- . "$EXCLUDE" > "$ENTRY_DIR/changes.patch"
DIFF_RC=$?
set -e
unset GIT_INDEX_FILE

if [[ $DIFF_RC -ge 2 ]]; then
  echo "❌ git diff 执行失败（退出码 $DIFF_RC）" >&2
  exit 1
fi

if [[ ! -s "$ENTRY_DIR/changes.patch" ]]; then
  rm -rf "$ENTRY_DIR"
  echo "⚠️  基线 $BASE 以来没有任何代码变更，未生成归档。"
  echo "   如果改动已经提交，试试手动指定基线： bash scripts/codex-log.sh \"$TITLE\" --base HEAD~5"
  exit 0
fi

# ---------- 变更清单与统计（直接从 patch 解析，保证与实际 diff 完全一致） ----------
NUMSTAT="$(git apply --numstat "$ENTRY_DIR/changes.patch" 2>/dev/null || true)"
FILES_CHANGED="$(printf '%s\n' "$NUMSTAT" | awk 'NF{n++} END{print n+0}')"
INSERTIONS="$(printf '%s\n' "$NUMSTAT" | awk 'NF && $1!="-"{a+=$1} END{print a+0}')"
DELETIONS="$(printf '%s\n' "$NUMSTAT" | awk 'NF && $2!="-"{a+=$2} END{print a+0}')"
UNTRACKED="$(git ls-files --others --exclude-standard -- . "$EXCLUDE" | sed 's/^/- /')"

{
  echo "# 变更清单"
  echo
  echo "基线 \`${BASE}\` → 当前工作区 ｜ ${FILES_CHANGED} 个文件，+${INSERTIONS} / -${DELETIONS}"
  echo
  echo '```'
  printf '%s\n' "$NUMSTAT"
  echo '```'
  if [[ -n "$UNTRACKED" ]]; then
    echo
    echo "## 本次新增（归档前未纳入 git 跟踪）的文件"
    echo
    printf '%s\n' "$UNTRACKED"
  fi
} > "$ENTRY_DIR/files.txt"

# ---------- 生成四段式说明骨架 ----------
TEMPLATE_FILE=""
for cand in "$LOG_DIR/TEMPLATE.md" "$(dirname "$0")/../templates/entry.md" "$(dirname "$0")/templates/entry.md"; do
  if [[ -f "$cand" ]]; then TEMPLATE_FILE="$cand"; break; fi
done

render_entry() {
  sed -e "s/{{TITLE}}/${TITLE//\//\\/}/g" \
      -e "s/{{DATE}}/$(date +%Y-%m-%d\ %H:%M)/g" \
      -e "s/{{BASE}}/${BASE}/g" \
      -e "s/{{FILES}}/${FILES_CHANGED}/g" \
      -e "s/{{INSERTIONS}}/${INSERTIONS}/g" \
      -e "s/{{DELETIONS}}/${DELETIONS}/g" \
      -e "s/{{DIRNAME}}/$(basename "$ENTRY_DIR")/g"
}

if [[ -n "$TEMPLATE_FILE" ]]; then
  render_entry < "$TEMPLATE_FILE" > "$ENTRY_DIR/README.md"
else
  # 脚本自带兜底模板，保证单文件也能独立运行
  render_entry <<'TPL' > "$ENTRY_DIR/README.md"
# {{TITLE}}

- **日期**：{{DATE}}
- **基线 commit**：`{{BASE}}`
- **规模**：{{FILES}} 个文件，+{{INSERTIONS}} / -{{DELETIONS}}
- **完整 diff**：[`changes.patch`](./changes.patch) ｜ [变更清单](./files.txt)

## 1. 需求：要解决什么问题

**原始需求 / prompt：**

```text
（粘贴当时的指令）
```

**验收标准：**

- [ ] …

## 2. 改了什么

1. **`path/to/file`**：……

**关键 diff：**

```diff
（从 changes.patch 摘最有代表性的一段）
```

**验证方式：**（命令 + 结果）

## 3. 为什么这么改

- **候选方案**：A …（放弃，因为 …）／ B …（采用，因为 …）
- **权衡点**：
- **风险与后续**：

## 4. 我学到了什么

1. **技术点**：
2. **工程判断**：
3. **一句话总结**：
TPL
fi

# ---------- 维护总索引 ----------
INDEX_FILE="$LOG_DIR/INDEX.md"
if [[ ! -f "$INDEX_FILE" ]]; then
  {
    echo "# Codex 变更归档索引"
    echo
    echo "| 日期 | 变更 | 文件数 | +/- | 目录 |"
    echo "| --- | --- | --- | --- | --- |"
  } > "$INDEX_FILE"
fi
echo "| $DATE | $TITLE | $FILES_CHANGED | +$INSERTIONS / -$DELETIONS | [$(basename "$ENTRY_DIR")](./$(basename "$ENTRY_DIR")/README.md) |" >> "$INDEX_FILE"

# ---------- 推进基线（快照当前工作区，不提交、不改工作区、不碰真实索引） ----------
# 用临时索引把「已跟踪改动 + 未跟踪新文件」写成一个 tree，再 commit-tree 成基线 commit，
# 由 refs/codex-logs/baseline 持有，因此不会被 gc 回收。
BASE_INDEX="$(mktemp "${TMPDIR:-/tmp}/codex-log-base.XXXXXX")"
[[ -f "$REAL_INDEX" ]] && cp "$REAL_INDEX" "$BASE_INDEX"
export GIT_INDEX_FILE="$BASE_INDEX"
git add -A -- . "$EXCLUDE" >/dev/null 2>&1 || true
NEW_TREE="$(git write-tree 2>/dev/null || true)"
unset GIT_INDEX_FILE
rm -f "$BASE_INDEX"

if [[ -n "$NEW_TREE" ]]; then
  NEW_BASE="$(git -c user.name='codex-log' -c user.email='codex-log@local' \
      commit-tree "$NEW_TREE" -p "$(git rev-parse HEAD)" -m "codex-log baseline snapshot" 2>/dev/null || true)"
  if [[ -n "$NEW_BASE" ]]; then
    git update-ref "$BASELINE_REF" "$NEW_BASE"
  fi
fi

# ---------- 输出 ----------
echo ""
echo "✅ 已归档到 $ENTRY_DIR/"
echo "   changes.patch  : $(wc -l < "$ENTRY_DIR/changes.patch" | tr -d ' ') 行 diff（${FILES_CHANGED} 个文件，+${INSERTIONS} / -${DELETIONS}）"
echo "   README.md      : 四段式说明骨架，请补全「为什么这么改」和「学到什么」"
echo "   files.txt      : 变更清单"
echo "   基线已推进到 $(git rev-parse --short "$BASELINE_REF" 2>/dev/null || echo '?')"
echo ""
echo "   回放这次改动： git apply --check $ENTRY_DIR/changes.patch"
