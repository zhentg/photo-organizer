"""姓名搜索：拼音首字母 / 全拼 / 中文子串 匹配。

供人脸标注页下拉框筛选使用：
- 输入 "zs"      → 张三（拼音首字母）
- 输入 "lisi"    → 李四（全拼前缀）
- 输入 "李"      → 李四（中文子串）
- 输入 "an"      → Anne（英文名子串）

匹配大小写不敏感。name_keys 带进程内缓存（人数很少，首次计算后零成本）。
依赖 pypinyin（已列入 requirements.txt）；未安装时退化为原文匹配，
功能降级但不会报错。
"""

from typing import List, Sequence

_keys_cache = {}  # name -> (initials, full)


def name_keys(name: str) -> tuple:
    """返回 (拼音首字母, 全拼)，如 "李四" -> ("ls", "lisi")。"""
    if name in _keys_cache:
        return _keys_cache[name]
    try:
        from pypinyin import lazy_pinyin, Style
        initials = "".join(lazy_pinyin(name, style=Style.FIRST_LETTER)).lower()
        full = "".join(lazy_pinyin(name)).lower()
    except ImportError:  # 没装 pypinyin：退化为原文，仅支持子串匹配
        initials = full = name.lower()
    _keys_cache[name] = (initials, full)
    return _keys_cache[name]


def filter_persons(names: Sequence[str], query: str) -> List[str]:
    """按 query 过滤姓名列表；query 为空时返回原列表的拷贝。

    匹配规则（任一命中即入选）：
    1. 拼音首字母包含 query        （"s" → 所有首字母串含 s 的人）
    2. 全拼以 query 开头           （"lis" → 李四）
    3. 姓名原文包含 query          （"李" → 李四；"an" → Anne）
    """
    q = (query or "").strip().lower()
    if not q:
        return list(names)
    matched = []
    for name in names:
        initials, full = name_keys(name)
        if q in name.lower() or q in initials or full.startswith(q):
            matched.append(name)
    return matched
