"""姓名 combobox：一个输入框完成「筛选 → 点选 / 新增」。

替代早期的双控件组合（筛选 text_input + selectbox）和更早的两步流程
（selectbox 选「➕ 新增...」再填新姓名 text_input）。

状态约定（前缀 p = f"{照片序号}_{人脸序号}"，页面传入）：
  sb_p  已确认的姓名（"" = 未选；可以是人物库中不存在的新名字）
  cb_p  输入框当前文本。cb 与 sb 不同 → 展开建议列表（打字中），
        相同 → 收起（已确认）。不需要额外的展开标志位。
  rd_p  st.radio 的选中值。点选后下一轮渲染开始时被消费并清空。

交互：
  - 输入拼音首字母/全拼/中文/英文即筛选（回车或失焦提交后生效，
    具体节奏取决于 Streamlit 版本对 text_input 的提交时机）
  - 回车时：输入原样等于某个已有姓名 → 自动确认。
    **唯一匹配不自动确认**——拼音首字母会撞车（"lzg" 只命中
    "刘志高"，但用户可能想新增一个完全不同的人），自动确认会把
    建议列表收起，用户来不及点「➕ 新增」。有匹配时必须把选择权
    留给用户，且无论有没有匹配，列表里都带「➕ 新增「x」」
  - 列表行点选即确认；「➕ 新增「x」」把输入原样作为新姓名确认
  - 「✕ 清除选择」清空已选（仅在有已选姓名时出现）

关于「渲染时消费点选」：radio 的 on_change 回调在 AppTest（无头测试）
里不会因 set_value 触发，但在真实浏览器里正常。把提交逻辑放在渲染
开头统一处理，两条路径行为一致，也不依赖回调闭包。
"""

import re
from typing import List, Optional

import streamlit as st

from core.name_search import filter_persons

# 建议列表最多显示几行（超出显示「…还有 N 人」提示，避免列表无限长）
_MAX_SHOW = 8

_CLEAR_LABEL = "✕ 清除选择"
_ADD_RE = re.compile(r"^➕ 新增「(.+)」$")


def resolve_pick(picked: str, person_names: List[str]) -> Optional[str]:
    """把 radio 的选项标签解析回真实姓名。

    返回 None 表示该标签已过期（人物被删等），调用方应忽略。
    """
    if picked == _CLEAR_LABEL:
        return ""
    m = _ADD_RE.match(picked)
    if m:
        return m.group(1).strip() or None
    if picked in person_names:
        return picked
    return None


def auto_select_name(person_names: List[str], typed: str) -> Optional[str]:
    """回车/失焦时的快捷确认：原样等于已有姓名 → 直接确认。

    只有"原样等于某个已有姓名"才自动确认（无歧义）。**唯一匹配
    不自动确认**：拼音首字母会撞车（"lzg" 唯一命中"刘志高"，但用户
    可能想新增一个完全不同的人），自动确认会收起建议列表，用户
    来不及点「➕ 新增」。

    返回 None 表示不自动确认（展开建议列表让用户点选）。
    """
    typed = (typed or "").strip()
    if not typed:
        return None
    if typed in person_names:
        return typed
    return None


def render_name_combobox(prefix: str, person_names: List[str],
                         label: str = "姓名") -> str:
    """渲染一个姓名 combobox，返回已确认的姓名（"" = 未选）。

    prefix       同一页面内唯一的前缀（如 "0_1"）
    person_names 人物库中的姓名列表（建议按照片数排序，常用的在前）
    """
    sb_key = "sb_" + prefix
    cb_key = "cb_" + prefix
    rd_key = "rd_" + prefix

    def _commit(name: str):
        st.session_state[sb_key] = name
        st.session_state[cb_key] = name

    # ── 1) 消费上一次交互留下的 radio 点选（在渲染输入框之前处理，
    #       框里直接显示选中的姓名，不需要额外 rerun） ──
    picked = st.session_state.get(rd_key)
    if picked is not None:
        st.session_state[rd_key] = None
        name = resolve_pick(picked, person_names)
        if name is not None:
            _commit(name)

    confirmed = (st.session_state.get(sb_key) or "").strip()

    # ── 2) 输入框（值即显示内容；已确认的姓名直接显示在框里） ──
    def _on_type():
        # 回车/失焦：原样等于已有姓名，或只有一个匹配 → 直接确认
        name = auto_select_name(person_names,
                                st.session_state.get(cb_key) or "")
        if name is not None:
            _commit(name)

    st.text_input(
        label,
        key=cb_key,
        placeholder="🔍 拼音首字母/姓名，回车筛选",
        label_visibility="collapsed",
        on_change=_on_type,
    )

    query = (st.session_state.get(cb_key) or "").strip()
    if query == confirmed:
        return st.session_state.get(sb_key, "")

    # ── 3) 展开建议列表 ──
    matches = filter_persons(person_names, query)
    options = list(matches[:_MAX_SHOW])
    if query and query not in person_names:
        # 没有同名人物 → 允许把输入原样作为新姓名
        options.append("➕ 新增「{}」".format(query))
    if confirmed:
        options.append(_CLEAR_LABEL)

    if not options:
        # 人物库为空且没输入：提示直接输入
        st.caption("人物库为空，输入姓名后回车即可新增")
        return st.session_state.get(sb_key, "")

    if matches:
        cap = "匹配 {} 人，点选确认".format(len(matches))
        if len(matches) > _MAX_SHOW:
            cap += "（显示前 {} 个）".format(_MAX_SHOW)
        if query and query not in person_names:
            # 首字母会撞车：有匹配也必须保留新增入口
            cap += "；要新增的人不在里面？点最下方「➕ 新增」"
        st.caption(cap)
    else:
        st.caption("未匹配「{}」，点「➕ 新增」直接采用".format(query))

    st.radio(
        "建议",
        options=options,
        key=rd_key,
        index=None,
        label_visibility="collapsed",
    )

    return st.session_state.get(sb_key, "")
