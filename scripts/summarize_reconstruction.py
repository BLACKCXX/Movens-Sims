#!/usr/bin/env python3
"""Build the three-row delivery entry, retaining every failed or incomplete sample."""
import argparse
import csv
import html
import io
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.util import ROOT,read,write,sha,now

def delivery_note(run,accept):
    """Human-readable values come from the final independent disk check."""
    result=read(run/'result.json');review=read(run/'final_review.json')['review'];camera=accept['camera_verified_from_xml'];ep=result['episode']
    text=[f'# Episode {ep} 最终产物与验收',
          '这是静态场景重建及原生软层静置验收，不含策略推理或训练。所有失败项保留。',
          '[真实/初始/最终对照](comparison.png) · [独立落盘复验](reacceptance/final/acceptance.json) · [Terra 最终原始审阅](final_review.json) · [完整参数](parameters_final.json) · [XML 核对相机](reacceptance/final/camera_verified_from_xml.json)',
          '## 可重载场景',
          '[静态 XML](delivery/static/portable/scene.xml)、[静态状态](delivery/static/portable/scene.state.npz)；[动态 XML](delivery/dynamic/portable/scene.xml)、[动态状态](delivery/dynamic/portable/scene.state.npz)。每个 portable 目录包含自身全部依赖资产；动态版本保留原生 flex 和节点状态。',
          '用 MuJoCo 加载 XML，再将同目录 scene.state.npz 的 qpos、qvel、time 写回 MjData，调用 mj_forward；这与独立验证器 load_state() 相同。静态显示网格不能代替动态版本的物理测试。',
          '## 参数和尺寸',
          '苹果初始化倍率 1.0 相对于最后训练使用的已烘焙资产；额外苹果补偿、额外桌面抬高均为零。下面尺寸为最终重载模型中物体坐标系视觉包围尺寸，单位 mm；碰撞和世界包围尺寸另见复验 JSON。']
    table=['| 物体 | 后续整体倍率 | X / Y / Z (mm) | 世界位置 (m) |','|---|---:|---|---|']
    for label in ('apple','plate'):
        g=accept['geometry'][label];table.append(f"| {label} | {g['absolute_scale']:.9f} | "+' / '.join(f'{x*1000:.3f}' for x in g['dimensions_body_m'])+' | '+', '.join(f'{x:.6f}' for x in g['xyz_m'])+' |')
    text.append('\n'.join(table))
    text+=['',f"相机世界位置 (m)：{camera['xyz_m']}；垂直 FOV {camera['fovy_deg']:.6f}°，640×480，fy={camera['fy_px']:.6f}px；XML 导出俯角 {camera['down_angle_deg']:.6f}°。完整旋转矩阵见相机 JSON。这是受限假设，未做实物标定。",'## 静置终态和过程峰值',
           '积分参考是精确加载状态、第一步积分之前；固定 4000 步×0.25 ms=1 s。终态门槛 10 mm / 10° / 2 mm；过程峰值和末段速度仅作诊断。没有积分中清零、固定物体或移动参考起点。']
    table=['| 物体 | 终态位移 mm | 终态转角 ° | 终态穿透 mm | 峰值位移 mm | 峰值转角 ° | 峰值穿透 mm |','|---|---:|---:|---:|---:|---:|---:|']
    for label,s in accept['physical']['objects'].items():
        values=(s['translation_final_m']*1000,s['rotation_final_deg'],s['penetration_final_m']*1000,s['translation_max_m']*1000,s['rotation_max_deg'],s['penetration_max_m']*1000)
        table.append('| '+label+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
    text.append('\n'.join(table))
    text+=['','[静置视频](reacceptance/final/rest/settle_diagnostic.mp4) · [精确终态图](reacceptance/final/rest/rest_terminal.png) · [时间序列 CSV](diagnostic_timeseries.csv) · [原生接触、距离和力](reacceptance/final/rest/contact_timeline.json) · [软层/速度记录](reacceptance/final/rest/diagnostic_timeline.json)',
           '## 最终分类审阅']
    for item in review['category_results']:text+=['',f"**{item['category']}：{item['verdict']}** — {item['evidence']}"]
    text+=['','## 实际修复与门槛变化',
           '[逐项阈值和改善归因](improvement_attribution.json)同时保留原候选指标、新候选指标、旧候选仅换门槛后的判定。旧苹果仅改终态口径及新阈值仍失败；本轮姿态与布局修复才把终态转角和位移降到新门槛内。当前苹果按原 5 mm / 5° 仍失败；不能把通过全部归因于物理修复。盘子按样本分别有终态口径或放宽阈值的影响，原始数值均保留。',
           '## 审计与用量',
           f"实际 Terra 请求 {result['terra_actual_http_requests']}/20，模型 {result['model']}；包括失败和重试。最终请求只读，已绑定同一份 XML、状态、资产和图像。独立复验没有追加模型请求。",'[API 响应及 token 用量](api_usage.json) · [完整不可退款账本](budget_ledger.json) · [统一计算/渲染/MCP 开销](cost_report.json) · [文件和请求绑定审计](delivery_audit.json) · [真实 MCP 操作记录](mcp_tool_calls.jsonl) · [候选和操作依据](mcp_plans/)']
    (run/'DELIVERY.md').write_text('\n\n'.join(text)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=read(ROOT/a.config);root=ROOT/c['run_root'];rows=[];cards=[]
    for ep in [0,1,2]:
        name=f'episode_{ep:06}';run=root/name;r=read(run/'result.json') if (run/'result.json').exists() else {};accept=read(run/'reacceptance/final/acceptance.json') if (run/'reacceptance/final/acceptance.json').exists() else (read(run/'final_acceptance/acceptance.json') if (run/'final_acceptance/acceptance.json').exists() else {})
        row=dict(episode=ep,construction_complete=r.get('construction_complete'),geometry_numeric_passed=accept.get('geometry_numeric_passed'),geometry_passed=r.get('geometry_passed'),coverage_passed=r.get('coverage_passed'),appearance_passed=r.get('appearance_passed'),initial_support_passed=accept.get('initial_support_passed'),rest_terminal_passed=accept.get('physical_passed'),reload_passed=accept.get('reload_passed'),final_terra_verdict=r.get('final_terra_verdict','unreviewed'),final_reacceptance=(run/'reacceptance/final/acceptance.json').exists(),terra_requests=r.get('terra_actual_http_requests'),artifact_path=str(run),report=name+'/index.html')
        rows.append(row)
        if accept.get('camera_verified_from_xml'):delivery_note(run,accept)
        def status(value):return 'PASS' if value is True else ('FAIL' if value is False else ('PENDING' if value is None else str(value)))
        cells=''.join('<td>'+html.escape(status(row[k]))+'</td>' for k in ['construction_complete','geometry_numeric_passed','geometry_passed','coverage_passed','appearance_passed','initial_support_passed','rest_terminal_passed','reload_passed','final_terra_verdict'])
        cards.append('<tr><td><a href="'+name+'/index.html">'+str(ep)+'</a></td>'+cells+'<td><a href="'+name+'/reacceptance/final/acceptance.json">落盘复验</a></td></tr>')
    old=read(root/'history_protection.json') if (root/'history_protection.json').exists() else dict(files={});changed=[name for name,h in old['files'].items() if not (ROOT/name).is_file() or sha(ROOT/name)!=h]
    write(root/'history_integrity.json',dict(passed=not changed,changed=changed,files=len(old['files']),full402_stop_present=(ROOT/'runs/full402_h0_v1/STOP').exists(),checked_utc=now()))
    stopped=(ROOT/'runs/full402_h0_v1/STOP').exists() and read(ROOT/'runs/full402_h0_v1/summary.json')['counts']['running']==0
    summary=dict(version=c['version'],episodes=rows,all_three_built=all(x['construction_complete'] for x in rows),all_three_reaccepted=all(x['final_reacceptance'] for x in rows),historical_integrity_passed=not changed,full402_stopped=stopped,policy_inference=False,training=False,updated_utc=now());write(root/'delivery_summary.json',summary)
    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);(root/'delivery_results.csv').write_text(buf.getvalue())
    gallery=''.join('<h2>Episode '+str(row['episode'])+'</h2><p><a href="packages/episode_'+f"{row['episode']:06}"+'.tar.gz">完整产物包</a> · <a href="episode_'+f"{row['episode']:06}"+'/DELIVERY.md">参数、尺寸、静置及失败详情</a> · <a href="episode_'+f"{row['episode']:06}"+'/reacceptance/final/camera_verified_from_xml.json">XML 核对相机参数</a></p><a href="'+row['report']+'"><img src="episode_'+f"{row['episode']:06}"+'/comparison.png"></a>' for row in rows)
    (root/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Movens-Sim 三样本重建验收</title><style>body{font:16px system-ui;background:#161b22;color:#eee;margin:24px}a{color:#8cf}td,th{padding:8px;border-bottom:1px solid #555}img{width:100%}pre{white-space:pre-wrap}</style><h1>Movens-Sim · episode 0、1、2 重建与落盘验收</h1><p>真实图与三维重建对照；原生软层固定一秒终态诊断。没有策略推理或训练，402 条任务保持停止。失败/待审阅状态完整列出。</p><p><a href="delivery_results.csv">三条汇总 CSV</a> · <a href="delivery_summary.json">完整汇总</a> · <a href="final_reacceptance.json">最终重新验收</a> · <a href="history_integrity.json">历史保护核验</a> · <a href="verification/pytest.log">回归测试</a> · <a href="packages/modified_code.tar.gz">代码与补丁</a> · <a href="code_delivery.json">实际代码版本</a></p><table><tr><th>Episode</th><th>构建</th><th>几何数值</th><th>几何综合</th><th>全场景覆盖</th><th>外观</th><th>初始支撑</th><th>静置终态</th><th>保存重载</th><th>Terra 终审</th><th>产物</th></tr>'+''.join(cards)+'</table>'+gallery+'</html>')
    print(str(root/'index.html'))

if __name__=='__main__':main()
