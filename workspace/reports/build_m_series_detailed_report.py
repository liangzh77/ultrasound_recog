from pathlib import Path
import json, hashlib, csv, re, os
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'workspace/experiments/active/exp_2026-07_patient_multimodal_v1/reports'
OUT = ROOT / 'docs/project/M0至M3语义病灶轮廓研究详细报告_2026-09-17'
RAW = BASE / 'semantic_contours/M0_M3_comprehensive_evaluation.json'
D = json.loads(RAW.read_text(encoding='utf-8'))
E = D['evaluation']; MODELS = E['models']
NAMES = ['正常','类风湿性关节炎','痛风性关节炎','脊柱关节炎','骨性关节炎','损伤']
SHORT = ['正常','类风湿','痛风','脊柱关节炎','骨关节炎','损伤']
LABEL = {'M0':'M0','M1':'M1','M2':'M2','M0_C3_hierarchical':'M0＋C3','M3':'M3'}
parts=[]
def h(s, level=2): parts.append('#'*level+' '+s+'\n')
def p(s): parts.append(s+'\n')
def table(headers, rows):
    parts.append('| '+' | '.join(map(str,headers))+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n')
def f(x): return '未定义' if x is None else f'{x:.4f}'
def metrics_table(reports):
    table(['组别','macro F1','准确率','平衡准确率','macro AUC'],[[LABEL.get(k,k),f(v['macro_f1']),f(v['accuracy']),f(v['balanced_accuracy']),f(v['macro_auc'])] for k,v in reports.items()])
    table(['组别','Top 2准确率','ECE','Brier'],[[LABEL.get(k,k),f(v['top2_accuracy']),f(v['ece']),f(v['multiclass_brier'])] for k,v in reports.items()])
def class_table(m, names):
    table(['类别','人数','F1','灵敏度','特异度','精确率','NPV','AUC'],[[names[i],r['support']]+[f(r[k]) for k in ['f1','sensitivity','specificity','precision','npv','auc']] for i,r in enumerate(m['per_class'])])
def matrix_table(m, names):
    table(['真值／预测']+names,[[names[i]]+row for i,row in enumerate(m['confusion_matrix'])])

# Verify immutable inputs before compiling the report.
paths={'M0':BASE/'oof/E1_oof.csv','M1':BASE/'oof/M1_oof.csv','M2':BASE/'oof/M2_oof.csv','C3':BASE/'clinical/oof/C3_oof.csv'}
checks=[]
for name,path in paths.items():
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==D['input_sha256'][name],name
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    assert len(rows)==(767 if name=='C3' else 967)
    checks.append((name,len(rows),digest))
for name,r in MODELS.items():
    m=r['metrics']; cm=m['confusion_matrix']
    assert sum(map(sum,cm))==967
    assert abs(sum(cm[i][i] for i in range(6))/967-m['accuracy'])<1e-12
    assert abs(sum(x['f1'] for x in m['per_class'])/6-m['macro_f1'])<1e-12

h('膝关节超声 M0 至 M3 语义病灶轮廓研究详细报告',1)
p('方法学  全部实验结果  统计分析与研究解释')
p('报告编制日期：2026年9月17日｜实验完成日期：2026年8月28日｜研究编号：knee_patient_multimodal_v1_20260724')
h('摘要')
p('本轮研究回答三个递进问题：医生按类别标注的病灶轮廓能否改善患者级超声六分类；完整图像与轮廓像素是否比病灶类别、数量和面积更有价值；进一步加入临床数值能否超过更简单的图像加临床分层方案。研究采用967名患者、4543张图像的既有开发队列，固定患者级五折，比较M0纯图像、M1人工结构化征象、M2图像加13类语义轮廓、M3固定临床融合，并加入M0＋C3分层对照。2026时间外队列不参与本轮训练、选择或评价。[1–7]')
p('M1的macro F1为0.6351，准确率为0.6991；M2的macro F1为0.3327，比M0提高0.1172，但比M1低0.3024。M3加入临床后达到0.5676，比M2提高0.2348，仍低于M0＋C3的0.6018，差值为−0.0342，配对95%区间为−0.0578至−0.0096。当前结果支持人工结构化征象具有内部可分性，也支持语义通道相对纯图像的改善；没有建立完整轮廓像素模型优于结构化征象的证据，也不支持替换既有分层方案。')
p('解释中最重要的两个限制是：历史轮廓可能混合真实征象与诊断相关标注行为；临床C3只覆盖767名异常患者，200名正常患者无C3，因而临床可用性本身与参考诊断相关。M1不是自动图像识别结果，M3及M0＋C3不是无偏的前瞻性六分类性能。')
h('阅读导航')
p('第1至4章介绍研究问题、队列、模型及统计方法；第5至9章给出全部主要实验结果与解释；第10章说明局限和后续验证；附录提供完整逐类指标、混淆矩阵、复现入口和证据来源。表中概率指标保留四位小数，差值由未舍入原值计算。')
h('1 研究问题与实验边界')
p('本研究是历史开发队列上的回顾性方法比较。目标是每名患者输出一个主要诊断类别，而不是对每张图分别下诊断，也不是预测每一个病灶的分割质量。M2使用的病灶类别和轮廓由医生已有标注提供，因此本轮没有评价自动病灶检出或自动分割器，也没有Dice、IoU或病灶定位准确率作为本轮终点。')
table(['比较','要回答的问题'],[['M1与M0','人工给定病灶类别、数量、面积具有多大可分性'],['M2与M0','语义轮廓通道加入完整有效视野后有无整体收益'],['M2与M1','所测试的像素模型是否超过结构化征象模型'],['M3与M2','固定临床融合是否改善语义图像流程'],['M3与M0＋C3','较复杂的语义轮廓流程是否值得替代简单分层对照']])
p('预登记不设置可事后调整的单一通过线，而是综合效应量、置信区间、五折方向、逐类表现、校准与标注工作量。M2五折启动后不按单折结果修改方法；M3只执行一次0.75与0.25的固定融合，不搜索权重。M0复用既有E1严格OOF，因为队列、折分、ROI、分辨率、骨干、训练参数及患者聚合方式与本轮定义一致。[1]')
h('2 队列与语义轮廓数据')
h('2.1 患者构成与评价单位',3)
table(['主要诊断','患者数','占比'],[[NAMES[i],r['support'],f"{r['support']/967:.2%}"] for i,r in enumerate(MODELS['M0']['metrics']['per_class'])])
p('全部模型在相同967名患者上形成六分类OOF评价。每名患者可包含多张图像，训练、验证和外层测试按患者隔离，同一患者的多张图不能跨集合。4543张图像不是4543个独立样本。图像、语义通道及临床OOF依照伪匿名患者键和固定outer fold对齐。临床子分析仅包括767名异常患者，不能与967人的六分类指标直接作同任务比较。')
h('2.2 类别和空间表示',3)
p('冻结13类病灶或征象，每类对应一个独立二值通道；解剖结构类和含义尚未统一的“滑膜”“鹅足腱滑囊”不进入本轮。不同类别允许同像素重叠，不强制互斥。同类同图多边形的面积取二值并集，但对象数仍分别累计。完整有效视野RGB保留病灶以外区域和病灶之间的相对位置，不裁成病灶小块。[2]')
source=(ROOT/'docs/project/第二阶段M0_M3语义轮廓数据准备结果_2026-08-28.md').read_text(encoding='utf-8')
start=source.index('| 类别 | 对象数'); end=source.index('\n\n',start)
parts.append(source[start:end]+'\n')
p('4543张图中，2144张至少有一个非零病灶通道，2399张所有病灶通道均为零；稀疏保存2346个“图像×类别”mask。加载时按冻结顺序恢复完整13通道。全零或某类零通道只能解释为历史标注未观察到，不能解释为经统一协议确认的阴性。稀有类如半月板囊肿、滑膜囊肿均只有1名患者支持，无法据此建立稳定的独立类别学习或泛化结论。')
h('2.3 质量控制与隔离',3)
p('mask与RGB统一映射到384×384的等比例缩放加填充画布，保留完整有效视野。manifest覆盖全部4543张图且不包含姓名、原始路径、文件名、诊断或2026信息；加载器检查文件哈希及图像键覆盖。fold 0用于图像训练的训练集为618人2929图，内层验证为155人719图，外层测试为194人895图。外层测试不用于训练、早停和资源pilot。[2,4]')
h('3 模型与训练方法')
h('3.1 M0 纯图像参照',3)
p('M0为E1 EfficientNet B2图像模型的既有OOF。每张完整有效视野RGB经384像素letterbox后进入ImageNet预训练骨干，经全局平均池化、dropout和线性六分类头得到逐图softmax概率。患者概率为该患者全部有效图像概率的等权平均；它不是对类别标签投票，也不是先平均特征再分类。最终预测取患者六类概率中的最大值。[1,8]')
p('患者i有n张图时，Pᵢ(c) = (1/n) × Σⱼ softmax(zᵢⱼ)₍c₎。训练损失为患者真实类别平均概率的负对数，即 −log Pᵢ(yᵢ)。图像模型使用患者类别反频率的有放回平衡抽样，避免图像较多的患者被当作更多独立病例。')
h('3.2 M1 人工结构化征象模型',3)
p('每名患者对13类征象分别计算三项输入，共39维。presence表示任一图是否有该类有效轮廓；object count为全部图中的该类有效多边形总数；area fraction为该类逐图并集阳性像素数之和除以患者图像数×384×384。分母包含letterbox画布，因而是画布面积占比，不是物理病灶面积或解剖组织面积。对象数没有作对数变换。[1,3]')
p('M1只使用这些征象，不读取RGB、临床Excel、日期、文件名或身份信息。每个外层训练折拟合StandardScaler，再拟合多项逻辑回归：C=1.0、class weight=balanced、lbfgs、最大2000次迭代；无超参数搜索。预处理与模型拟合均不使用该折外层测试患者。M1不同于图像模型的内层验证早停流程，它是固定参数的外层训练折直接拟合。')
h('3.3 M2 图像与语义轮廓联合模型',3)
p('M2沿用M0的骨干和患者概率平均结构，输入改为3个RGB通道与13个独立语义mask，共16通道。ImageNet预训练通过timm的in_chans参数适配输入层；不能将16通道网络说成从零训练，也不能把mask当作模型自行预测的分割结果。每个正式fold从同一预训练权重独立初始化，不复用pilot checkpoint。[4,5,8]')
table(['训练项目','冻结设置'],[['骨干与输出','efficientnet_b2.ra_in1k，六分类，dropout 0.30'],['画布与患者采样','384×384；每训练epoch每患者最多随机取6图'],['验证和测试','读取患者全部图像；最佳验证epoch冻结后评价外层测试'],['batch','实际患者batch 1；有效患者batch 8，梯度累积'],['优化器','AdamW；编码器学习率0.0001，分类头0.0003'],['正则与稳定性','weight decay 0.0001；梯度裁剪1.0；AMP开启'],['学习率与训练长度','3轮warmup后余弦调度；最多60轮'],['早停','内层验证macro F1；patience 10'],['随机种子','20260724至20260728，按fold递增'],['数据加载与线程','worker 2；CPU线程上限4，interop线程1']])
p('训练增强对RGB和全部mask同步水平翻转（概率0.5）、−7至＋7度旋转、横纵不超过画布3%的平移、0.95至1.05倍缩放；RGB采用双线性插值，mask采用最近邻插值。亮度与对比度倍率0.85至1.15只作用于RGB。RGB用ImageNet均值和标准差归一化，mask保留独立的二值语义。')
h('3.4 C3临床模型及输入边界',3)
p('C3是已经冻结的异常五分类临床基线，输入为性别、年龄、ESR、CRP、抗CCP、RF、HLA B27和尿酸八项字段。性别按男1女0，HLA B27按阳性1阴性0、缺失中性值0.5；其余所需中位数插补与标准化仅在外层训练折拟合。C3使用固定的平衡多项逻辑回归，不加入显式缺失指示器。身份、诊断文字、病程、日期、文件名和源路径不进入模型。[9]')
p('“Excel是否存在”虽然不作为C3的输入特征，却决定分层流程走哪条分支。因此不显式加入缺失特征，并不能消除整个融合流程中的临床可用性偏倚。')
h('3.5 M3与分层对照的数学定义',3)
p('令p(c)为M2六分类概率，c=0表示正常；q(c)为M2在异常五类中的条件概率，即p(c)/Σₖ₌₁⁵p(k)；r(c)为C3五分类概率。对于有C3的患者，M3按下列公式计算：[6,8]')
p('P_M3(0) = p(0)\nP_M3(c) = [1−p(0)] × [0.75 r(c)＋0.25 q(c)]，c为异常类别。')
p('无C3时直接保留M2原始六分类概率。M0＋C3对照采用M0的正常概率，对有C3者将剩余概率质量完全按C3五类分配；无C3时保留M0六分类。两者都不是“正常概率大于0.5就判正常”的硬阈值门控，最终仍对完整六类概率取argmax。')
p('M3没有新增模型训练，也没有学习融合权重。其与M0＋C3的差别同时包括正常概率来源（M2或M0）以及异常条件概率是否混入25%图像分量，故完整六分类差异不能只归因于某一个因素。767人异常条件分析可单独观察混合概率的表现，但未构成独立的正常门控因果消融。')
h('4 统计评价方法')
p('主要终点是合并967名患者OOF后的六类macro F1，即每类F1的等权平均。OOF指每人的预测来自没有用该人进行拟合的对应外层模型。总体OOF macro F1不是五个fold macro F1的算术平均，因为F1对混淆矩阵是非线性的。准确率按患者比例计算；平衡准确率为各类灵敏度平均；Top 2为真值进入最高两个概率类别的比例。[8]')
p('逐类灵敏度为TP/(TP+FN)，特异度为TN/(TN+FP)，精确率为TP/(TP+FP)，NPV为TN/(TN+FN)，F1为精确率和灵敏度的调和平均。各类AUC按一类对其余类计算，macro AUC为其平均值；AUC反映排序，不表示argmax分类在当前阈值下同样好。分母为零的精确率等指标记为未定义，F1按代码zero_division=0处理。')
p('ECE将最高类别置信度划分为10个等宽区间，按各区间样本占比加权置信度与实际正确率的绝对差；Brier为六类预测概率与one hot真值平方差之和，再按患者取平均。两者越低越好，但低ECE单独不能证明识别能力强；弱模型也可能以低置信度获得较低ECE。五类与六类Brier的任务结构不同，应在相同任务内解释。')
p('95%区间使用2000次按参考诊断分层的患者bootstrap，随机种子20260724，每类有放回抽取原类人数。配对差值对两模型使用相同抽样患者，取差值分布2.5%和97.5%分位数。该区间基于已固定OOF预测，不在每次bootstrap中重新训练，不能覆盖全部训练随机性、中心差异或未来人群偏移。正向bootstrap比例是重采样差值大于零的频率，不等于贝叶斯模型胜出概率，也不应直接当作传统p值。五组比较未进行多重比较校正，区间应作为预登记探索性证据解释。')
h('5 数据准备与资源验证结果')
p('语义数据准备完成了患者和图像全覆盖、13类可重叠mask、固定通道顺序、稀疏manifest与哈希检查；这证明数据接口可以按合同使用，不证明标注已经穷尽或无诊断泄漏。M1正式CPU评价耗时16.50秒；M2先做fold 0五epoch资源pilot，再独立启动五折正式训练；M3评价耗时48.42秒，未使用GPU。[2–6]')
table(['pilot指标','结果'],[['用时','7.70分钟，共5 epoch'],['显存','峰值分配1.354 GB；峰值保留1.963 GB'],['CPU与内存','epoch后CPU采样11.3%至14.1%；最低可用内存35.14 GB'],['温度与异常','GPU 60至62摄氏度；NaN、溢出、异常停止均为0'],['验证macro F1','0.1963，仅作运行记录，不用于选择模型'],['外层测试','未迭代'],['结论','通过资源门；没有修改worker、batch、AMP、类别或模型']])
table(['fold','完成epoch','最佳epoch','M2外层macro F1'],[[i,a,b,f(MODELS['M2']['fold_macro_f1'][str(i)])] for i,(a,b) in enumerate([(28,17),(21,10),(14,3),(22,11),(16,5)])])
p('M2正式训练总耗时2.4430小时，峰值保留显存2.3711 GB。最佳epoch与完成epoch沿用原始报告的计数口径；各折不同的停止位置是验证早停结果，不是按外层测试选择训练时长。')
h('6 全部模型总体结果')
metrics_table({k:r['metrics'] for k,r in MODELS.items()})
table(['组别','macro F1','95%区间','正确患者数'],[[LABEL[k],f(r['metrics']['macro_f1']),f"{f(r['macro_f1_ci95'][0])} 至 {f(r['macro_f1_ci95'][2])}",sum(r['metrics']['confusion_matrix'][i][i] for i in range(6))] for k,r in MODELS.items()])
p('按当前开发队列六分类macro F1，排序为M1、M0＋C3、M3、M2、M0；这不是相同自动化程度下的产品排名。M1预先获得医生的类别和轮廓，M0只看RGB；M0＋C3与M3还存在临床可用性关联。M1准确率69.91%也不能表述为“AI自动看超声达到约70%”。')
h('6.1 五折稳定性',3)
table(['fold']+[LABEL[k] for k in MODELS],[[i]+[f(r['fold_macro_f1'][str(i)]) for r in MODELS.values()] for i in range(5)])
p('M1五折始终高于M0，M2也五折高于M0，但M2所有折都低于M1。M3五折高于M2，而相对于M0＋C3只有2折正向。稳定的折间方向说明结果不是仅由单一外层折产生，但不能排除整个历史队列都共享的标注流程偏倚。')
h('6.2 冻结的五项配对比较',3)
cn={'M1_minus_M0':'M1−M0','M2_minus_M0':'M2−M0','M2_minus_M1':'M2−M1','M3_minus_M2':'M3−M2','M3_minus_M0_C3_hierarchical':'M3−M0＋C3'}
table(['比较','差值','95%区间','正向折','重采样正向比例'],[[cn[k],f(r['observed_delta']),f"{f(r['ci_low'])} 至 {f(r['ci_high'])}",str(r['positive_folds'])+'/5',f(r['probability_delta_gt_zero'])] for k,r in E['comparisons'].items()])
table(['fold']+list(cn.values()),[[i]+[f(r['fold_macro_f1_deltas'][str(i)]) for r in E['comparisons'].values()] for i in range(5)])
p('M2相对M0的提升在该内部比较中明确，但M2相对M1的区间完全为负，说明所测试的网络没有兑现比结构化征象更好的分类表现。它不等于证明轮廓形状在所有模型和数据下都无用：M1与M2的表示、学习器和优化难度不同，本轮比较提供的是方法层面的增量证据，而非严格的信息论或因果分解。')
h('7 分类别结果与失败模式')
table(['类别']+[LABEL[k] for k in MODELS],[[SHORT[i]]+[f(r['metrics']['per_class'][i]['f1']) for r in MODELS.values()] for i in range(6)])
p('M1在痛风与骨关节炎上的F1分别为0.9383和0.8175；其正常灵敏度0.9500，但损伤灵敏度只有0.1690。痛风石、结晶、骨赘等类别携带医学征象信息，也可能受到标注者已知诊断和选择性标注影响；本轮不能量化两部分的贡献。正常的高灵敏度可能部分来自“少标或未标”的模式，需要盲法完整标注才能检验。')
p('M2正常灵敏度只有0.1100，即200名正常患者中仅22人判对；损伤灵敏度0.0845，即71人中仅6人判对。其正常特异度达到1.0000与正常识别很差并不矛盾：模型极少预测正常，造成假阳性很少，但漏掉大量真正正常患者。整体性能改善仍未解决逐类安全问题。')
p('M3继承M2的正常概率，正常患者又没有C3，因此仍只判对22名正常人。M3的正常F1较M2更低，是因为异常患者融合后的预测也会改变，产生了更多误报为正常者；保持正常概率并不保证所有患者的最终argmax正常标签不变。')
table(['类别','M0＋C3 F1','M3 F1','差值'],[[SHORT[i],f(MODELS['M0_C3_hierarchical']['metrics']['per_class'][i]['f1']),f(MODELS['M3']['metrics']['per_class'][i]['f1']),f(MODELS['M3']['metrics']['per_class'][i]['f1']-MODELS['M0_C3_hierarchical']['metrics']['per_class'][i]['f1'])] for i in range(6)])
p('M3主要改善痛风，骨关节炎变化很小；正常、类风湿与损伤下降。正常F1下降0.1397，是完整六分类结果恶化的重要表现。不能因单个疾病改善就认为复杂融合方案有整体收益。附录A给出所有组的完整逐类指标与混淆矩阵，便于区分漏判和错判方向。')
h('8 异常五分类条件分析')
p('本节仅选择具有C3的767名异常患者，将M2及M3去除正常类别后重新归一化，与C3五分类比较。它回答“已经限定为异常患者时，混入图像概率是否有帮助”，不评价正常与异常的区分。以下为描述性结果，没有为这组五分类差值另外报告配对区间。')
metrics_table(E['abnormal_condition_analysis'])
table(['类别','M2 F1','C3 F1','M3 F1'],[[SHORT[i+1]]+[f(E['abnormal_condition_analysis'][k]['per_class'][i]['f1']) for k in ['M2','C3','M3']] for i in range(5)])
p('C3 macro F1为0.7185，M2异常条件模型为0.3868，M3为0.7141；M3比C3低约0.0044。融合后的ECE由0.0475升至0.1391，损伤F1由0.5714降至0.5235。该观察与“25% M2分量没有形成稳定净收益”一致，但不能把未经单独区间估计的小幅macro F1差异描述为已证实的显著下降。')
h('9 方法论解释与研究结论')
h('9.1 语义标签信息与像素信息需要分开解释',3)
p('M1没有看RGB，却达到本轮最高macro F1，说明医生已经给出的病灶类别及其分布是强信号。既有S1只用13类是否出现的macro F1为0.6087，M1加入对象数和面积后为0.6351，名义差约0.0264；这是既有结果的背景对照，不是本轮新增的同一配对消融，不能把全部差值确定归因于数量和面积。[3]')
p('M2同时获得类别、轮廓与图像，其超过M0只能证明整体输入方案有增量，不能独立证明“形状”或“空间关系”有贡献。要分解来源，需要事先设计的去类别、打乱空间、仅mask、仅结构化征象等针对性消融；本轮没有运行这些额外实验，因此不为它们给出推断性能。')
h('9.2 融合并不天然产生互补收益',3)
p('M3相对M2的大幅提升表明临床分支为该流程提供了有效信息，但较强模型与较弱模型的平均不必优于强模型。异常条件分析显示M2明显弱于C3，固定混合后分类和校准没有稳定获益；完整六分类同时受到M2正常分支表现限制。本轮结果不足以支持继续在同一开发集搜索融合比例。')
h('9.3 可以与不可以得出的结论',3)
p('可以得出：当前历史队列中，人工结构化征象具有较强的内部诊断可分性；所测试的语义通道网络优于其纯图像参照；所测试的完整图像与轮廓模型没有超过结构化征象；固定临床融合没有超过简单分层对照。这些结论对应明确的模型、数据版本和评价方式。')
p('不能得出：模型已经能自动识别全部病灶；医生轮廓无论采用何种方法都没有价值；约70%的内部准确率可以迁移到前瞻性医院流程；某个高分组已具备独立临床诊断能力。M1不是真实诊断能力的数学上界，也不是已验证的产品，其高分应作为后续标准化征象研究的可行性线索。')
h('10 局限与后续验证条件')
p('第一，历史标注不是逐图穷尽的阴阳性记录。不存在轮廓可能对应真阴性、未观察、未记录或切面不可见，类别支持与诊断集中也明显不均。患者级分折能防止同一人的图像泄漏，却不能消除跨患者稳定存在的诊断相关标注行为。')
p('第二，临床覆盖与诊断关联。所有正常患者缺C3，所有异常患者有C3，导致流程中不同参考类别走不同分支。当前融合结果仅适合历史工作流敏感性分析；未来必须在真实目标人群中统一采集临床字段并预先固定缺失处理，才能评价部署流程。')
p('第三，967名患者并不意味着每种征象都有足够独立病例，单患者支持的稀有类不能提供稳定的跨折阳性学习。图像数、对象数和像素数都不能代替独立患者数。M1对象数及面积还可能受每人采集图数、切面和重复记录影响。')
p('第四，当前研究只比较冻结的学习器与参数，没有在独立数据上评价医生之间一致性、标注耗时、真实严重程度或跨中心变化。bootstrap只刻画固定OOF在当前队列上的患者抽样不确定性。既往在相同开发队列上的多轮研究还会带来研究者适应性选择风险；本轮预登记降低了本轮自由度，但不能把历史开发队列变成外部验证集。')
p('后续优先条件是重新建立标准化独立标注：逐图记录存在、不存在、不可见、不确定；每个轮廓绑定冻结类别；记录标注者及版本，尽量对最终诊断和AI结果盲法；先检查一致性和支持度，再固定模型和统计计划。M1应作为正式对照持续保留，M2需要在这种数据上重新证明增量。另需记录医生标注时间，评价复杂轮廓流程的实际成本，而不只比较分数。')
p('本轮决策为停止同源历史语义轮廓调优，不替换R0＋E1＋C3冻结研究预测；2026队列的医院参考诊断应按既有回收合同独立评价。该队列没有进入本报告的M0至M3性能比较，不能把开发结果与时间外验证混写。[7]')
h('附录A 全部六分类逐类指标与混淆矩阵')
p('每组均为967名患者。混淆矩阵行为参考诊断，列为模型预测，格内为患者数；各行和等于对应支持数。NPV为阴性预测值。类别名称在表内适当缩写，顺序始终固定。')
for k,r in MODELS.items():
    h('A'+str(list(MODELS).index(k)+1)+' '+LABEL[k]+' 完整结果',3)
    class_table(r['metrics'],SHORT)
    matrix_table(r['metrics'],SHORT)
h('附录B 异常五分类完整结果')
p('以下每组为同一767名异常患者，行列顺序与附录A去除正常类别后一致。')
for i,(k,m) in enumerate(E['abnormal_condition_analysis'].items()):
    h('B'+str(i+1)+' '+k+' 异常条件结果',3)
    class_table(m,SHORT[1:]); matrix_table(m,SHORT[1:])
h('附录C 复现入口与版本证据')
p('研究数据指纹：62ecb01c4d77ec0012704611ecc8d18ef51ebb4e0ea744fb3896948829f0b675。标注版本：annotations_62ecb01c4d77。以下入口均相对项目根目录；必须保持配置、数据、分折和预训练权重版本一致。报告汇编不需要重新训练模型。')
table(['步骤','脚本入口'],[['语义数据准备','tools/67_prepare_m_series_semantic_contours.py'],['M1固定五折','tools/68_evaluate_m1_semantic_features.py'],['M2资源pilot','tools/26_train_patient_image_mean.py'],['M2正式五折','tools/69_run_m2_formal_folds.py'],['M0至M3综合评价','tools/70_evaluate_m0_m3_semantic_study.py']])
p('M2 pilot完整命令：python tools/26_train_patient_image_mean.py --config configs/research/m2_semantic_contour_mean_b2.yaml --fold 0 --pilot --pilot-epochs 5。其余步骤使用python运行对应脚本，正式参数以冻结配置为准。')
table(['配置','SHA 256'],[[s,hashlib.sha256((ROOT/'configs/research'/s).read_bytes()).hexdigest()] for s in ['m_series_semantic_contour_preparation.yaml','m1_semantic_finding_features_logreg.yaml','m2_semantic_contour_mean_b2.yaml','m3_semantic_c3_fixed_fusion.yaml']])
p('正式Git版本：M1为c08dc0f9ab37b436781dd4e034ce83cdbacb83ab；M2为fe1020717a7648a367c67960b83db312873ed3d0；综合评价为8e421fee35f163d299327a161a61fe9f29449542。')
for name,n,digest in checks: p(f'{name}输入OOF：{n}行；SHA 256：{digest}。')
p('综合评价JSON：workspace/experiments/active/exp_2026-07_patient_multimodal_v1/reports/semantic_contours/M0_M3_comprehensive_evaluation.json；SHA 256：'+hashlib.sha256(RAW.read_bytes()).hexdigest()+'。')
p('M3输出OOF SHA 256：'+hashlib.sha256((BASE/'oof/M3_oof.csv').read_bytes()).hexdigest()+'。正式MLflow综合评价parent run：5042dbcdb2684bd6b5e01027781a4b88。M1、M2 checkpoint及逐折tracking标识见来源[3–5]，本报告不展开患者级记录。')
h('附录D 来源与数据追溯')
sources=[
('1','第二阶段M0_M3语义病灶轮廓实验预登记_2026-08-28.md'),
('2','第二阶段M0_M3语义轮廓数据准备结果_2026-08-28.md'),
('3','M1分类别病灶征象患者六分类五折结果_2026-08-28.md'),
('4','M2分类别语义轮廓资源pilot结果_2026-08-28.md'),
('5','M2分类别语义轮廓患者六分类五折结果_2026-08-28.md'),
('6','M0_M3语义病灶轮廓综合评价结果_2026-08-28.md')]
for no,name in sources: p(f'[{no}] docs/project/{name}')
p('[7] docs/decisions/ADR-025-M3未超过自动分层对照并停止历史语义轮廓调优.md')
p('[8] 方法实现：src/research_models.py、src/research_training.py、src/research_semantic_contours.py、src/research_semantic_features.py、src/research_semantic_study.py、src/research_metrics.py；冻结参数见附录C。完整总体、逐折、逐类、校准与混淆矩阵来自附录C综合评价JSON。')
p('[9] docs/project/C0_C4临床输入决策与实验预登记_2026-08-02.md，以及C0_C4临床五折基线与缺失偏差结果_2026-08-02.md。M0历史训练背景见E0_E1整图与ROI五折基线结果_2026-08-02.md。')
p('数据校核：本报告生成时对四个输入OOF重新核对冻结SHA 256与行数；对全部六分类矩阵核对患者总数、准确率与macro F1的一致性。综合指标直接读取原始评价JSON，不由已舍入的汇总表倒推。方法说明依据预登记、冻结配置及实现核对；本次没有新增训练、改变预测或补做未登记的模型搜索。')

text='\n'.join(parts)
# Physician-facing delivery: retain results and methodological boundaries while
# moving software provenance into a separate local verification record.
cstart=text.index('## 附录C ')
text=text[:cstart]+'''## 附录C 研究资料与可追溯性

本报告依据2026年8月28日完成的预登记研究及其原始评价记录编制，报告整理日期为2026年9月17日。本次为既有实验的完整汇报，没有重新训练或更改预测。各组使用同一患者级分组进行比较，输入版本、模型配置、预测结果及统计程序均已留存，可按需要提供技术复核材料。

报告编制时重新核对了四份输入预测文件的版本校验值和患者行数，并复核全部六分类混淆矩阵的患者总数、准确率与宏平均F1。表内指标直接取自原始统计结果；差值在舍入前计算，因此可能与表内四位小数直接相减有末位差异。

主要研究记录如下。

[1] 第二阶段 M0 至 M3 语义病灶轮廓实验预登记，2026年8月28日。

[2] 第二阶段 M0 至 M3 语义轮廓数据准备结果，2026年8月28日。

[3] M1 分类别病灶征象患者六分类五折结果，2026年8月28日。

[4] M2 分类别语义轮廓资源试运行结果，2026年8月28日。

[5] M2 分类别语义轮廓患者六分类五折结果，2026年8月28日。

[6] M0 至 M3 语义病灶轮廓综合评价结果及原始统计数据，2026年8月28日。

[7] 本轮研究结论及后续路线决策记录，2026年8月28日。

[8] 患者级图像模型、语义轮廓处理、固定融合及统计评价的研究程序与冻结配置。

[9] C0 至 C4 临床输入预登记与五折基线结果，以及 E0 至 E1 图像五折基线结果，2026年8月2日。
'''
replacements={
'方法学  全部实验结果  统计分析与研究解释':'医院沟通版  研究方法  完整结果与分析',
'研究编号：knee_patient_multimodal_v1_20260724':'研究范围：患者级主要诊断六分类',
'历史轮廓可能混合真实征象与诊断相关标注行为':'既往轮廓资料同时反映影像征象与当时的检查、记录方式',
'因而临床可用性本身与参考诊断相关':'因此本批资料中临床信息的完整程度与疾病分组存在关联',
'不是无偏的前瞻性六分类性能':'尚不能直接代表未来实际就诊人群的六分类表现',
'可能受到标注者已知诊断和选择性标注影响':'也可能受到原有临床信息及记录重点差异的影响',
'诊断相关标注行为':'与疾病分组相关的既往记录差异',
'诊断相关标注':'与疾病分组相关的既往标注',
'历史标注流程偏倚':'既往资料形成方式带来的偏差',
'标注流程偏倚':'标注方式差异带来的偏差',
'临床可用性偏倚':'临床资料覆盖差异带来的偏差',
'内部可分性':'本批资料中的疾病区分能力',
'具有较强的内部诊断可分性':'在本批资料中具有较强的疾病区分能力',
'不证明标注已经穷尽或无诊断泄漏':'并不意味着既往记录已达到前瞻性统一采集的要求',
'同一人的图像泄漏':'同一患者资料同时进入训练与测试',
'M1不是真实诊断能力的数学上界，也不是已验证的产品，其高分应作为后续标准化征象研究的可行性线索。':'M1的表现为后续标准化征象研究提供了依据，其实际辅助诊断价值仍需独立资料验证。',
'本轮决策为停止同源历史语义轮廓调优，不替换R0＋E1＋C3冻结研究预测；2026队列的医院参考诊断应按既有回收合同独立评价。':'本轮尚未获得支持变更既有研究预测方案的证据。后续工作的重点宜由继续调整同一批资料上的模型，转向标准化资料补充与独立验证；2026队列可按既定方案，在获得医院参考诊断后开展独立评价。',
'未来必须在真实目标人群中统一采集临床字段并预先固定缺失处理':'后续宜在真实目标人群中统一采集临床字段并预先约定缺失处理',
'本轮不能量化两部分的贡献':'本轮尚不能分别量化这些因素的贡献',
'产品排名':'临床应用价值排序',
'现有结果不足以支持继续在同一开发集搜索融合比例':'现有结果提示，继续调整同一批资料上的融合比例，其新增临床价值仍需独立验证',
}
for old,new in replacements.items(): text=text.replace(old,new)
text=text.replace('## 阅读导航','## 如何理解本报告')
text=text.replace('第1至4章介绍研究问题、队列、模型及统计方法；', '医生可先阅读摘要及第6至10章，了解效果、具体疾病的表现及后续验证重点。第1至4章介绍研究问题、队列、模型及统计方法；')
text=text.replace('本研究是历史开发队列上的回顾性方法比较。','本研究使用既往临床影像及标注资料开展回顾性方法比较。既往资料首先服务于当时的诊疗和记录，其采集目的与本次统一研究评价有所不同；以下局限反映两种用途的差异，不是对临床诊疗或标注质量的评价。')
text=text.replace('## 7 分类别结果与失败模式','## 7 各疾病类别的表现与主要不足')
text=text.replace('## 9.3 可以与不可以得出的结论','## 9.3 研究结论及适用范围')
text=text.replace('## 9 方法论解释与研究结论','## 9 结果的临床研究意义')
text=text.replace('没有在独立数据上评价医生之间一致性','没有在独立资料上评价不同标注者之间的一致性')
text=text.replace('优先条件是重新建立标准化独立标注','建议优先开展标准化的独立资料采集与标注')
text=text.replace('本轮没有运行这些额外实验，因此不为它们给出推断性能。','这些分析不在本轮实验范围内，需要后续另行设计验证。')
text=text.replace('当前结果支持人工结构化征象具有本批资料中的疾病区分能力','当前结果支持人工结构化征象在本批资料中具有疾病区分能力')
text=text.replace('患者级分组进行比较','患者级分组进行比较')
text=text.replace('完整结果与分析\n','完整结果与分析\n')
text=text.replace('附录提供完整逐类指标、混淆矩阵、复现入口和证据来源','附录提供完整逐类指标、混淆矩阵和研究资料来源')
text=text.replace('不能将16通道网络说成从零训练，也不能把mask当作模型自行预测的分割结果。','16通道网络在预训练基础上适配输入层；其中的轮廓来自人工标注。')
text=text.replace('M1准确率69.91%也不能表述为“AI自动看超声达到约70%”。','M1准确率69.91%对应的是人工提供征象后的辅助分类表现。')
text=text.replace('可以得出：','本轮证据支持以下结论：')
text=text.replace('不能得出：模型已经能自动识别全部病灶；医生轮廓无论采用何种方法都没有价值；约70%的内部准确率可以迁移到前瞻性医院流程；某个高分组已具备独立临床诊断能力。','这些结果尚不能外推为自动病灶识别能力或前瞻性临床诊断性能；当前轮廓模型的不足，也不代表轮廓信息在其他方法中没有研究价值。')
insert='''
### 核心术语

本报告中的“患者级六分类”是根据同一患者的多张超声图及相应资料，给出正常、类风湿性关节炎、痛风性关节炎、脊柱关节炎、骨性关节炎、损伤中的一个主要诊断类别。“有效视野”指用于分析的完整超声成像区域；“语义轮廓”指带有明确病灶类别的人工轮廓；“临床分层对照”指图像负责提供正常概率、临床数值负责区分异常疾病的组合方法。它们均属于本阶段的研究方法。

宏平均F1（macro F1）兼顾每类的漏判与误判，并对六类同等加权，越高越好；它与“多少患者分对了”的准确率不同。“五折验证”将患者分为五组，轮流用其中一组测试，其余患者用于模型开发；“OOF预测”是每名患者作为测试对象时得到的预测，可用于汇总整批患者的表现。

'''
text=text.replace('## 1 研究问题与实验边界',insert+'## 1 研究问题与实验边界')
qa_dir=ROOT/'workspace/reports/m_series_report_qa'; qa_dir.mkdir(parents=True,exist_ok=True)
(qa_dir/'source_verification.json').write_text(json.dumps({'inputs':checks,'evaluation_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'patient_count':967,'report_audience':'hospital_physicians'},ensure_ascii=False,indent=2),encoding='utf-8')
OUT.with_suffix('.md').write_text(text,encoding='utf-8')

doc=Document(); sec=doc.sections[0]
sec.page_width=Cm(21); sec.page_height=Cm(29.7)
sec.top_margin=Cm(2); sec.bottom_margin=Cm(2); sec.left_margin=Cm(1.8); sec.right_margin=Cm(1.8)
for sty in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3']:
    s=doc.styles[sty]; s.font.name='Microsoft YaHei'; s.font.color.rgb=RGBColor(0,0,0)
    s.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    for attr in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:
        s.element.get_or_add_rPr().rFonts.attrib.pop(qn('w:'+attr),None)
for border in list(doc.styles.element.iter(qn('w:pBdr'))): border.getparent().remove(border)
doc.styles['Normal'].font.size=Pt(10)
doc.styles['Normal'].paragraph_format.line_spacing=1.2
doc.styles['Normal'].paragraph_format.space_after=Pt(6)
for sty,size in [('Title',22),('Heading 1',15),('Heading 2',12)]:
    doc.styles[sty].font.size=Pt(size)
    doc.styles[sty].paragraph_format.space_before=Pt(12)
    doc.styles[sty].paragraph_format.space_after=Pt(7)
footer=sec.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
footer.add_run('M0 至 M3 研究详细报告  ·  ')
fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); footer._p.append(fld)
for r in footer.runs: r.font.size=Pt(8)
lines=text.splitlines(); i=0
while i<len(lines):
    line=lines[i]
    if not line.strip(): i+=1; continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            cells=[c.strip() for c in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r'[-: ]+',c) for c in cells): rows.append(cells)
            i+=1
        t=doc.add_table(rows=0,cols=len(rows[0])); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
        cols=len(rows[0]); total=17.4
        widths=([4.3,13.1] if cols==2 else ([3.3]+[(total-3.3)/(cols-1)]*(cols-1)))
        if cols==3: widths=[5.4,5.0,7.0]
        for col,w in zip(t.columns,widths): col.width=Cm(w)
        for ri,values in enumerate(rows):
            row=t.add_row(); pr=row._tr.get_or_add_trPr()
            no=OxmlElement('w:cantSplit'); pr.append(no)
            if ri==0: pr.append(OxmlElement('w:tblHeader'))
            for ci,(cell,value) in enumerate(zip(row.cells,values)):
                cell.width=Cm(widths[ci]); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                cell.text=value.replace('`','')
                cp=cell._tc.get_or_add_tcPr()
                sh=OxmlElement('w:shd'); sh.set(qn('w:fill'),'DCE6F1' if ri==0 else ('F5F7F9' if ri%2==0 else 'FFFFFF')); cp.append(sh)
                mar=OxmlElement('w:tcMar')
                for edge,v in [('top',65),('bottom',65),('start',90),('end',90)]:
                    e=OxmlElement('w:'+edge); e.set(qn('w:w'),str(v)); e.set(qn('w:type'),'dxa'); mar.append(e)
                cp.append(mar)
                for para in cell.paragraphs:
                    para.paragraph_format.space_after=Pt(0); para.paragraph_format.line_spacing=1.1
                    if ri==0: para.paragraph_format.keep_with_next=True
                    if ci>0 and cols>=4: para.alignment=WD_ALIGN_PARAGRAPH.CENTER
                    for run in para.runs: run.font.size=Pt(8 if cols>=7 else 9); run.bold=(ri==0)
        borders=OxmlElement('w:tblBorders')
        for edge in ['top','left','bottom','right','insideH','insideV']:
            el=OxmlElement('w:'+edge); el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'4'); el.set(qn('w:color'),'D9D9D9'); borders.append(el)
        t._tbl.tblPr.append(borders)
        doc.add_paragraph().paragraph_format.space_after=Pt(1)
        continue
    if line.startswith('# '): doc.add_paragraph(line[2:],'Title')
    elif line.startswith('## '):
        para=doc.add_paragraph(line[3:],'Heading 1')
        if line.startswith('## 附录') or line.startswith('## 10 '): para.paragraph_format.page_break_before=True
    elif line.startswith('### '):
        para=doc.add_paragraph(line[4:],'Heading 2')
        if re.match(r'### [AB][2-9]',line): para.paragraph_format.page_break_before=True
    else: doc.add_paragraph(line)
    i+=1
doc.core_properties.title='膝关节超声 M0 至 M3 语义病灶轮廓研究详细报告'
doc.core_properties.subject='方法学 全部实验结果 统计分析与研究解释'
doc.core_properties.author=''
doc.save(OUT.with_suffix('.docx'))
print(json.dumps({'docx':str(OUT.with_suffix('.docx')),'markdown':str(OUT.with_suffix('.md')),'characters':len(text),'tables':len(doc.tables),'input_hashes_verified':len(checks)},ensure_ascii=False))
