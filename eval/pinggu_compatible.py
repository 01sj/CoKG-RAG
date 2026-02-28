"""
------------------------------------------------------------------------
    修改版评估脚本 - 兼容 hybrid_rag_query.py 输出格式
    
    支持的字段映射：
    - prediction / final_answer → 模型预测
    - answer / Gold answer → 标准答案
    
    作者: 基于原 pinggu.py 修改
    时间: 2025-12-25
------------------------------------------------------------------------
"""
import json
import logging
from function_utils import compute_rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import os
import numpy as np
import jieba

# 🔧 不设置离线模式，允许从HuggingFace下载模型
# os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 已注释，允许在线下载

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def compute_flzx(data_dict, output_save_path=None):
    """
    评估指标：ROUGE-1,2,L（字符级） BLEU, BERTScore
    
    兼容多种数据格式：
    - prediction / final_answer
    - answer / Gold answer
    """
    references, predictions = [], []
    
    # 兼容 dict / list 两种输入格式
    if isinstance(data_dict, dict):
        examples = list(data_dict.values())
    elif isinstance(data_dict, list):
        examples = data_dict
    else:
        raise TypeError("输入数据必须是 dict 或 list 类型")

    # 🔧 修改：兼容多种字段名
    for idx, example in enumerate(examples):
        # 提取预测结果（兼容 prediction / final_answer）
        if "prediction" in example:
            prediction = example["prediction"]
        elif "final_answer" in example:
            prediction = example["final_answer"]
        else:
            logger.warning(f"样本 {idx} 缺少预测字段 (prediction/final_answer)，跳过")
            continue
        
        # 提取标准答案（兼容 answer / Gold answer）
        if "answer" in example:
            answer = example["answer"]
        elif "Gold answer" in example:
            answer = example["Gold answer"]
        else:
            logger.warning(f"样本 {idx} 缺少答案字段 (answer/Gold answer)，跳过")
            continue
        
        # 清理答案格式（去除"答案:"前缀）
        if answer.startswith("答案:"):
            answer = answer[3:].strip()
        if prediction.startswith("答案:"):
            prediction = prediction[3:].strip()
        
        predictions.append(prediction)
        references.append(answer)

    if len(predictions) == 0:
        logger.error("❌ 没有有效的样本可以评估！")
        return None

    logger.info(f"✅ 成功加载 {len(predictions)} 个样本")

    try:
        rouge_scores = compute_rouge(predictions, references)
        rouge1_r, rouge1_p, rouge1_f = [], [], []
        rouge2_r, rouge2_p, rouge2_f = [], [], []
        rougel_r, rougel_p, rougel_f = [], [], []
        bleu_scores = []

        logger.info(f"\n🔍 每条样本指标：")
        smooth_fn = SmoothingFunction().method1

        for idx, score in enumerate(rouge_scores):
            r1, p1, f1 = score["rouge-1"]["r"], score["rouge-1"]["p"], score["rouge-1"]["f"]
            r2, p2, f2 = score["rouge-2"]["r"], score["rouge-2"]["p"], score["rouge-2"]["f"]
            rl_r, rl_p, rl_f = score["rouge-l"]["r"], score["rouge-l"]["p"], score["rouge-l"]["f"]

            rouge1_r.append(r1)
            rouge1_p.append(p1)
            rouge1_f.append(f1)
            rouge2_r.append(r2)
            rouge2_p.append(p2)
            rouge2_f.append(f2)
            rougel_r.append(rl_r)
            rougel_p.append(rl_p)
            rougel_f.append(rl_f)

            # BLEU计算（使用jieba分词）
            ref_tokens = list(jieba.cut(references[idx]))
            pred_tokens = list(jieba.cut(predictions[idx]))
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_fn)
            bleu_scores.append(bleu)

            logger.info(f"样本 {idx:>3}: ROUGE-L F1={rl_f:.4f}, BLEU={bleu:.4f}")

        # ========== 文本截断处理函数 ==========
        def truncate_text(text, max_length=1024):
            """截断文本到指定长度，避免BERT模型序列长度超限
            
            注意：BERT模型的最大序列长度通常为512，但这里设置为1024
            因为BERTScore会自动处理超长文本（分段计算）
            如果仍然遇到OOM错误，可以降低到512或768
            """
            if len(text) <= max_length:
                return text
            logger.warning(f"文本长度 {len(text)} 超过限制 {max_length}，已截断")
            return text[:max_length]

        # ========== BERTScore计算（添加错误处理） ==========
        bertscore_available = False
        try:
            # 截断过长的文本
            truncated_predictions = [truncate_text(pred) for pred in predictions]
            truncated_references = [truncate_text(ref) for ref in references]

            logger.info("🚀 开始计算BERTScore...")
            
            # 🔧 优先使用本地模型，如果不存在则从HuggingFace下载
            bert_model_path = os.environ.get("BERT_MODEL_PATH", None)
            
            if bert_model_path and os.path.exists(bert_model_path):
                # 使用本地模型
                logger.info(f"📂 使用本地BERT模型: {bert_model_path}")
                model_type = bert_model_path
            else:
                # 从HuggingFace下载（第一次会下载并缓存）
                logger.info("🌐 本地模型不存在，将从HuggingFace下载 bert-base-chinese")
                logger.info("💡 首次下载可能需要几分钟，模型会缓存到 ~/.cache/huggingface/")
                model_type = "bert-base-chinese"
            
            P, R, F1 = bert_score(
                truncated_predictions,
                truncated_references,
                model_type=model_type,
                num_layers=12,
                lang="zh",
                verbose=True  # 显示下载进度
            )
            bertscore_available = True
            logger.info("✅ BERTScore计算成功")

        except Exception as e:
            logger.warning(f"⚠️ BERTScore计算出错，跳过该指标: {e}")
            logger.info("💡 提示：可以使用 eval/pinggu_no_bert.py 脚本（无需BERT模型）")
            logger.info("💡 或者设置代理: export HF_ENDPOINT=https://hf-mirror.com")
            bertscore_available = False

        result = {
            "ROUGE-1": {
                "Recall": float(np.mean(rouge1_r)),
                "Precision": float(np.mean(rouge1_p)),
                "F1": float(np.mean(rouge1_f))
            },
            "ROUGE-2": {
                "Recall": float(np.mean(rouge2_r)),
                "Precision": float(np.mean(rouge2_p)),
                "F1": float(np.mean(rouge2_f))
            },
            "ROUGE-L": {
                "Recall": float(np.mean(rougel_r)),
                "Precision": float(np.mean(rougel_p)),
                "F1": float(np.mean(rougel_f))
            },
            "BLEU": float(np.mean(bleu_scores))
        }

        # 如果BERTScore计算成功，添加到结果中
        if bertscore_available:
            result["BERTScore"] = {
                "Precision": float(P.mean().item()),
                "Recall": float(R.mean().item()),
                "F1": float(F1.mean().item())
            }
        else:
            logger.info("ℹ️ 结果中不包含BERTScore指标")

        logger.info("\n📊 平均指标如下：")
        logger.info(json.dumps(result, indent=2, ensure_ascii=False))

        if output_save_path:
            with open(output_save_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 结果已保存至: {output_save_path}")

        return result

    except Exception as e:
        logger.error(f"❌ 指标计算出错: {e}")
        raise

def load_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"✅ 成功加载 JSON 文件: {file_path}")
        return data
    except Exception as e:
        logger.error(f"❌ 加载 JSON 文件失败: {e}")
        raise

def process_json_file(input_path, output_score_path=None):
    """
    处理JSON文件并计算评估指标
    
    Args:
        input_path: 输入JSON文件路径（hybrid_rag_query.py的输出）
        output_score_path: 评估结果保存路径
    """
    data = load_json_file(input_path)
    return compute_flzx(data, output_score_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评估 hybrid_rag_query.py 的输出结果")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入JSON文件路径（hybrid_rag_query.py的输出）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="评估结果保存路径（默认为输入文件名_metrics.json）"
    )
    
    args = parser.parse_args()
    
    # 确定输出路径
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(input_dir, f"{input_name}_metrics.json")
    
    # 执行评估
    try:
        logger.info(f"📂 输入文件: {args.input}")
        logger.info(f"📂 输出文件: {args.output}")
        logger.info("="*60)
        
        result = process_json_file(args.input, args.output)
        
        if result:
            logger.info(f"\n{'='*60}")
            logger.info("✅ 评估完成！")
            logger.info(f"{'='*60}")
            logger.info(f"\n最终结果:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            logger.error("❌ 评估失败！")
            
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
