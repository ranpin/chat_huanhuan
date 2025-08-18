#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
甄嬛角色模型训练脚本

基于《甄嬛传》角色数据进行LoRA微调，
训练出具有甄嬛语言风格的对话模型。

参考项目: https://github.com/KMnO4-zx/huanhuan-chat
使用方法:
    python training/huanhuan_train.py
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from loguru import logger

# 正确配置tokenizers并行化
# 在训练开始前设置，避免fork后的并行化冲突
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置logger
logger.remove()  # 移除默认的控制台输出
logger.add(sys.stdout, level="INFO")  # 添加控制台输出
script_dir = Path(__file__).parent
log_dir = script_dir / "logs"
log_dir.mkdir(exist_ok=True)

logger.add(str(log_dir / "training.log"), rotation="10 MB", retention="7 days", level="INFO")



class HuanHuanDataset(Dataset):
    """
    甄嬛角色对话数据集
    """
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = self.load_conversations(data_file)
        
        # 设置特殊token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def load_conversations(self, data_file: str) -> List[Dict]:
        """
        加载对话数据
        """
        conversations = []
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        conversations.append(data)
            logger.info(f"加载了 {len(conversations)} 条对话数据")
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
        
        return conversations
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # 构建输入文本
        instruction = conversation.get('instruction', '')
        input_text = conversation.get('input', '')
        output = conversation.get('output', '')
        
        # 格式化对话
        if input_text:
            prompt = f"指令：{instruction}\n输入：{input_text}\n回应："
        else:
            prompt = f"指令：{instruction}\n回应："
        
        # 完整文本
        full_text = prompt + output + self.tokenizer.eos_token
        
        # 编码
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 计算标签（只对回应部分计算损失）
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )
        
        labels = encoding['input_ids'].clone()
        prompt_length = len(prompt_encoding['input_ids'][0])
        labels[0, :prompt_length] = -100  # 忽略prompt部分的损失
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

class HuanHuanTrainer:
    """
    甄嬛角色模型训练器
    """

    def __init__(self, config_path: str = "./training/huanhuan_config_fast.yaml"):
        self.config_path = config_path
        self.config = None
        self.device = None
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
        
        logger.info(f"HuanHuanTrainer 初始化完成，配置文件: {config_path}")
        
        # 初始化所有组件
        self.load_config()
        self.setup_device()
        self.load_model_and_tokenizer()
        self.setup_lora()
        self.setup_training_arguments()
    
    def load_config(self):
        """
        加载配置文件
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # 创建输出目录
            os.makedirs(self.config['training']['output_dir'], exist_ok=True)
            logger.info(f"配置文件加载成功: {self.config_path}")
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise
    
    def setup_device(self):
        """
        设置计算设备
        """
        device_config = self.config['system']['device'] # type: ignore
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info(f"使用CUDA设备: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                logger.info("使用Apple Silicon MPS设备")
            else:
                self.device = torch.device('cpu')
                logger.info("使用CPU设备")
        else:
            self.device = torch.device(device_config)
            logger.info(f"使用指定设备: {self.device}")
    
    def load_model_and_tokenizer(self):
        """
        加载模型和分词器
        """
        model_name = self.config['model']['base_model'] # type: ignore
        
        logger.info(f"加载分词器: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # 设置特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"加载模型: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,  # 强制使用float32
            device_map=None  # 禁用自动设备映射
        )
        
        # 手动移动到指定设备
        self.model = self.model.to(self.device)
        logger.info(f"模型已移动到设备: {self.device}")
        
        logger.info(f"模型参数量: {self.model.num_parameters():,}")
    
    def setup_lora(self):
        """
        设置LoRA配置
        """
        # 检查是否启用LoRA（默认启用）
        if not self.config['lora'].get('enabled', True):
            logger.info("未启用LoRA，使用全参数微调")
            return
        
        logger.info("配置LoRA参数")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['lora'].get('r', 8),
            lora_alpha=self.config['lora'].get('lora_alpha', 16),
            lora_dropout=self.config['lora'].get('lora_dropout', 0.1),
            target_modules=self.config['lora'].get('target_modules', ["q_proj", "v_proj"]),
            bias=self.config['lora'].get('bias', 'none')
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        logger.info(f"总参数量: {total_params:,}")
    
    def prepare_datasets(self):
        """
        准备训练数据集
        """
        # 分别加载三个文件
        train_dataset = HuanHuanDataset(
            data_file=self.config['data']['train_file'],
            tokenizer=self.tokenizer,
            max_length=self.config['model'].get('max_length', 2048)
        )
        
        val_dataset = HuanHuanDataset(
            data_file=self.config['data']['validation_file'],
            tokenizer=self.tokenizer,
            max_length=self.config['model'].get('max_length', 2048)
        )
        
        # 测试集在训练时不用，但要保留用于最终评估
        test_dataset = HuanHuanDataset(
            data_file=self.config['data']['test_file'],
            tokenizer=self.tokenizer,
            max_length=self.config['model'].get('max_length', 2048)
        )
        
        logger.info(f"训练集: {len(train_dataset)} 样本")
        logger.info(f"验证集: {len(val_dataset)} 样本")
        logger.info(f"测试集: {len(test_dataset)} 样本")
        
        return train_dataset, val_dataset, test_dataset
    def setup_training_arguments(self):
        """
        设置训练参数并创建Trainer
        """
        training_config = self.config['training']
        
        # 根据设备类型动态调整参数
        is_mps_device = self.device.type == 'mps'
        is_cuda_device = self.device.type == 'cuda'
        
        # 数据加载器配置：根据设备和系统能力调整
        if is_mps_device:
            # MPS设备优化配置
            dataloader_num_workers = 0  # MPS设备建议使用单线程
            dataloader_pin_memory = False  # MPS不支持pin_memory
            logger.info("检测到MPS设备，使用MPS优化配置")
        elif is_cuda_device:
            # CUDA设备可以使用更多优化
            dataloader_num_workers = min(4, os.cpu_count() or 1)
            dataloader_pin_memory = True
            logger.info(f"检测到CUDA设备，使用{dataloader_num_workers}个数据加载器工作进程")
        else:
            # CPU设备配置
            dataloader_num_workers = min(2, os.cpu_count() or 1)
            dataloader_pin_memory = False
            logger.info(f"使用CPU设备，配置{dataloader_num_workers}个数据加载器工作进程")
        
        training_args = TrainingArguments(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=float(training_config['learning_rate']),
            weight_decay=float(training_config['weight_decay']),
            warmup_ratio=training_config.get('warmup_ratio', 0.1),
            max_grad_norm=float(training_config.get('max_grad_norm', 1.0)),
            
            # 保存和评估
            save_steps=training_config.get('save_steps', 100),
            eval_steps=training_config.get('eval_steps', 50),
            logging_steps=training_config.get('logging_steps', 10),
            eval_strategy=training_config.get('evaluation_strategy', 'steps'),
            save_strategy=training_config.get('save_strategy', 'steps'),
            
            # 设备相关优化配置
            fp16=training_config.get('fp16', False) and not is_mps_device,  # MPS不支持fp16
            bf16=training_config.get('bf16', False),
            dataloader_num_workers=dataloader_num_workers,
            dataloader_pin_memory=dataloader_pin_memory,
            seed=self.config['system'].get('seed', 42),
            
            # 日志和报告
            logging_dir=f"{training_config['output_dir']}/logs",
            report_to=training_config.get('report_to', []),
            
            # 最佳模型保存
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
            
            # 移除未使用的列
            remove_unused_columns=False,
        )
        
        # 准备数据集
        train_dataset, val_dataset, test_dataset = self.prepare_datasets()
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # 因果语言模型
            pad_to_multiple_of=8 if training_args.fp16 else None
        )
        
        # 创建训练器
        # PEFT模型的label_names警告是信息性的，不影响训练
        # Trainer会自动从数据集中检测labels列
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        logger.info("训练器创建完成")
        
        return training_args
    
    def train(self):
        """
        开始训练
        """
        logger.info("=== 开始甄嬛角色模型训练 ===")
        
        try:
            # 开始训练
            logger.info("🚀 开始模型训练...")
            train_result = self.trainer.train()
            
            # 保存模型
            logger.info("💾 保存最终模型...")
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config['training']['output_dir'])
            
            logger.info("=== 训练完成 ===")
            logger.info(f"📁 模型保存在: {self.config['training']['output_dir']}")
            
            return train_result
            
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            raise

def main():
    """
    主函数：执行甄嬛角色模型训练
    """
    try:
        # 创建训练器实例
        trainer = HuanHuanTrainer()
        
        # 训练模型
        train_result = trainer.train()
        
        logger.info("🎉 甄嬛角色模型训练完成！")
        logger.info("📁 接下来可以运行部署脚本: python deployment/huanhuan_deploy.py")
        
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        raise

if __name__ == "__main__":
    main()