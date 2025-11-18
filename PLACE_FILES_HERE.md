# 📁 AI Models Placement Guide

## 🎯 Where to Place Your Files

### **Your .pt RT-DETR file:**
Place it as: `ai_models/custom_finetuned.pt`

**OR**

Place it as: `ai_models/RTDETR.pt`

### **Your .t7 DeepSORT file:**
Place it as: `ai_models/deepsort/ckpt.t7`

## 📂 Final Structure Should Look Like:
```
MPT-Karan-5/
├── ai_models/
│   ├── custom_finetuned.pt    ← Your .pt RT-DETR file (recommended name)
│   └── deepsort/
│       └── ckpt.t7            ← Your .t7 DeepSORT weights file
├── complete_dashboard.py
├── local_ai_engine.py
└── ...
```

## 🚀 Steps to Setup:

1. **Copy your .pt file** to: `ai_models/custom_finetuned.pt`
2. **Copy your .t7 file** to: `ai_models/deepsort/ckpt.t7`
3. **Test setup**: `python -c "from local_ai_engine import LocalAIEngine; LocalAIEngine()"`
4. **Launch dashboard**: `python complete_dashboard.py`

## ✅ What Will Happen:
- System will detect your fine-tuned model
- DeepSORT tracking will be enabled
- Dashboard will show "Fine-tuned RT-DETR + DeepSORT ACTIVE"
- Real AI processing on your videos

**Ready to copy your files!**