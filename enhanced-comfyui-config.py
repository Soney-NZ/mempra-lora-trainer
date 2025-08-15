# Enhanced ComfyUI Configuration for Mempra
# Based on the ComfyUI notebooks and Ostris AI Toolkit resources

import os
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedComfyUIConfig:
    """Enhanced ComfyUI configuration following the notebook patterns"""
    
    def __init__(self):
        self.base_models = {
            "flux": {
                "model_path": "/workspace/models/checkpoints/flux1-dev.safetensors",
                "vae_path": "/workspace/models/vae/ae.safetensors", 
                "clip_path": "/workspace/models/clip/t5xxl_fp16.safetensors",
                "base_resolution": 1024
            }
        }
        
    def get_training_workflow(self, config):
        """Generate ComfyUI workflow for LoRA training following notebook patterns"""
        workflow = {
            "1": {
                "inputs": {
                    "ckpt_name": "flux1-dev.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": f"training images for {config['trigger_word']}",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "dataset_path": config["zip_url"],
                    "trigger_word": config["trigger_word"],
                    "steps": config.get("steps", 600),
                    "learning_rate": config.get("learning_rate", 0.0002),
                    "network_dim": config.get("network_dimension", 32),
                    "model": ["1", 0],
                    "clip": ["1", 1],
                    "vae": ["1", 2]
                },
                "class_type": "LoRATrainer"
            },
            "4": {
                "inputs": {
                    "filename_prefix": f"lora_{config['model_name']}",
                    "lora_model": ["3", 0]
                },
                "class_type": "SaveLora"
            }
        }
        
        return workflow
    
    def get_training_config(self, model_name, trigger_word, training_type, **kwargs):
        """Generate training configuration following ComfyUI patterns"""
        config = {
            "model_name": model_name,
            "trigger_word": trigger_word,
            "training_type": training_type,
            "steps": kwargs.get("steps", 600),
            "learning_rate": kwargs.get("learning_rate", 0.0002),
            "network_dimension": kwargs.get("network_dimension", 32),
            "batch_size": kwargs.get("batch_size", 1),
            "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 1),
            "optimizer": "adamw8bit",
            "lr_scheduler": "cosine_with_restarts",
            "mixed_precision": "fp16",
            "save_precision": "fp16",
            # Enhanced settings from ComfyUI notebooks
            "enable_xformers": True,
            "cache_latents": True,
            "cache_text_encoder_outputs": True,
            "gradient_checkpointing": True,
            "noise_offset": 0.1,
            "min_snr_gamma": 5.0
        }
        
        # Training type specific settings
        if training_type == "character":
            config.update({
                "steps": 800,
                "learning_rate": 0.0001,
                "network_dimension": 64,
                "network_alpha": 32
            })
        elif training_type == "style":
            config.update({
                "steps": 1000,
                "learning_rate": 0.0003,
                "network_dimension": 32,
                "network_alpha": 16
            })
        
        return config
    
    def validate_training_inputs(self, config):
        """Validate training inputs following ComfyUI requirements"""
        required_fields = ["model_name", "trigger_word", "zip_url", "training_type"]
        
        for field in required_fields:
            if field not in config or not config[field]:
                raise ValueError(f"Missing required field: {field}")
        
        if config["steps"] < 100 or config["steps"] > 2000:
            raise ValueError("Steps must be between 100 and 2000")
            
        if config["learning_rate"] < 0.00001 or config["learning_rate"] > 0.01:
            raise ValueError("Learning rate must be between 0.00001 and 0.01")
            
        if config["network_dimension"] < 8 or config["network_dimension"] > 128:
            raise ValueError("Network dimension must be between 8 and 128")
            
        logger.info(f"✅ Training configuration validated for {config['model_name']}")
        return True

# Global configuration instance
enhanced_config = EnhancedComfyUIConfig()