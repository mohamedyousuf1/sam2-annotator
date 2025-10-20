"""
SAM2 Model loading and initialization
"""
import os
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from config import MODEL_CHECKPOINT_PATH, MODEL_CONFIG, DEVICE


class ModelLoader:
    """Handles loading and initialization of SAM2 model"""
    
    @staticmethod
    def load_sam2_model():
        """
        Load SAM2 model from checkpoint and config files
        
        Returns:
            SAM2ImagePredictor or None: Initialized predictor or None if loading fails
        """
        print("Loading SAM2 model...")
        
        # Check if model files exist
        if not os.path.exists(MODEL_CHECKPOINT_PATH):
            print(f"❌ Model checkpoint not found: {MODEL_CHECKPOINT_PATH}")
            return None
            
        if not os.path.exists(MODEL_CONFIG):
            print(f"❌ Model config not found: {MODEL_CONFIG}")
            return None
        
        try:
            # Build and load the model
            sam_model = build_sam2(MODEL_CONFIG)
            sam_model.to(DEVICE)
            
            # Load checkpoint
            checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
            state_dict = checkpoint.get('model', checkpoint)
            sam_model.load_state_dict(state_dict)
            
            # Create predictor
            predictor = SAM2ImagePredictor(sam_model)
            print(f"✅ Model loaded successfully on {DEVICE}.")
            
            return predictor
            
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            return None
    
    @staticmethod
    def check_model_files():
        """
        Check if model files exist
        
        Returns:
            tuple: (bool, str) - (files_exist, error_message)
        """
        if not os.path.exists(MODEL_CHECKPOINT_PATH):
            return False, f"Model checkpoint not found: {MODEL_CHECKPOINT_PATH}"
            
        if not os.path.exists(MODEL_CONFIG):
            return False, f"Model config not found: {MODEL_CONFIG}"
            
        return True, ""
