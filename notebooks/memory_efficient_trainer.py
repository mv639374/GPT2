"""
Production-ready memory-optimized training module for GPT-2 355M
Implements multiple strategies to handle GPU memory constraints
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from typing import Optional, Tuple, List
import gc


class CheckpointedTransformerBlock(nn.Module):
    """Wrapper for transformer blocks with gradient checkpointing"""
    
    def __init__(self, block):
        super().__init__()
        self.block = block
    
    def forward(self, x):
        if self.training:
            # Use gradient checkpointing during training
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                x,
                use_reentrant=False
            )
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        return self.block(x)


class MemoryEfficientTrainer:
    """
    Memory-efficient trainer with multiple optimization strategies:
    1. 8-bit Adam optimizer (reduces optimizer memory by 75%)
    2. Gradient checkpointing (trades compute for memory)
    3. Mixed precision training
    4. Aggressive memory management
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.00005,
        weight_decay: float = 0.1,
        accumulation_steps: int = 8,
        use_8bit_optimizer: bool = True,
        use_gradient_checkpointing: bool = True
    ):
        self.model = model
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.scaler = GradScaler('cuda')
        
        # Enable gradient checkpointing to save memory
        if use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Use 8-bit Adam to reduce optimizer memory footprint
        if use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    model.parameters(),
                    lr=learning_rate / accumulation_steps,
                    weight_decay=weight_decay,
                    betas=(0.9, 0.95)
                )
                print("✓ Using 8-bit AdamW optimizer (75% memory reduction)")
            except ImportError:
                print("⚠ bitsandbytes not installed, falling back to standard AdamW")
                print("  Install with: pip install bitsandbytes")
                self.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate / accumulation_steps,
                    weight_decay=weight_decay
                )
        else:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate / accumulation_steps,
                weight_decay=weight_decay
            )
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for transformer blocks"""
        if hasattr(self.model, 'trf_blocks'):
            # Wrap each transformer block with checkpointing
            checkpointed_blocks = nn.Sequential(*[
                CheckpointedTransformerBlock(block)
                for block in self.model.trf_blocks
            ])
            self.model.trf_blocks = checkpointed_blocks
            print("✓ Gradient checkpointing enabled")
        else:
            print("⚠ Model doesn't have 'trf_blocks', skipping gradient checkpointing")
    
    def train_epoch(
        self,
        train_loader,
        val_loader,
        eval_freq: int,
        eval_iter: int,
        start_context: str,
        tokenizer,
        epoch: int
    ) -> Tuple[List[float], List[float], List[int]]:
        """Train for one epoch with memory-efficient practices"""
        
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1
        
        self.model.train()
        
        for i, (input_batch, target_batch) in enumerate(train_loader):
            # Mixed precision forward pass
            with autocast('cuda'):
                loss = self._calc_loss_batch(input_batch, target_batch)
                loss = loss / self.accumulation_steps
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            tokens_seen += input_batch.numel()
            
            # Update weights every accumulation_steps
            if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(train_loader):
                # Gradient clipping for stability
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
                global_step += 1
                
                # Evaluation
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self._evaluate_model(
                        train_loader, val_loader, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                
                # Aggressive memory cleanup
                if i % 20 == 0:
                    self._cleanup_memory()
        
        # Generate sample text
        self._generate_and_print_sample(start_context, tokenizer)
        
        return train_losses, val_losses, track_tokens_seen
    
    def _calc_loss_batch(self, input_batch, target_batch):
        """Calculate loss for a batch"""
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target_batch.flatten()
        )
        return loss
    
    def _evaluate_model(self, train_loader, val_loader, eval_iter):
        """Evaluate model on train and validation sets"""
        self.model.eval()
        with torch.no_grad():
            train_loss = self._calc_loss_loader(train_loader, eval_iter)
            val_loss = self._calc_loss_loader(val_loader, eval_iter)
        self.model.train()
        return train_loss, val_loss
    
    def _calc_loss_loader(self, data_loader, num_batches):
        """Calculate average loss over data loader"""
        total_loss = 0.
        num_batches = min(num_batches, len(data_loader))
        
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break
            with autocast('cuda'):
                loss = self._calc_loss_batch(input_batch, target_batch)
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def _generate_and_print_sample(self, start_context, tokenizer):
        """Generate and print sample text"""
        from previous_chapters import text_to_token_ids, token_ids_to_text, generate_text_simple
        
        self.model.eval()
        context_size = self.model.pos_emb.weight.shape[0]
        encoded = text_to_token_ids(start_context, tokenizer).to(self.device)
        
        with torch.no_grad():
            token_ids = generate_text_simple(
                model=self.model,
                idx=encoded,
                max_new_tokens=50,
                context_size=context_size
            )
            decoded_text = token_ids_to_text(token_ids, tokenizer)
            print(decoded_text.replace("\n", " "))
        
        self.model.train()
    
    @staticmethod
    def _cleanup_memory():
        """Aggressively cleanup GPU memory"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def train_memory_efficient(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs: int = 2,
    eval_freq: int = 5,
    eval_iter: int = 5,
    start_context: str = "",
    tokenizer = None,
    learning_rate: float = 0.00005,
    accumulation_steps: int = 8,
    use_8bit_optimizer: bool = True,
    use_gradient_checkpointing: bool = True
):
    """
    Memory-efficient training function for GPT-2 355M
    
    Args:
        model: GPT model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Number of training epochs
        eval_freq: Evaluation frequency (in steps)
        eval_iter: Number of batches for evaluation
        start_context: Context for text generation
        tokenizer: Tokenizer for text generation
        learning_rate: Base learning rate
        accumulation_steps: Gradient accumulation steps (higher = less memory)
        use_8bit_optimizer: Use 8-bit Adam (requires bitsandbytes)
        use_gradient_checkpointing: Enable gradient checkpointing
    
    Returns:
        train_losses, val_losses, tokens_seen
    """
    
    # Initial memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Print memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n{'='*60}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Accumulation Steps: {accumulation_steps}")
    print(f"Effective Batch Size: {accumulation_steps}")
    print(f"{'='*60}\n")
    
    # Initialize trainer
    trainer = MemoryEfficientTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        accumulation_steps=accumulation_steps,
        use_8bit_optimizer=use_8bit_optimizer,
        use_gradient_checkpointing=use_gradient_checkpointing
    )
    
    # Training loop
    all_train_losses, all_val_losses, all_tokens_seen = [], [], []
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        train_losses, val_losses, tokens_seen = trainer.train_epoch(
            train_loader=train_loader,
            val_loader=val_loader,
            eval_freq=eval_freq,
            eval_iter=eval_iter,
            start_context=start_context,
            tokenizer=tokenizer,
            epoch=epoch
        )
        
        all_train_losses.extend(train_losses)
        all_val_losses.extend(val_losses)
        all_tokens_seen.extend(tokens_seen)
        
        # Memory cleanup between epochs
        trainer._cleanup_memory()
    
    return all_train_losses, all_val_losses, all_tokens_seen


# Additional utility functions for production use

def estimate_memory_requirements(model, batch_size=1, seq_length=1024):
    """
    Estimate memory requirements for training
    
    Returns:
        dict with memory estimates in GB
    """
    # Calculate model size
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    
    # Estimate activations (rough approximation)
    activation_memory = (batch_size * seq_length * 1024 * 4) / 1024**3  # 4 bytes per float32
    
    # Optimizer state (Adam = 2x parameters for momentum buffers)
    optimizer_memory_standard = param_memory * 2
    optimizer_memory_8bit = param_memory * 0.5  # 8-bit Adam saves ~75%
    
    # Gradients
    gradient_memory = param_memory
    
    return {
        "model_parameters": param_memory,
        "gradients": gradient_memory,
        "activations_estimate": activation_memory,
        "optimizer_standard": optimizer_memory_standard,
        "optimizer_8bit": optimizer_memory_8bit,
        "total_standard": param_memory + gradient_memory + activation_memory + optimizer_memory_standard,
        "total_8bit": param_memory + gradient_memory + activation_memory + optimizer_memory_8bit
    }


def print_memory_usage():
    """Print current GPU memory usage"""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"\nGPU Memory:")
    print(f"  Allocated: {allocated:.2f} GB / {total:.2f} GB ({allocated/total*100:.1f}%)")
    print(f"  Reserved:  {reserved:.2f} GB / {total:.2f} GB ({reserved/total*100:.1f}%)")
    print(f"  Free:      {total - allocated:.2f} GB")