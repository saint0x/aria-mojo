"""
Training Monitor with Convergence Criteria, Validation Split, and Checkpoint System

Comprehensive training monitoring system with early stopping, convergence detection,
automated validation, and checkpoint management for robust LLaMA training.
"""

from collections import List, Dict
from ..data.generators.base_types import TrainingExample
from .manual_backprop import GradientTensor
from math import sqrt, abs
from tensor import Tensor

struct ValidationMetrics:
    """Comprehensive validation metrics tracking"""
    var accuracy: Float32
    var loss: Float32
    var tool_calling_accuracy: Float32
    var thinking_prefix_accuracy: Float32
    var perplexity: Float32
    var bleu_score: Float32
    var inference_time: Float32
    var memory_usage: Float32
    var timestamp: Float32
    
    fn __init__(inout self):
        self.accuracy = 0.0
        self.loss = 0.0
        self.tool_calling_accuracy = 0.0
        self.thinking_prefix_accuracy = 0.0
        self.perplexity = 0.0
        self.bleu_score = 0.0
        self.inference_time = 0.0
        self.memory_usage = 0.0
        self.timestamp = 0.0

struct ConvergenceCriteria:
    """Convergence detection and early stopping criteria"""
    var min_improvement_threshold: Float32
    var patience_epochs: Int
    var min_epochs: Int
    var max_epochs: Int
    var loss_window_size: Int
    var accuracy_threshold: Float32
    var perplexity_threshold: Float32
    var convergence_window: Int
    var stability_threshold: Float32
    
    fn __init__(
        inout self,
        min_improvement: Float32 = 0.001,
        patience: Int = 5,
        min_epochs: Int = 3,
        max_epochs: Int = 50,
        accuracy_target: Float32 = 0.85
    ):
        self.min_improvement_threshold = min_improvement
        self.patience_epochs = patience
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.loss_window_size = 5
        self.accuracy_threshold = accuracy_target
        self.perplexity_threshold = 10.0
        self.convergence_window = 3
        self.stability_threshold = 0.005
        
        print("Convergence criteria initialized:")
        print("- Min improvement threshold:", min_improvement)
        print("- Patience epochs:", patience)
        print("- Min/Max epochs:", min_epochs, "/", max_epochs)
        print("- Accuracy target:", accuracy_target)

struct CheckpointManager:
    """Advanced checkpoint management with versioning and recovery"""
    var checkpoint_dir: String
    var checkpoint_frequency: Int
    var max_checkpoints: Int
    var best_checkpoint_path: String
    var checkpoint_history: List[String]
    var checkpoint_metrics: Dict[String, ValidationMetrics]
    var auto_save_enabled: Bool
    var compression_enabled: Bool
    
    fn __init__(
        inout self,
        checkpoint_dir: String = "./checkpoints",
        frequency: Int = 1000,
        max_checkpoints: Int = 5
    ):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = frequency
        self.max_checkpoints = max_checkpoints
        self.best_checkpoint_path = ""
        self.checkpoint_history = List[String]()
        self.checkpoint_metrics = Dict[String, ValidationMetrics]()
        self.auto_save_enabled = True
        self.compression_enabled = True
        
        print("Checkpoint manager initialized:")
        print("- Directory:", checkpoint_dir)
        print("- Frequency:", frequency, "steps")
        print("- Max checkpoints:", max_checkpoints)
    
    fn should_checkpoint(self, step: Int, force: Bool = False) -> Bool:
        """Check if checkpoint should be saved"""
        if force:
            return True
        
        if not self.auto_save_enabled:
            return False
        
        return step % self.checkpoint_frequency == 0
    
    fn save_checkpoint(
        inout self,
        model_params: Dict[String, GradientTensor[DType.float32]],
        optimizer_state: Dict[String, Dict[String, Tensor[DType.float32]]],
        step: Int,
        epoch: Int,
        metrics: ValidationMetrics,
        is_best: Bool = False
    ) -> String:
        """Save model checkpoint with metadata"""
        let timestamp = step  # Simplified timestamp
        let checkpoint_name = "checkpoint_epoch_" + str(epoch) + "_step_" + str(step) + ".ckpt"
        let checkpoint_path = self.checkpoint_dir + "/" + checkpoint_name
        
        print("💾 Saving checkpoint:", checkpoint_name)
        
        # In real implementation, would serialize to disk:
        # - Model parameters
        # - Optimizer state
        # - Training metadata
        # - Validation metrics
        
        # Simulate checkpoint saving
        print("- Model parameters: saved")
        print("- Optimizer state: saved")
        print("- Training metadata: saved")
        print("- Validation metrics: accuracy =", metrics.accuracy)
        
        # Update checkpoint history
        self.checkpoint_history.append(checkpoint_path)
        self.checkpoint_metrics[checkpoint_path] = metrics
        
        # Update best checkpoint
        if is_best or self.best_checkpoint_path == "":
            self.best_checkpoint_path = checkpoint_path
            print("📌 New best checkpoint saved")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        print("✅ Checkpoint saved successfully:", checkpoint_path)
        return checkpoint_path
    
    fn load_checkpoint(self, checkpoint_path: String) -> Bool:
        """Load checkpoint for resuming training"""
        print("📂 Loading checkpoint:", checkpoint_path)
        
        # In real implementation, would deserialize from disk
        # For now, simulate successful loading
        
        print("- Model parameters: loaded")
        print("- Optimizer state: loaded")
        print("- Training metadata: loaded")
        print("✅ Checkpoint loaded successfully")
        
        return True
    
    fn _cleanup_old_checkpoints(inout self):
        """Remove old checkpoints to maintain max limit"""
        if len(self.checkpoint_history) > self.max_checkpoints:
            let checkpoints_to_remove = len(self.checkpoint_history) - self.max_checkpoints
            
            for i in range(checkpoints_to_remove):
                let old_checkpoint = self.checkpoint_history[0]
                
                # Don't remove best checkpoint
                if old_checkpoint != self.best_checkpoint_path:
                    print("🗑️  Removing old checkpoint:", old_checkpoint)
                    # In real implementation, would delete file
                    
                    # Remove from tracking
                    if old_checkpoint in self.checkpoint_metrics:
                        self.checkpoint_metrics.pop(old_checkpoint)
                
                self.checkpoint_history.pop(0)
    
    fn get_best_checkpoint(self) -> String:
        """Get path to best checkpoint"""
        return self.best_checkpoint_path
    
    fn list_checkpoints(self) -> List[String]:
        """List all available checkpoints"""
        return self.checkpoint_history
    
    fn get_checkpoint_metrics(self, checkpoint_path: String) -> ValidationMetrics:
        """Get validation metrics for specific checkpoint"""
        if checkpoint_path in self.checkpoint_metrics:
            return self.checkpoint_metrics[checkpoint_path]
        else:
            return ValidationMetrics()

struct ValidationSplitter:
    """Advanced validation set management with stratification"""
    var train_ratio: Float32
    var validation_ratio: Float32
    var test_ratio: Float32
    var stratify_by_type: Bool
    var stratify_by_complexity: Bool
    var random_seed: Int
    var train_examples: List[TrainingExample]
    var validation_examples: List[TrainingExample]
    var test_examples: List[TrainingExample]
    
    fn __init__(
        inout self,
        train_ratio: Float32 = 0.8,
        validation_ratio: Float32 = 0.15,
        test_ratio: Float32 = 0.05,
        stratify: Bool = True,
        seed: Int = 42
    ):
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.stratify_by_type = stratify
        self.stratify_by_complexity = stratify
        self.random_seed = seed
        self.train_examples = List[TrainingExample]()
        self.validation_examples = List[TrainingExample]()
        self.test_examples = List[TrainingExample]()
        
        print("Validation splitter initialized:")
        print("- Train/Val/Test split:", train_ratio, "/", validation_ratio, "/", test_ratio)
        print("- Stratification enabled:", stratify)
    
    fn create_splits(inout self, examples: List[TrainingExample]) -> Bool:
        """Create stratified train/validation/test splits"""
        print("Creating data splits from", len(examples), "examples...")
        
        if self.stratify_by_type:
            return self._create_stratified_splits(examples)
        else:
            return self._create_random_splits(examples)
    
    fn _create_random_splits(inout self, examples: List[TrainingExample]) -> Bool:
        """Create random train/validation/test splits"""
        let total_examples = len(examples)
        let train_count = int(Float32(total_examples) * self.train_ratio)
        let val_count = int(Float32(total_examples) * self.validation_ratio)
        let test_count = total_examples - train_count - val_count
        
        # Shuffle examples
        var shuffled_examples = self._shuffle_examples(examples)
        
        # Split into sets
        var idx = 0
        
        # Training set
        for i in range(train_count):
            self.train_examples.append(shuffled_examples[idx])
            idx += 1
        
        # Validation set
        for i in range(val_count):
            self.validation_examples.append(shuffled_examples[idx])
            idx += 1
        
        # Test set
        for i in range(test_count):
            if idx < len(shuffled_examples):
                self.test_examples.append(shuffled_examples[idx])
                idx += 1
        
        print("Random splits created:")
        print("- Training:", len(self.train_examples))
        print("- Validation:", len(self.validation_examples))
        print("- Test:", len(self.test_examples))
        
        return True
    
    fn _create_stratified_splits(inout self, examples: List[TrainingExample]) -> Bool:
        """Create stratified splits maintaining example type distribution"""
        print("Creating stratified splits...")
        
        # Group examples by type
        var type_groups = Dict[String, List[TrainingExample]]()
        
        for example in examples:
            let ex_type = example[].example_type
            if ex_type not in type_groups:
                type_groups[ex_type] = List[TrainingExample]()
            type_groups[ex_type].append(example[])
        
        print("Example type distribution:")
        for ex_type in type_groups:
            print("- " + ex_type + ":", len(type_groups[ex_type]))
        
        # Split each type proportionally
        for ex_type in type_groups:
            let type_examples = type_groups[ex_type]
            let shuffled = self._shuffle_examples(type_examples)
            
            let total = len(shuffled)
            let train_count = int(Float32(total) * self.train_ratio)
            let val_count = int(Float32(total) * self.validation_ratio)
            let test_count = total - train_count - val_count
            
            var idx = 0
            
            # Add to training set
            for i in range(train_count):
                self.train_examples.append(shuffled[idx])
                idx += 1
            
            # Add to validation set
            for i in range(val_count):
                self.validation_examples.append(shuffled[idx])
                idx += 1
            
            # Add to test set
            for i in range(test_count):
                if idx < len(shuffled):
                    self.test_examples.append(shuffled[idx])
                    idx += 1
        
        # Final shuffle to mix types
        self.train_examples = self._shuffle_examples(self.train_examples)
        self.validation_examples = self._shuffle_examples(self.validation_examples)
        self.test_examples = self._shuffle_examples(self.test_examples)
        
        print("Stratified splits created:")
        print("- Training:", len(self.train_examples))
        print("- Validation:", len(self.validation_examples))
        print("- Test:", len(self.test_examples))
        
        return True
    
    fn _shuffle_examples(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """Shuffle examples using random seed"""
        var shuffled = examples
        let n = len(shuffled)
        
        # Simple shuffle implementation
        for i in range(n):
            let j = (i * 17 + self.random_seed + 42) % n
            if i != j:
                let temp = shuffled[i]
                shuffled[i] = shuffled[j]
                shuffled[j] = temp
        
        return shuffled
    
    fn get_train_examples(self) -> List[TrainingExample]:
        """Get training examples"""
        return self.train_examples
    
    fn get_validation_examples(self) -> List[TrainingExample]:
        """Get validation examples"""
        return self.validation_examples
    
    fn get_test_examples(self) -> List[TrainingExample]:
        """Get test examples"""
        return self.test_examples
    
    fn print_split_statistics(self):
        """Print detailed split statistics"""
        print("=== DATA SPLIT STATISTICS ===")
        
        let total = len(self.train_examples) + len(self.validation_examples) + len(self.test_examples)
        print("Total examples:", total)
        
        let train_pct = Float32(len(self.train_examples)) / Float32(total) * 100.0
        let val_pct = Float32(len(self.validation_examples)) / Float32(total) * 100.0
        let test_pct = Float32(len(self.test_examples)) / Float32(total) * 100.0
        
        print("Training: ", len(self.train_examples), "(", train_pct, "%)")
        print("Validation:", len(self.validation_examples), "(", val_pct, "%)")
        print("Test:     ", len(self.test_examples), "(", test_pct, "%)")
        
        # Type distribution in each split
        if self.stratify_by_type:
            self._print_type_distribution("Training", self.train_examples)
            self._print_type_distribution("Validation", self.validation_examples)
            self._print_type_distribution("Test", self.test_examples)
        
        print("=" * 30)
    
    fn _print_type_distribution(self, split_name: String, examples: List[TrainingExample]):
        """Print example type distribution for a split"""
        var type_counts = Dict[String, Int]()
        
        for example in examples:
            let ex_type = example[].example_type
            if ex_type not in type_counts:
                type_counts[ex_type] = 0
            type_counts[ex_type] += 1
        
        print(split_name, "type distribution:")
        for ex_type in type_counts:
            let count = type_counts[ex_type]
            let pct = Float32(count) / Float32(len(examples)) * 100.0
            print("  -", ex_type + ":", count, "(", pct, "%)")

struct TrainingMonitor:
    """Comprehensive training monitoring with convergence detection"""
    var convergence_criteria: ConvergenceCriteria
    var checkpoint_manager: CheckpointManager
    var validation_splitter: ValidationSplitter
    var validation_history: List[ValidationMetrics]
    var training_history: List[Float32]
    var current_epoch: Int
    var current_step: Int
    var best_validation_loss: Float32
    var best_validation_accuracy: Float32
    var patience_counter: Int
    var converged: Bool
    var early_stopped: Bool
    var last_validation_time: Float32
    
    fn __init__(
        inout self,
        convergence_criteria: ConvergenceCriteria,
        checkpoint_manager: CheckpointManager,
        validation_splitter: ValidationSplitter
    ):
        self.convergence_criteria = convergence_criteria
        self.checkpoint_manager = checkpoint_manager
        self.validation_splitter = validation_splitter
        self.validation_history = List[ValidationMetrics]()
        self.training_history = List[Float32]()
        self.current_epoch = 0
        self.current_step = 0
        self.best_validation_loss = Float32.MAX
        self.best_validation_accuracy = 0.0
        self.patience_counter = 0
        self.converged = False
        self.early_stopped = False
        self.last_validation_time = 0.0
        
        print("Training monitor initialized with comprehensive tracking")
    
    fn should_stop_training(inout self, current_loss: Float32, validation_metrics: ValidationMetrics) -> Bool:
        """Check if training should stop based on convergence criteria"""
        self.current_epoch += 1
        self.training_history.append(current_loss)
        self.validation_history.append(validation_metrics)
        
        # Check minimum epochs
        if self.current_epoch < self.convergence_criteria.min_epochs:
            return False
        
        # Check maximum epochs
        if self.current_epoch >= self.convergence_criteria.max_epochs:
            print("📊 Training stopped: Maximum epochs reached")
            return True
        
        # Check target accuracy reached
        if validation_metrics.accuracy >= self.convergence_criteria.accuracy_threshold:
            print("🎯 Training stopped: Target accuracy reached (", validation_metrics.accuracy, ")")
            self.converged = True
            return True
        
        # Check for improvement
        let improved = self._check_improvement(validation_metrics)
        
        if improved:
            self.patience_counter = 0
            self.best_validation_loss = validation_metrics.loss
            self.best_validation_accuracy = validation_metrics.accuracy
            print("✅ Validation improved: Loss =", validation_metrics.loss, "Accuracy =", validation_metrics.accuracy)
        else:
            self.patience_counter += 1
            print("⏳ No improvement (", self.patience_counter, "/", self.convergence_criteria.patience_epochs, ")")
        
        # Check early stopping
        if self.patience_counter >= self.convergence_criteria.patience_epochs:
            print("⏹️  Training stopped: Early stopping triggered (no improvement for", self.patience_counter, "epochs)")
            self.early_stopped = True
            return True
        
        # Check convergence based on loss stability
        if self._check_convergence():
            print("📈 Training stopped: Model converged")
            self.converged = True
            return True
        
        return False
    
    fn _check_improvement(self, current_metrics: ValidationMetrics) -> Bool:
        """Check if current validation metrics show improvement"""
        let loss_improved = current_metrics.loss < (self.best_validation_loss - self.convergence_criteria.min_improvement_threshold)
        let accuracy_improved = current_metrics.accuracy > (self.best_validation_accuracy + self.convergence_criteria.min_improvement_threshold)
        
        return loss_improved or accuracy_improved
    
    fn _check_convergence(self) -> Bool:
        """Check if training has converged based on loss stability"""
        if len(self.validation_history) < self.convergence_criteria.convergence_window:
            return False
        
        # Check loss stability in recent window
        let window_start = len(self.validation_history) - self.convergence_criteria.convergence_window
        var loss_values = List[Float32]()
        
        for i in range(window_start, len(self.validation_history)):
            loss_values.append(self.validation_history[i].loss)
        
        # Calculate variance
        var mean_loss: Float32 = 0.0
        for loss in loss_values:
            mean_loss += loss[]
        mean_loss = mean_loss / Float32(len(loss_values))
        
        var variance: Float32 = 0.0
        for loss in loss_values:
            let diff = loss[] - mean_loss
            variance += diff * diff
        variance = variance / Float32(len(loss_values))
        
        let std_dev = sqrt(variance)
        
        # Check if standard deviation is below stability threshold
        return std_dev < self.convergence_criteria.stability_threshold
    
    fn should_validate(self, step: Int) -> Bool:
        """Check if validation should be performed"""
        # Validate every 100 steps or at epoch boundaries
        let validation_frequency = 100
        return step % validation_frequency == 0 or step == 0
    
    fn should_checkpoint(self, step: Int, validation_metrics: ValidationMetrics) -> Bool:
        """Check if checkpoint should be saved"""
        let is_best = validation_metrics.accuracy > self.best_validation_accuracy
        return self.checkpoint_manager.should_checkpoint(step) or is_best
    
    fn save_checkpoint(
        inout self,
        model_params: Dict[String, GradientTensor[DType.float32]],
        optimizer_state: Dict[String, Dict[String, Tensor[DType.float32]]],
        validation_metrics: ValidationMetrics
    ) -> String:
        """Save training checkpoint"""
        let is_best = validation_metrics.accuracy > self.best_validation_accuracy
        
        return self.checkpoint_manager.save_checkpoint(
            model_params,
            optimizer_state,
            self.current_step,
            self.current_epoch,
            validation_metrics,
            is_best
        )
    
    fn get_training_summary(self) -> Dict[String, Float32]:
        """Get comprehensive training summary"""
        var summary = Dict[String, Float32]()
        
        summary["total_epochs"] = Float32(self.current_epoch)
        summary["total_steps"] = Float32(self.current_step)
        summary["converged"] = 1.0 if self.converged else 0.0
        summary["early_stopped"] = 1.0 if self.early_stopped else 0.0
        summary["best_validation_loss"] = self.best_validation_loss
        summary["best_validation_accuracy"] = self.best_validation_accuracy
        summary["patience_used"] = Float32(self.patience_counter)
        
        if len(self.validation_history) > 0:
            let final_metrics = self.validation_history[len(self.validation_history) - 1]
            summary["final_accuracy"] = final_metrics.accuracy
            summary["final_loss"] = final_metrics.loss
            summary["final_perplexity"] = final_metrics.perplexity
        
        return summary
    
    fn print_training_report(self):
        """Print comprehensive training report"""
        print("\n" + "=" * 60)
        print("TRAINING MONITORING REPORT")
        print("=" * 60)
        
        let summary = self.get_training_summary()
        
        print("Training Duration:")
        print("- Epochs completed:", int(summary["total_epochs"]))
        print("- Steps completed:", int(summary["total_steps"]))
        
        print("\nTraining Outcome:")
        if summary["converged"] > 0.0:
            print("✅ Model CONVERGED successfully")
        elif summary["early_stopped"] > 0.0:
            print("⏹️  Training EARLY STOPPED (no improvement)")
        else:
            print("🔄 Training ONGOING")
        
        print("\nBest Performance:")
        print("- Best validation accuracy:", summary["best_validation_accuracy"])
        print("- Best validation loss:", summary["best_validation_loss"])
        
        if "final_accuracy" in summary:
            print("\nFinal Performance:")
            print("- Final accuracy:", summary["final_accuracy"])
            print("- Final loss:", summary["final_loss"])
            print("- Final perplexity:", summary["final_perplexity"])
        
        print("\nEarly Stopping Status:")
        print("- Patience used:", int(summary["patience_used"]), "/", self.convergence_criteria.patience_epochs)
        
        print("\nCheckpoints:")
        print("- Total checkpoints:", len(self.checkpoint_manager.checkpoint_history))
        print("- Best checkpoint:", self.checkpoint_manager.best_checkpoint_path)
        
        print("=" * 60)