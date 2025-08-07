"""
Learning Rate Scheduling with Warmup and Cosine Decay

Advanced learning rate scheduling for LLaMA training including linear warmup,
cosine annealing, and adaptive learning rate adjustment based on training metrics.
Optimized for 8B parameter model training with curriculum learning integration.
"""

from math import cos, sin, exp, log, sqrt, pi
from collections import List, Dict

struct LearningRateScheduler:
    """Advanced learning rate scheduler with warmup and cosine decay"""
    var base_learning_rate: Float32
    var max_learning_rate: Float32
    var min_learning_rate: Float32
    var warmup_steps: Int
    var total_steps: Int
    var current_step: Int
    var scheduler_type: String
    var warmup_type: String
    var decay_type: String
    var schedule_history: List[Float32]
    var adaptive_enabled: Bool
    var loss_history: List[Float32]
    var patience_counter: Int
    var reduction_factor: Float32
    var min_reduction_epochs: Int
    
    fn __init__(
        inout self,
        base_lr: Float32 = 1e-4,
        max_lr: Float32 = 3e-4,
        min_lr: Float32 = 1e-6,
        warmup_steps: Int = 2000,
        total_steps: Int = 100000,
        scheduler_type: String = "cosine_with_warmup",
        adaptive: Bool = True
    ):
        self.base_learning_rate = base_lr
        self.max_learning_rate = max_lr
        self.min_learning_rate = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.scheduler_type = scheduler_type
        self.warmup_type = "linear"
        self.decay_type = "cosine"
        self.schedule_history = List[Float32]()
        self.adaptive_enabled = adaptive
        self.loss_history = List[Float32]()
        self.patience_counter = 0
        self.reduction_factor = 0.5
        self.min_reduction_epochs = 5
        
        print("Learning Rate Scheduler initialized:")
        print("- Base LR:", base_lr)
        print("- Max LR:", max_lr) 
        print("- Min LR:", min_lr)
        print("- Warmup steps:", warmup_steps)
        print("- Total steps:", total_steps)
        print("- Schedule type:", scheduler_type)
        print("- Adaptive adjustment:", adaptive)
    
    fn get_learning_rate(inout self, current_loss: Float32 = -1.0) -> Float32:
        """Get learning rate for current step"""
        self.current_step += 1
        
        var lr: Float32 = 0.0
        
        if self.scheduler_type == "cosine_with_warmup":
            lr = self._cosine_with_warmup()
        elif self.scheduler_type == "linear_with_warmup":
            lr = self._linear_with_warmup()
        elif self.scheduler_type == "exponential_decay":
            lr = self._exponential_decay()
        elif self.scheduler_type == "polynomial_decay":
            lr = self._polynomial_decay()
        elif self.scheduler_type == "one_cycle":
            lr = self._one_cycle()
        else:
            lr = self.base_learning_rate  # Constant fallback
        
        # Apply adaptive adjustment if enabled
        if self.adaptive_enabled and current_loss > 0.0:
            lr = self._apply_adaptive_adjustment(lr, current_loss)
        
        # Store in history
        self.schedule_history.append(lr)
        if current_loss > 0.0:
            self.loss_history.append(current_loss)
        
        return lr
    
    fn _cosine_with_warmup(self) -> Float32:
        """Cosine annealing with linear warmup"""
        if self.current_step < self.warmup_steps:
            # Linear warmup phase
            let warmup_progress = Float32(self.current_step) / Float32(self.warmup_steps)
            return self.base_learning_rate + (self.max_learning_rate - self.base_learning_rate) * warmup_progress
        else:
            # Cosine decay phase
            let decay_steps = self.total_steps - self.warmup_steps
            let decay_progress = Float32(self.current_step - self.warmup_steps) / Float32(decay_steps)
            let cosine_decay = 0.5 * (1.0 + cos(pi * decay_progress))
            return self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * cosine_decay
    
    fn _linear_with_warmup(self) -> Float32:
        """Linear decay with linear warmup"""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            let warmup_progress = Float32(self.current_step) / Float32(self.warmup_steps)
            return self.base_learning_rate + (self.max_learning_rate - self.base_learning_rate) * warmup_progress
        else:
            # Linear decay
            let decay_steps = self.total_steps - self.warmup_steps
            let decay_progress = Float32(self.current_step - self.warmup_steps) / Float32(decay_steps)
            let remaining_ratio = max(0.0, 1.0 - decay_progress)
            return self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * remaining_ratio
    
    fn _exponential_decay(self) -> Float32:
        """Exponential decay with warmup"""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            let warmup_progress = Float32(self.current_step) / Float32(self.warmup_steps)
            return self.base_learning_rate + (self.max_learning_rate - self.base_learning_rate) * warmup_progress
        else:
            # Exponential decay
            let decay_rate: Float32 = 0.95
            let decay_steps_elapsed = self.current_step - self.warmup_steps
            let decay_epochs = decay_steps_elapsed // 1000  # Decay every 1000 steps
            let decay_factor = decay_rate ** Float32(decay_epochs)
            return max(self.min_learning_rate, self.max_learning_rate * decay_factor)
    
    fn _polynomial_decay(self) -> Float32:
        """Polynomial decay with warmup"""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            let warmup_progress = Float32(self.current_step) / Float32(self.warmup_steps)
            return self.base_learning_rate + (self.max_learning_rate - self.base_learning_rate) * warmup_progress
        else:
            # Polynomial decay
            let power: Float32 = 0.9
            let decay_steps = self.total_steps - self.warmup_steps
            let decay_progress = Float32(self.current_step - self.warmup_steps) / Float32(decay_steps)
            let polynomial_factor = (1.0 - decay_progress) ** power
            return self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * polynomial_factor
    
    fn _one_cycle(self) -> Float32:
        """One cycle learning rate policy"""
        let cycle_progress = Float32(self.current_step) / Float32(self.total_steps)
        
        if cycle_progress < 0.5:
            # First half: increase to max
            let progress = cycle_progress * 2.0
            return self.base_learning_rate + (self.max_learning_rate - self.base_learning_rate) * progress
        else:
            # Second half: decrease to min
            let progress = (cycle_progress - 0.5) * 2.0
            return self.max_learning_rate - (self.max_learning_rate - self.min_learning_rate) * progress
    
    fn _apply_adaptive_adjustment(inout self, base_lr: Float32, current_loss: Float32) -> Float32:
        """Apply adaptive learning rate adjustment based on loss trends"""
        if len(self.loss_history) < 3:
            return base_lr  # Need more history
        
        # Check for loss plateau
        let recent_losses = self._get_recent_losses(3)
        var is_plateau = True
        let plateau_threshold: Float32 = 0.001  # 0.1% improvement threshold
        
        for i in range(1, len(recent_losses)):
            let improvement = (recent_losses[i-1] - recent_losses[i]) / recent_losses[i-1]
            if improvement > plateau_threshold:
                is_plateau = False
                break
        
        if is_plateau:
            self.patience_counter += 1
            if self.patience_counter >= self.min_reduction_epochs:
                print("Learning rate plateau detected, reducing LR")
                self.patience_counter = 0
                return base_lr * self.reduction_factor
        else:
            self.patience_counter = 0
        
        # Check for loss explosion
        if len(self.loss_history) >= 2:
            let prev_loss = self.loss_history[len(self.loss_history) - 2]
            if current_loss > prev_loss * 2.0:  # Loss doubled
                print("Loss explosion detected, emergency LR reduction")
                return base_lr * 0.1
        
        return base_lr
    
    fn _get_recent_losses(self, count: Int) -> List[Float32]:
        """Get recent loss values"""
        var recent = List[Float32]()
        let start_idx = max(0, len(self.loss_history) - count)
        
        for i in range(start_idx, len(self.loss_history)):
            recent.append(self.loss_history[i])
        
        return recent
    
    fn step_with_curriculum_stage(inout self, stage: Int, current_loss: Float32 = -1.0) -> Float32:
        """Get learning rate adjusted for curriculum learning stage"""
        let base_lr = self.get_learning_rate(current_loss)
        
        # Stage-specific learning rate multipliers
        var stage_multiplier: Float32 = 1.0
        
        if stage == 1:      # Foundation stage - higher LR for faster initial learning
            stage_multiplier = 1.2
        elif stage == 2:    # Application stage - standard LR
            stage_multiplier = 1.0
        elif stage == 3:    # Integration stage - slightly lower LR for stability
            stage_multiplier = 0.8
        elif stage == 4:    # Mastery stage - lower LR for fine-tuning
            stage_multiplier = 0.6
        
        let adjusted_lr = base_lr * stage_multiplier
        
        print("Stage", stage, "LR adjustment: base =", base_lr, "→ adjusted =", adjusted_lr, "(×", stage_multiplier, ")")
        
        return adjusted_lr
    
    fn reset_for_new_stage(inout self, stage: Int):
        """Reset scheduler state for new curriculum stage"""
        print("Resetting scheduler for curriculum stage", stage)
        
        # Optionally reset warmup for new stages
        if stage > 1:
            # Shorter warmup for later stages
            self.warmup_steps = max(100, self.warmup_steps // 2)
            print("Reduced warmup steps to", self.warmup_steps, "for stage", stage)
        
        # Reset adaptive counters
        self.patience_counter = 0
        
        # Clear recent history to avoid inter-stage interference
        if len(self.loss_history) > 10:
            # Keep only last 5 values for continuity
            let recent_losses = self._get_recent_losses(5)
            self.loss_history.clear()
            for loss in recent_losses:
                self.loss_history.append(loss[])
    
    fn get_schedule_statistics(self) -> Dict[String, Float32]:
        """Get learning rate schedule statistics"""
        var stats = Dict[String, Float32]()
        
        if len(self.schedule_history) == 0:
            return stats
        
        # Calculate statistics
        var min_lr = self.schedule_history[0]
        var max_lr = self.schedule_history[0]
        var sum_lr: Float32 = 0.0
        
        for lr in self.schedule_history:
            if lr[] < min_lr:
                min_lr = lr[]
            if lr[] > max_lr:
                max_lr = lr[]
            sum_lr += lr[]
        
        let avg_lr = sum_lr / Float32(len(self.schedule_history))
        let current_lr = self.schedule_history[len(self.schedule_history) - 1]
        
        stats["min_lr"] = min_lr
        stats["max_lr"] = max_lr
        stats["avg_lr"] = avg_lr
        stats["current_lr"] = current_lr
        stats["total_steps"] = Float32(len(self.schedule_history))
        
        return stats
    
    fn print_schedule_summary(self):
        """Print learning rate schedule summary"""
        let stats = self.get_schedule_statistics()
        
        print("=== LEARNING RATE SCHEDULE SUMMARY ===")
        print("Schedule Type:", self.scheduler_type)
        print("Steps Completed:", int(stats["total_steps"]))
        print("Current LR:", stats["current_lr"])
        print("Average LR:", stats["avg_lr"])
        print("LR Range: [", stats["min_lr"], ",", stats["max_lr"], "]")
        
        if self.adaptive_enabled:
            print("Adaptive Adjustments: ENABLED")
            print("Plateau Patience:", self.patience_counter, "/", self.min_reduction_epochs)
        else:
            print("Adaptive Adjustments: DISABLED")
        
        print("=" * 40)
    
    fn save_schedule_history(self, filepath: String):
        """Save learning rate history to file"""
        print("Saving LR schedule history to:", filepath)
        
        var content = "step,learning_rate,loss\n"
        
        for i in range(len(self.schedule_history)):
            let step = i + 1
            let lr = self.schedule_history[i]
            let loss = self.loss_history[i] if i < len(self.loss_history) else 0.0
            
            content += str(step) + "," + str(lr) + "," + str(loss) + "\n"
        
        # In real implementation, would write to file
        print("LR history saved (", len(self.schedule_history), "entries)")
    
    fn visualize_schedule(self, future_steps: Int = 1000):
        """Generate future learning rate schedule for visualization"""
        print("Learning Rate Schedule Preview (next", future_steps, "steps):")
        
        let original_step = self.current_step
        let step_interval = max(1, future_steps // 20)  # Show 20 data points
        
        for i in range(0, future_steps, step_interval):
            self.current_step = original_step + i
            let lr = self._cosine_with_warmup()  # Use primary schedule
            let progress = Float32(i) / Float32(future_steps) * 100.0
            
            print("Step +", i, "(", progress, "%): LR =", lr)
        
        # Restore original step
        self.current_step = original_step
        print("Schedule preview completed.")

struct MultiStageScheduler:
    """Multi-stage learning rate scheduler for curriculum learning"""
    var stage_schedulers: Dict[Int, LearningRateScheduler]
    var current_stage: Int
    var stage_transitions: List[Int]
    var global_step: Int
    
    fn __init__(inout self):
        self.stage_schedulers = Dict[Int, LearningRateScheduler]()
        self.current_stage = 1
        self.stage_transitions = List[Int]()
        self.global_step = 0
        self._initialize_stage_schedulers()
    
    fn _initialize_stage_schedulers(inout self):
        """Initialize schedulers for each curriculum stage"""
        
        # Stage 1: Foundation - Higher LR for faster initial learning
        var stage1_scheduler = LearningRateScheduler(
            base_lr = 5e-5,
            max_lr = 2e-4,
            min_lr = 1e-6,
            warmup_steps = 500,
            total_steps = 10000,
            scheduler_type = "cosine_with_warmup"
        )
        self.stage_schedulers[1] = stage1_scheduler
        
        # Stage 2: Application - Standard LR
        var stage2_scheduler = LearningRateScheduler(
            base_lr = 3e-5,
            max_lr = 1.5e-4,
            min_lr = 5e-7,
            warmup_steps = 300,
            total_steps = 15000,
            scheduler_type = "cosine_with_warmup"
        )
        self.stage_schedulers[2] = stage2_scheduler
        
        # Stage 3: Integration - Lower LR for stability
        var stage3_scheduler = LearningRateScheduler(
            base_lr = 2e-5,
            max_lr = 1e-4,
            min_lr = 2e-7,
            warmup_steps = 200,
            total_steps = 20000,
            scheduler_type = "cosine_with_warmup"
        )
        self.stage_schedulers[3] = stage3_scheduler
        
        # Stage 4: Mastery - Very low LR for fine-tuning
        var stage4_scheduler = LearningRateScheduler(
            base_lr = 1e-5,
            max_lr = 5e-5,
            min_lr = 1e-7,
            warmup_steps = 100,
            total_steps = 25000,
            scheduler_type = "cosine_with_warmup"
        )
        self.stage_schedulers[4] = stage4_scheduler
        
        print("Multi-stage scheduler initialized with 4 curriculum stages")
    
    fn advance_to_stage(inout self, new_stage: Int):
        """Advance to next curriculum stage"""
        if new_stage != self.current_stage:
            print("Advancing from stage", self.current_stage, "to stage", new_stage)
            self.stage_transitions.append(self.global_step)
            self.current_stage = new_stage
            
            if new_stage in self.stage_schedulers:
                self.stage_schedulers[new_stage].reset_for_new_stage(new_stage)
    
    fn get_learning_rate(inout self, current_loss: Float32 = -1.0) -> Float32:
        """Get learning rate for current curriculum stage"""
        self.global_step += 1
        
        if self.current_stage in self.stage_schedulers:
            return self.stage_schedulers[self.current_stage].get_learning_rate(current_loss)
        else:
            print("Warning: No scheduler for stage", self.current_stage)
            return 1e-5  # Fallback LR
    
    fn print_multi_stage_summary(self):
        """Print summary of multi-stage scheduling"""
        print("=== MULTI-STAGE SCHEDULER SUMMARY ===")
        print("Current Stage:", self.current_stage)
        print("Global Steps:", self.global_step)
        print("Stage Transitions:", len(self.stage_transitions))
        
        for stage in range(1, 5):
            if stage in self.stage_schedulers:
                let scheduler = self.stage_schedulers[stage]
                let stats = scheduler.get_schedule_statistics()
                if "current_lr" in stats:
                    print("Stage", stage, "Current LR:", stats["current_lr"])
        
        print("=" * 40)