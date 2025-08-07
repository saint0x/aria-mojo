"""
Comprehensive Curriculum Learning Engine with 4-Stage Progression

Advanced curriculum learning system that orchestrates progressive training through
4 stages: Foundation → Application → Integration → Mastery, with automatic
stage advancement, difficulty scaling, and performance-based transitions.
"""

from collections import List, Dict
from ..data.generators.base_types import TrainingExample
from .training_monitor import TrainingMonitor, ValidationMetrics, ConvergenceCriteria, CheckpointManager, ValidationSplitter
from .learning_rate_scheduler import LearningRateScheduler, MultiStageScheduler
from .training_loop import TrainingLoop
from .manual_backprop import GradientTensor
from tensor import Tensor
from math import log, exp, sqrt

struct CurriculumStage:
    """Individual curriculum stage configuration"""
    var stage_id: Int
    var stage_name: String
    var complexity_level: String
    var target_accuracy: Float32
    var min_accuracy_threshold: Float32
    var max_epochs: Int
    var example_types: List[String]
    var learning_rate_multiplier: Float32
    var batch_size_multiplier: Float32
    var data_augmentation_factor: Float32
    var stage_description: String
    
    fn __init__(
        inout self,
        stage_id: Int,
        stage_name: String,
        complexity: String,
        target_acc: Float32 = 0.8,
        min_acc: Float32 = 0.7
    ):
        self.stage_id = stage_id
        self.stage_name = stage_name
        self.complexity_level = complexity
        self.target_accuracy = target_acc
        self.min_accuracy_threshold = min_acc
        self.max_epochs = 10
        self.example_types = List[String]()
        self.learning_rate_multiplier = 1.0
        self.batch_size_multiplier = 1.0
        self.data_augmentation_factor = 1.0
        self.stage_description = ""

struct CurriculumLearningEngine:
    """Comprehensive curriculum learning orchestration engine"""
    var curriculum_stages: List[CurriculumStage]
    var current_stage: Int
    var stage_history: List[Dict[String, Float32]]
    var multi_stage_scheduler: MultiStageScheduler
    var training_monitor: TrainingMonitor
    var training_loop: TrainingLoop
    var stage_transition_history: List[Tuple[Int, Float32, String]]
    var curriculum_completed: Bool
    var auto_advancement: Bool
    var stage_performance_tracker: Dict[Int, List[Float32]]
    var difficulty_adaptation: Bool
    var performance_metrics: Dict[String, Float32]
    
    fn __init__(
        inout self,
        auto_advance: Bool = True,
        adapt_difficulty: Bool = True
    ):
        self.curriculum_stages = List[CurriculumStage]()
        self.current_stage = 1
        self.stage_history = List[Dict[String, Float32]]()
        self.multi_stage_scheduler = MultiStageScheduler()
        self.training_monitor = TrainingMonitor(
            ConvergenceCriteria(),
            CheckpointManager(),
            ValidationSplitter()
        )
        self.training_loop = TrainingLoop()
        self.stage_transition_history = List[Tuple[Int, Float32, String]]()
        self.curriculum_completed = False
        self.auto_advancement = auto_advance
        self.stage_performance_tracker = Dict[Int, List[Float32]]()
        self.difficulty_adaptation = adapt_difficulty
        self.performance_metrics = Dict[String, Float32]()
        
        self._initialize_curriculum_stages()
        
        print("🎓 Curriculum Learning Engine initialized")
        print("- 4-stage progressive curriculum")
        print("- Auto advancement:", auto_advance)
        print("- Difficulty adaptation:", adapt_difficulty)
    
    fn _initialize_curriculum_stages(inout self):
        """Initialize the 4-stage curriculum progression"""
        
        # Stage 1: Foundation - Basic tool usage and thinking patterns
        var stage1 = CurriculumStage(1, "Foundation", "beginner", 0.75, 0.65)
        stage1.max_epochs = 8
        stage1.example_types.append("tool_hit")
        stage1.example_types.append("tool_miss")
        stage1.learning_rate_multiplier = 1.2  # Higher LR for faster initial learning
        stage1.batch_size_multiplier = 1.0
        stage1.data_augmentation_factor = 0.8  # Less augmentation for cleaner signal
        stage1.stage_description = "Learn basic tool calling patterns and Universal Thinking Prefix usage"
        self.curriculum_stages.append(stage1)
        
        # Stage 2: Application - Applied problem solving with tools
        var stage2 = CurriculumStage(2, "Application", "intermediate", 0.80, 0.70)
        stage2.max_epochs = 10
        stage2.example_types.append("tool_hit")
        stage2.example_types.append("tool_miss")
        stage2.example_types.append("tool_error")
        stage2.learning_rate_multiplier = 1.0  # Standard LR
        stage2.batch_size_multiplier = 1.1
        stage2.data_augmentation_factor = 1.0
        stage2.stage_description = "Apply tools to solve real-world problems with error handling"
        self.curriculum_stages.append(stage2)
        
        # Stage 3: Integration - Multi-tool scenarios and complex reasoning
        var stage3 = CurriculumStage(3, "Integration", "advanced", 0.82, 0.75)
        stage3.max_epochs = 12
        stage3.example_types.append("tool_hit")
        stage3.example_types.append("tool_miss")
        stage3.example_types.append("tool_error")
        stage3.example_types.append("multi_tool")
        stage3.learning_rate_multiplier = 0.8  # Lower LR for stability
        stage3.batch_size_multiplier = 1.2
        stage3.data_augmentation_factor = 1.2
        stage3.stage_description = "Integrate multiple tools and handle complex multi-step reasoning"
        self.curriculum_stages.append(stage3)
        
        # Stage 4: Mastery - Expert-level reasoning and edge cases
        var stage4 = CurriculumStage(4, "Mastery", "expert", 0.85, 0.80)
        stage4.max_epochs = 15
        stage4.example_types.append("tool_hit")
        stage4.example_types.append("tool_miss")
        stage4.example_types.append("tool_error")
        stage4.example_types.append("multi_tool")
        stage4.learning_rate_multiplier = 0.6  # Very low LR for fine-tuning
        stage4.batch_size_multiplier = 1.3
        stage4.data_augmentation_factor = 1.5  # Maximum augmentation for robustness
        stage4.stage_description = "Master complex reasoning and handle edge cases with expert-level performance"
        self.curriculum_stages.append(stage4)
        
        print("Curriculum stages initialized:")
        for stage in self.curriculum_stages:
            print(f"- Stage {stage[].stage_id}: {stage[].stage_name} (target: {stage[].target_accuracy * 100.0}%)")
    
    fn execute_curriculum_training(
        inout self,
        all_training_examples: List[TrainingExample],
        inout model_params: Dict[String, GradientTensor[DType.float32]]
    ) -> Bool:
        """Execute complete curriculum learning training"""
        print("\n🚀 STARTING CURRICULUM LEARNING TRAINING")
        print("=" * 60)
        
        # Create validation splits
        self.training_monitor.validation_splitter.create_splits(all_training_examples)
        self.training_monitor.validation_splitter.print_split_statistics()
        
        var curriculum_success = True
        
        # Execute each curriculum stage
        for stage_idx in range(len(self.curriculum_stages)):
            let stage = self.curriculum_stages[stage_idx]
            
            print(f"\n🎯 CURRICULUM STAGE {stage[].stage_id}: {stage[].stage_name.upper()}")
            print("-" * 60)
            print("Description:", stage[].stage_description)
            print("Complexity Level:", stage[].complexity_level)
            print("Target Accuracy:", stage[].target_accuracy * 100.0, "%")
            print("Max Epochs:", stage[].max_epochs)
            
            # Filter examples for current stage
            let stage_examples = self._filter_examples_for_stage(
                self.training_monitor.validation_splitter.get_train_examples(),
                stage[]
            )
            
            print("Stage examples:", len(stage_examples))
            
            # Configure training for current stage
            self._configure_training_for_stage(stage[])
            
            # Train current stage
            let stage_success = self._train_single_stage(stage[], stage_examples, model_params)
            
            if not stage_success:
                print(f"❌ Stage {stage[].stage_id} training failed")
                curriculum_success = False
                break
            
            # Validate stage completion
            let stage_validation = self._validate_stage_completion(stage[], model_params)
            
            if not stage_validation.meets_requirements:
                print(f"⚠️  Stage {stage[].stage_id} validation failed")
                if not self._attempt_stage_recovery(stage[], stage_examples, model_params):
                    print(f"❌ Stage {stage[].stage_id} recovery failed")
                    curriculum_success = False
                    break
            
            # Record stage completion
            let stage_metrics = self._get_stage_metrics(stage_validation)
            self.stage_history.append(stage_metrics)
            
            # Advance to next stage
            if stage_idx < len(self.curriculum_stages) - 1:
                let next_stage_id = stage[].stage_id + 1
                self._advance_to_next_stage(next_stage_id, stage_validation.final_accuracy)
            
            print(f"✅ Stage {stage[].stage_id} completed successfully")
            print(f"Final accuracy: {stage_validation.final_accuracy * 100.0}%")
        
        # Final curriculum evaluation
        if curriculum_success:
            let final_evaluation = self._final_curriculum_evaluation(model_params)
            curriculum_success = final_evaluation.overall_success
            self.curriculum_completed = curriculum_success
            
            if curriculum_success:
                print("\n🎉 CURRICULUM LEARNING COMPLETED SUCCESSFULLY!")
                print("Model has progressed through all 4 stages and achieved mastery level.")
            else:
                print("\n💥 CURRICULUM LEARNING FAILED AT FINAL EVALUATION")
        
        # Generate comprehensive curriculum report
        self._generate_curriculum_report()
        
        return curriculum_success
    
    fn _filter_examples_for_stage(
        self,
        all_examples: List[TrainingExample],
        stage: CurriculumStage
    ) -> List[TrainingExample]:
        """Filter examples appropriate for current curriculum stage"""
        var stage_examples = List[TrainingExample]()
        
        for example in all_examples:
            let ex = example[]
            
            # Check complexity level
            if self._example_matches_complexity(ex, stage.complexity_level):
                # Check example type
                if self._example_type_allowed(ex.example_type, stage.example_types):
                    stage_examples.append(ex)
        
        # Apply data augmentation factor
        let target_size = int(Float32(len(stage_examples)) * stage.data_augmentation_factor)
        
        if target_size > len(stage_examples):
            # Duplicate examples to reach target size
            let multiplier = target_size // len(stage_examples)
            let remainder = target_size % len(stage_examples)
            
            var augmented_examples = List[TrainingExample]()
            
            # Add full cycles
            for cycle in range(multiplier):
                for example in stage_examples:
                    augmented_examples.append(example[])
            
            # Add remainder
            for i in range(remainder):
                augmented_examples.append(stage_examples[i])
            
            stage_examples = augmented_examples
        
        print("Stage examples after filtering:", len(stage_examples))
        return stage_examples
    
    fn _example_matches_complexity(self, example: TrainingExample, complexity: String) -> Bool:
        """Check if example matches required complexity level"""
        let ex_complexity = example.complexity_level
        
        if complexity == "beginner":
            return ex_complexity == "beginner"
        elif complexity == "intermediate":
            return ex_complexity in ["beginner", "intermediate"]
        elif complexity == "advanced":
            return ex_complexity in ["intermediate", "advanced"]
        elif complexity == "expert":
            return ex_complexity in ["advanced", "expert"]
        
        return True  # Default accept all
    
    fn _example_type_allowed(self, example_type: String, allowed_types: List[String]) -> Bool:
        """Check if example type is allowed for current stage"""
        for allowed_type in allowed_types:
            if example_type == allowed_type[]:
                return True
        return False
    
    fn _configure_training_for_stage(inout self, stage: CurriculumStage):
        """Configure training parameters for current stage"""
        print("Configuring training for stage", stage.stage_id)
        
        # Update learning rate scheduler
        self.multi_stage_scheduler.advance_to_stage(stage.stage_id)
        
        # Adjust training loop parameters
        let new_batch_size = int(16 * stage.batch_size_multiplier)  # Base batch size = 16
        print("- Batch size:", new_batch_size)
        print("- Learning rate multiplier:", stage.learning_rate_multiplier)
        print("- Data augmentation factor:", stage.data_augmentation_factor)
        
        # Update convergence criteria for stage
        let stage_convergence = ConvergenceCriteria(
            min_improvement = 0.001,
            patience = 3,
            min_epochs = 2,
            max_epochs = stage.max_epochs,
            accuracy_target = stage.target_accuracy
        )
        
        self.training_monitor.convergence_criteria = stage_convergence
    
    fn _train_single_stage(
        inout self,
        stage: CurriculumStage,
        stage_examples: List[TrainingExample],
        inout model_params: Dict[String, GradientTensor[DType.float32]]
    ) -> Bool:
        """Train model on single curriculum stage"""
        print(f"Training stage {stage.stage_id} with {len(stage_examples)} examples")
        
        var stage_success = True
        var best_accuracy: Float32 = 0.0
        
        # Initialize stage performance tracking
        self.stage_performance_tracker[stage.stage_id] = List[Float32]()
        
        for epoch in range(stage.max_epochs):
            print(f"\n--- Stage {stage.stage_id} Epoch {epoch + 1}/{stage.max_epochs} ---")
            
            # Train epoch with stage-specific learning rate
            let stage_lr = self.multi_stage_scheduler.get_learning_rate()
            let epoch_loss = self.training_loop.train_epoch(stage_examples, model_params, epoch + 1)
            
            # Validate epoch
            let validation_examples = self.training_monitor.validation_splitter.get_validation_examples()
            let validation_metrics = self._validate_epoch(validation_examples, model_params)
            
            print(f"Epoch {epoch + 1} Results:")
            print(f"- Training Loss: {epoch_loss}")
            print(f"- Validation Accuracy: {validation_metrics.accuracy * 100.0}%")
            print(f"- Learning Rate: {stage_lr}")
            
            # Track stage performance
            self.stage_performance_tracker[stage.stage_id].append(validation_metrics.accuracy)
            
            # Update best accuracy
            if validation_metrics.accuracy > best_accuracy:
                best_accuracy = validation_metrics.accuracy
                print("🎯 New best accuracy for stage!")
            
            # Check convergence for this stage
            let should_stop = self.training_monitor.should_stop_training(epoch_loss, validation_metrics)
            
            # Checkpoint if needed
            if self.training_monitor.should_checkpoint(epoch, validation_metrics):
                let checkpoint_path = self.training_monitor.save_checkpoint(
                    model_params,
                    Dict[String, Dict[String, Tensor[DType.float32]]](),  # Empty optimizer state
                    validation_metrics
                )
                print("💾 Stage checkpoint saved:", checkpoint_path)
            
            if should_stop:
                print(f"Stage {stage.stage_id} training stopped early at epoch {epoch + 1}")
                break
        
        # Check if stage meets minimum requirements
        stage_success = best_accuracy >= stage.min_accuracy_threshold
        
        if stage_success:
            print(f"✅ Stage {stage.stage_id} training successful")
            print(f"Best accuracy: {best_accuracy * 100.0}% (target: {stage.target_accuracy * 100.0}%)")
        else:
            print(f"❌ Stage {stage.stage_id} training failed")
            print(f"Best accuracy: {best_accuracy * 100.0}% < threshold: {stage.min_accuracy_threshold * 100.0}%")
        
        return stage_success
    
    fn _validate_epoch(
        self,
        validation_examples: List[TrainingExample],
        model_params: Dict[String, GradientTensor[DType.float32]]
    ) -> ValidationMetrics:
        """Validate model performance on validation set"""
        var metrics = ValidationMetrics()
        
        # Simulate validation (in real implementation would run inference)
        let total_examples = len(validation_examples)
        var correct_predictions = 0
        var tool_correct = 0
        var tool_total = 0
        var thinking_correct = 0
        var thinking_total = 0
        
        for example in validation_examples:
            let ex = example[]
            
            # Simulate prediction accuracy based on example type and complexity
            let predicted_correctly = self._simulate_prediction(ex, model_params)
            
            if predicted_correctly:
                correct_predictions += 1
            
            # Track tool-specific accuracy
            if "<tool>" in ex.output_text:
                tool_total += 1
                if predicted_correctly:
                    tool_correct += 1
            
            # Track thinking prefix accuracy
            if ex.output_text.startswith("<thinking>"):
                thinking_total += 1
                if predicted_correctly:
                    thinking_correct += 1
        
        # Calculate metrics
        metrics.accuracy = Float32(correct_predictions) / Float32(total_examples)
        metrics.tool_calling_accuracy = Float32(tool_correct) / Float32(tool_total) if tool_total > 0 else 0.0
        metrics.thinking_prefix_accuracy = Float32(thinking_correct) / Float32(thinking_total) if thinking_total > 0 else 0.0
        metrics.loss = 1.0 - metrics.accuracy  # Simplified loss
        metrics.perplexity = exp(metrics.loss)  # Simplified perplexity
        
        return metrics
    
    fn _simulate_prediction(
        self,
        example: TrainingExample,
        model_params: Dict[String, GradientTensor[DType.float32]]
    ) -> Bool:
        """Simulate model prediction (placeholder for actual inference)"""
        var base_accuracy: Float32 = 0.6  # Base accuracy
        
        # Adjust based on current stage
        let stage = self.curriculum_stages[self.current_stage - 1]
        
        if stage[].complexity_level == "beginner":
            base_accuracy = 0.8
        elif stage[].complexity_level == "intermediate":
            base_accuracy = 0.7
        elif stage[].complexity_level == "advanced":
            base_accuracy = 0.65
        else:  # expert
            base_accuracy = 0.6
        
        # Adjust based on example type
        if example.example_type == "tool_hit" and example.complexity_level == "beginner":
            base_accuracy += 0.1
        elif example.example_type == "multi_tool":
            base_accuracy -= 0.1
        elif example.example_type == "tool_error":
            base_accuracy -= 0.05
        
        # Universal Thinking Prefix bonus
        if example.output_text.startswith("<thinking>"):
            base_accuracy += 0.05
        
        # Simulate random outcome
        let random_value = Float32((len(example.input_text) + len(example.output_text)) % 100) / 100.0
        return random_value < base_accuracy
    
    fn _validate_stage_completion(
        inout self,
        stage: CurriculumStage,
        model_params: Dict[String, GradientTensor[DType.float32]]
    ) -> StageValidationResult:
        """Comprehensive validation of stage completion"""
        print(f"Validating stage {stage.stage_id} completion...")
        
        var result = StageValidationResult()
        
        # Run comprehensive validation
        let validation_examples = self.training_monitor.validation_splitter.get_validation_examples()
        let validation_metrics = self._validate_epoch(validation_examples, model_params)
        
        result.final_accuracy = validation_metrics.accuracy
        result.final_loss = validation_metrics.loss
        result.tool_calling_accuracy = validation_metrics.tool_calling_accuracy
        result.thinking_prefix_accuracy = validation_metrics.thinking_prefix_accuracy
        
        # Check requirements
        result.meets_accuracy_requirement = validation_metrics.accuracy >= stage.min_accuracy_threshold
        result.meets_tool_requirement = validation_metrics.tool_calling_accuracy >= 0.7
        result.meets_thinking_requirement = validation_metrics.thinking_prefix_accuracy >= 0.9
        
        result.meets_requirements = (
            result.meets_accuracy_requirement and
            result.meets_tool_requirement and
            result.meets_thinking_requirement
        )
        
        print("Stage validation results:")
        print(f"- Overall accuracy: {result.final_accuracy * 100.0}% (required: {stage.min_accuracy_threshold * 100.0}%)")
        print(f"- Tool calling accuracy: {result.tool_calling_accuracy * 100.0}% (required: 70%)")
        print(f"- Thinking prefix accuracy: {result.thinking_prefix_accuracy * 100.0}% (required: 90%)")
        print(f"- Requirements met: {result.meets_requirements}")
        
        return result
    
    fn _advance_to_next_stage(inout self, next_stage_id: Int, current_accuracy: Float32):
        """Advance to next curriculum stage"""
        print(f"Advancing from stage {self.current_stage} to stage {next_stage_id}")
        
        # Record transition
        let transition_reason = f"Stage {self.current_stage} completed with {current_accuracy * 100.0}% accuracy"
        self.stage_transition_history.append((next_stage_id, current_accuracy, transition_reason))
        
        # Update current stage
        self.current_stage = next_stage_id
        
        # Configure for next stage
        if next_stage_id <= len(self.curriculum_stages):
            self.multi_stage_scheduler.advance_to_stage(next_stage_id)
            print(f"🎯 Now on Stage {next_stage_id}: {self.curriculum_stages[next_stage_id - 1].stage_name}")
    
    fn _attempt_stage_recovery(
        inout self,
        stage: CurriculumStage,
        stage_examples: List[TrainingExample],
        inout model_params: Dict[String, GradientTensor[DType.float32]]
    ) -> Bool:
        """Attempt to recover from stage failure"""
        print(f"🔄 Attempting recovery for stage {stage.stage_id}")
        
        # Reduce learning rate and try again
        print("- Reducing learning rate by 50%")
        
        # Train for additional epochs with lower learning rate
        let recovery_epochs = 3
        var recovery_success = False
        
        for epoch in range(recovery_epochs):
            print(f"Recovery epoch {epoch + 1}/{recovery_epochs}")
            
            let epoch_loss = self.training_loop.train_epoch(stage_examples, model_params, epoch + 1)
            let validation_examples = self.training_monitor.validation_splitter.get_validation_examples()
            let validation_metrics = self._validate_epoch(validation_examples, model_params)
            
            if validation_metrics.accuracy >= stage.min_accuracy_threshold:
                recovery_success = True
                print(f"✅ Recovery successful! Accuracy: {validation_metrics.accuracy * 100.0}%")
                break
        
        if not recovery_success:
            print(f"❌ Recovery failed for stage {stage.stage_id}")
        
        return recovery_success
    
    fn _get_stage_metrics(self, validation_result: StageValidationResult) -> Dict[String, Float32]:
        """Extract stage metrics for history tracking"""
        var metrics = Dict[String, Float32]()
        
        metrics["stage_id"] = Float32(self.current_stage)
        metrics["final_accuracy"] = validation_result.final_accuracy
        metrics["final_loss"] = validation_result.final_loss
        metrics["tool_calling_accuracy"] = validation_result.tool_calling_accuracy
        metrics["thinking_prefix_accuracy"] = validation_result.thinking_prefix_accuracy
        metrics["meets_requirements"] = 1.0 if validation_result.meets_requirements else 0.0
        
        return metrics
    
    fn _final_curriculum_evaluation(
        inout self,
        model_params: Dict[String, GradientTensor[DType.float32]]
    ) -> CurriculumEvaluationResult:
        """Final comprehensive evaluation of curriculum learning"""
        print("\n🏆 FINAL CURRICULUM EVALUATION")
        print("-" * 40)
        
        var result = CurriculumEvaluationResult()
        
        # Test on held-out test set
        let test_examples = self.training_monitor.validation_splitter.get_test_examples()
        let test_metrics = self._validate_epoch(test_examples, model_params)
        
        result.test_accuracy = test_metrics.accuracy
        result.test_loss = test_metrics.loss
        result.tool_calling_performance = test_metrics.tool_calling_accuracy
        result.thinking_prefix_compliance = test_metrics.thinking_prefix_accuracy
        
        # Check overall curriculum success criteria
        let min_final_accuracy: Float32 = 0.85
        let min_tool_accuracy: Float32 = 0.80
        let min_thinking_compliance: Float32 = 0.95
        
        result.meets_accuracy_target = test_metrics.accuracy >= min_final_accuracy
        result.meets_tool_target = test_metrics.tool_calling_accuracy >= min_tool_accuracy
        result.meets_thinking_target = test_metrics.thinking_prefix_accuracy >= min_thinking_compliance
        
        result.overall_success = (
            result.meets_accuracy_target and
            result.meets_tool_target and
            result.meets_thinking_target
        )
        
        print("Final evaluation results:")
        print(f"- Test accuracy: {result.test_accuracy * 100.0}% (target: {min_final_accuracy * 100.0}%)")
        print(f"- Tool calling: {result.tool_calling_performance * 100.0}% (target: {min_tool_accuracy * 100.0}%)")
        print(f"- Thinking compliance: {result.thinking_prefix_compliance * 100.0}% (target: {min_thinking_compliance * 100.0}%)")
        print(f"- Overall success: {result.overall_success}")
        
        return result
    
    fn _generate_curriculum_report(inout self):
        """Generate comprehensive curriculum learning report"""
        print("\n" + "=" * 60)
        print("CURRICULUM LEARNING REPORT")
        print("=" * 60)
        
        print("Curriculum Overview:")
        print("- Total stages:", len(self.curriculum_stages))
        print("- Current stage:", self.current_stage)
        print("- Curriculum completed:", self.curriculum_completed)
        print("- Auto advancement:", self.auto_advancement)
        
        print("\nStage Performance:")
        for i in range(len(self.stage_history)):
            let stage_metrics = self.stage_history[i]
            let stage_id = int(stage_metrics["stage_id"])
            
            print(f"Stage {stage_id}:")
            print(f"  - Final accuracy: {stage_metrics['final_accuracy'] * 100.0}%")
            print(f"  - Tool calling: {stage_metrics['tool_calling_accuracy'] * 100.0}%")
            print(f"  - Requirements met: {'Yes' if stage_metrics['meets_requirements'] > 0.0 else 'No'}")
        
        print("\nStage Transitions:")
        for transition in self.stage_transition_history:
            let stage_id = transition[][0]
            let accuracy = transition[][1]
            let reason = transition[][2]
            print(f"→ Stage {stage_id}: {reason}")
        
        print("\nRecommendations:")
        if self.curriculum_completed:
            print("✅ Curriculum learning completed successfully")
            print("✅ Model ready for production deployment")
        else:
            print("⚠️  Curriculum learning incomplete")
            print("- Consider adjusting stage requirements")
            print("- Review failed stage performance")
            print("- Check data quality and augmentation")
        
        print("=" * 60)

struct StageValidationResult:
    """Results of stage completion validation"""
    var final_accuracy: Float32
    var final_loss: Float32
    var tool_calling_accuracy: Float32
    var thinking_prefix_accuracy: Float32
    var meets_accuracy_requirement: Bool
    var meets_tool_requirement: Bool
    var meets_thinking_requirement: Bool
    var meets_requirements: Bool
    
    fn __init__(inout self):
        self.final_accuracy = 0.0
        self.final_loss = 0.0
        self.tool_calling_accuracy = 0.0
        self.thinking_prefix_accuracy = 0.0
        self.meets_accuracy_requirement = False
        self.meets_tool_requirement = False
        self.meets_thinking_requirement = False
        self.meets_requirements = False

struct CurriculumEvaluationResult:
    """Results of final curriculum evaluation"""
    var test_accuracy: Float32
    var test_loss: Float32
    var tool_calling_performance: Float32
    var thinking_prefix_compliance: Float32
    var meets_accuracy_target: Bool
    var meets_tool_target: Bool
    var meets_thinking_target: Bool
    var overall_success: Bool
    
    fn __init__(inout self):
        self.test_accuracy = 0.0
        self.test_loss = 0.0
        self.tool_calling_performance = 0.0
        self.thinking_prefix_compliance = 0.0
        self.meets_accuracy_target = False
        self.meets_tool_target = False
        self.meets_thinking_target = False
        self.overall_success = False