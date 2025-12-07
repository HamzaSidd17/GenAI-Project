# AI Interaction Log: Technical Prompts & Strategy

## Category 1: Project Architecture & Requirements
- "Brief me about the project 'AI-Generated Text Detection' and outline the specific tasks for Member 1 (Feature Engineering & Classical Models)."
- "Clarify the hardware requirements: does the feature extraction pipeline (specifically Perplexity calculation) require a GPU?"
- "Do I need an API key to use the GPT-2 model locally via Hugging Face transformers?"

## Category 2: Feature Engineering Pipeline
- "Provide Kaggle-compatible Python code for the Feature Engineering Pipeline to extract Perplexity, Readability metrics, and the Naturalness Score."
- "Update the feature extraction loop to include a checkpoint system that saves progress every 10,000 rows to prevent data loss."

## Category 3: Model Training & Benchmarking
- "Provide the code pipeline for Phase 2: Training & Benchmarking. It should train Logistic Regression, SVM, and XGBoost on the extracted features and include a 'Resume Training' feature."
- "Generate code to evaluate the trained models on the Adversarial dataset (`adversarial_scratch_set_cleaned.csv`) to test robustness."
- "Optimize the testing pipeline to load pre-calculated features from `adversarial_feature_matrix.csv` instead of re-calculating perplexity."

## Category 4: Robustness Strategy (Fine-Tuning)
- "I want to perform transfer learning: Fine-tune the models on a balanced set containing 30% of the adversarial data and original human data. Provide the code for this robust fine-tuning pipeline."
- "Generate code to evaluate the fine-tuned models on the held-out adversarial test set and visualize the 'Before vs. After' robustness improvement."

## Category 5: Analysis & Reporting
- "Analyze these results: What does the drop in accuracy on adversarial data indicate regarding the reliability of statistical features like Perplexity?"
- "Provide the LaTeX code for the 'Track 1: Feature-Based Classification' section of the report, incorporating the methodology, baseline results, and fine-tuning analysis."
- "Update the LaTeX code to include side-by-side visualizations of the Confusion Matrices and Feature Importance plots."